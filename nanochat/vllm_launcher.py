"""
Subprocess lifecycle management for a pool of vLLM servers.

Launches one `vllm serve` process per GPU, each on its own port, and tears
them down on exit (SIGTERM, then SIGKILL after a timeout). Uses stdlib only
(no asyncio, no openai).

Typical use::

    from nanochat.vllm_launcher import VLLMPool

    with VLLMPool(model="google/gemma-4-31B-it", gpu_ids=[0, 1, 2, 3]) as pool:
        urls = pool.base_urls       # ["http://127.0.0.1:8000/v1", ...]
        # ... use urls with LocalLLM(base_urls=urls) ...
"""

from __future__ import annotations

import atexit
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from nanochat.common import get_base_dir


class VLLMStartupError(RuntimeError):
    pass


@dataclass
class VLLMServerSpec:
    gpu_id: int
    port: int
    model: str
    extra_args: list[str] = field(default_factory=list)
    base_url: str = ""

    def __post_init__(self):
        if not self.base_url:
            self.base_url = f"http://127.0.0.1:{self.port}/v1"


@dataclass
class VLLMServerHandle:
    spec: VLLMServerSpec
    proc: subprocess.Popen
    log_path: str
    pid: int


def parse_gpu_ids_arg(spec: str) -> list[int]:
    """Parse "0,1,2" or "0-3" or "0-1,4,6-7" into a deduped sorted list of ints."""
    if not spec:
        return []
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def wait_for_health(
    base_url: str,
    proc: subprocess.Popen,
    timeout: float,
    interval: float = 2.0,
) -> bool:
    """Poll `{base_url}/models` until a 200 response or until `proc` dies or timeout."""
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
            pass
        time.sleep(interval)
    return False


class VLLMPool:
    def __init__(
        self,
        model: str,
        gpu_ids: list[int],
        start_port: int = 8000,
        extra_args: Optional[list[str]] = None,
        log_dir: Optional[str] = None,
        startup_timeout: float = 600.0,
        shutdown_timeout: float = 30.0,
        health_poll_interval: float = 2.0,
    ):
        if not gpu_ids:
            raise ValueError("VLLMPool requires at least one GPU id")
        self.model = model
        self.gpu_ids = list(gpu_ids)
        self.start_port = start_port
        self.extra_args = list(extra_args) if extra_args else []
        self.log_dir = log_dir or os.path.join(get_base_dir(), "vllm_logs")
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.health_poll_interval = health_poll_interval
        self._handles: list[VLLMServerHandle] = []
        self._atexit_registered = False

    @property
    def base_urls(self) -> list[str]:
        return [h.spec.base_url for h in self._handles]

    def start(self) -> list[VLLMServerHandle]:
        if self._handles:
            return self._handles
        os.makedirs(self.log_dir, exist_ok=True)

        specs = [
            VLLMServerSpec(
                gpu_id=g,
                port=self.start_port + i,
                model=self.model,
                extra_args=self.extra_args,
            )
            for i, g in enumerate(self.gpu_ids)
        ]

        handles: list[VLLMServerHandle] = []
        try:
            for spec in specs:
                handles.append(self._launch_one(spec))
            self._handles = handles

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

            self._wait_all_healthy()
        except BaseException:
            self._handles = handles
            self.stop()
            raise
        return self._handles

    def _launch_one(self, spec: VLLMServerSpec) -> VLLMServerHandle:
        log_path = os.path.join(
            self.log_dir, f"vllm_gpu{spec.gpu_id}_port{spec.port}.log"
        )
        log_fh = open(log_path, "ab", buffering=0)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(spec.gpu_id)}
        cmd = [
            "vllm", "serve", spec.model,
            "--host", "0.0.0.0",
            "--port", str(spec.port),
            *spec.extra_args,
        ]
        print(
            f"[vllm-launcher] starting GPU {spec.gpu_id} port {spec.port}: "
            f"{' '.join(shlex.quote(c) for c in cmd)}  (log: {log_path})",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return VLLMServerHandle(spec=spec, proc=proc, log_path=log_path, pid=proc.pid)

    def _wait_all_healthy(self) -> None:
        with ThreadPoolExecutor(max_workers=len(self._handles)) as ex:
            futures = {
                ex.submit(
                    wait_for_health,
                    h.spec.base_url,
                    h.proc,
                    self.startup_timeout,
                    self.health_poll_interval,
                ): h
                for h in self._handles
            }
            for fut in as_completed(futures):
                h = futures[fut]
                ok = False
                try:
                    ok = fut.result()
                except Exception:
                    ok = False
                if not ok:
                    rc = h.proc.poll()
                    raise VLLMStartupError(
                        f"vLLM server on GPU {h.spec.gpu_id} port {h.spec.port} "
                        f"did not become healthy within {self.startup_timeout}s "
                        f"(exit={rc}; see log {h.log_path})"
                    )
                print(
                    f"[vllm-launcher] healthy: {h.spec.base_url} "
                    f"(GPU {h.spec.gpu_id}, pid {h.pid})",
                    flush=True,
                )

    def stop(self) -> None:
        if not self._handles:
            return
        handles, self._handles = self._handles, []
        for h in handles:
            try:
                os.killpg(h.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        deadline = time.monotonic() + self.shutdown_timeout
        for h in handles:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                h.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(h.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    pass
                try:
                    h.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass

    def __enter__(self) -> "VLLMPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _cli() -> None:
    """Minimal CLI for manual smoke testing: launches a pool and waits for Enter / SIGINT."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch a pool of vLLM servers, one per GPU.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True, help='e.g. "0,1,2,3" or "0-3"')
    parser.add_argument("--start-port", type=int, default=8000)
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

    pool = VLLMPool(
        model=args.model,
        gpu_ids=parse_gpu_ids_arg(args.gpus),
        start_port=args.start_port,
        extra_args=shlex.split(args.extra_args),
        startup_timeout=args.startup_timeout,
    )
    try:
        pool.start()
        print("\nReady. Export this to use with LocalLLM:", file=sys.stderr)
        print(f"  export OPENAI_BASE_URLS={','.join(pool.base_urls)}", file=sys.stderr)
        print("\nPress Ctrl-C to stop the pool.", file=sys.stderr)
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        pool.stop()


if __name__ == "__main__":
    _cli()
