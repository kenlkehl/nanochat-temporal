"""
Local LLM client for SFT bulk synthetic data generation.

Wraps one or more OpenAI-compatible endpoints (vLLM, SGLang, llama.cpp server,
Ollama-OpenAI-bridge, ...) using the openai SDK. Configure via env vars or
pass to LocalLLM(...) explicitly:

    OPENAI_BASE_URL=http://localhost:8000/v1   # single endpoint
    OPENAI_BASE_URLS=http://h0:8000/v1,http://h0:8001/v1,...  # pool (comma-separated)
    OPENAI_API_KEY=EMPTY                        # whatever the local server expects
    LOCAL_LLM_MODEL=Qwen/Qwen3-32B-Instruct     # model name the server exposes

If `OPENAI_BASE_URLS` is set (or a list is passed to `LocalLLM(base_urls=...)`),
requests are load-balanced round-robin across the listed endpoints, with an
independent `max_concurrency` semaphore *per server*. This lets the caller drive
one vLLM per GPU. Auto-launching such a pool is handled by
`nanochat.vllm_launcher.VLLMPool`; see that module's docstring.

Recommended local models (none pinned):
- Qwen3-32B-Instruct       - good quality, JSON mode supported by vLLM
- Llama-3.3-70B-Instruct   - better quality, more VRAM
- Llama-3.1-8B-Instruct    - fastest, lowest quality
- DeepSeek-V3              - best reasoning if available

Usage:

    import asyncio
    from nanochat.sft_generator import LocalLLM

    async def main():
        llm = LocalLLM()
        text = await llm.chat([{"role": "user", "content": "Say hi"}])
        print(text)

    asyncio.run(main())
"""

from __future__ import annotations

import os
import json
import asyncio
import itertools
from typing import Optional

import openai
from openai import AsyncOpenAI


def _resolve_base_urls(
    base_urls: Optional[list[str]],
    base_url: Optional[str],
) -> list[str]:
    if base_urls:
        return [u.strip() for u in base_urls if u and u.strip()]
    env_multi = os.environ.get("OPENAI_BASE_URLS")
    if env_multi:
        urls = [u.strip() for u in env_multi.split(",") if u.strip()]
        if urls:
            return urls
    single = base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
    return [single]


class LocalLLM:
    def __init__(
        self,
        base_url: Optional[str] = None,
        base_urls: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_concurrency: int = 16,
        timeout: float = 120.0,
    ):
        self.base_urls = _resolve_base_urls(base_urls, base_url)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-32B-Instruct")
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self._clients: list[AsyncOpenAI] = [
            AsyncOpenAI(base_url=u, api_key=self.api_key, timeout=timeout)
            for u in self.base_urls
        ]
        self._semaphores: list[asyncio.Semaphore] = [
            asyncio.Semaphore(max_concurrency) for _ in self.base_urls
        ]
        self._rr = itertools.count()

    @property
    def num_backends(self) -> int:
        return len(self._clients)

    @property
    def base_url(self) -> str:
        """Back-compat alias: first base_url (meaningful when num_backends == 1)."""
        return self.base_urls[0]

    def _pick_backend(self) -> tuple[AsyncOpenAI, asyncio.Semaphore, str]:
        idx = next(self._rr) % len(self._clients)
        return self._clients[idx], self._semaphores[idx], self.base_urls[idx]

    async def chat(
        self,
        messages,
        max_tokens=2048,
        temperature=0.9,
        response_format=None,
        max_attempts=5,
    ):
        last_exc = None
        last_url = None
        for attempt in range(1, max_attempts + 1):
            client, sem, url = self._pick_backend()
            last_url = url
            try:
                async with sem:
                    kwargs = dict(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if response_format is not None:
                        kwargs["response_format"] = response_format
                    response = await client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except (
                openai.APIConnectionError,
                openai.APIStatusError,
                openai.APITimeoutError,
                asyncio.TimeoutError,
            ) as e:
                last_exc = e
                if attempt == max_attempts:
                    raise
                wait = min(2 ** attempt, 60)
                print(
                    f"[LocalLLM] attempt {attempt}/{max_attempts} to {url} failed "
                    f"({type(e).__name__}); retrying in {wait}s"
                )
                await asyncio.sleep(wait)
        if last_exc:
            raise last_exc
        # unreachable
        _ = last_url
        return ""

    async def chat_json(self, messages, **kwargs):
        """Like chat but parses the response as JSON. Strips markdown code fences if present."""
        text = await self.chat(messages, **kwargs)
        text = text.strip()
        if text.startswith("```"):
            # ```json\n...\n``` or ```\n...\n```
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()
        return json.loads(text)

    async def close(self) -> None:
        for c in self._clients:
            try:
                await c.close()
            except Exception:
                pass
