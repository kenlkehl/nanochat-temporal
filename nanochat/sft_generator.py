"""
Local LLM client for SFT bulk synthetic data generation.

Wraps one or more OpenAI-compatible endpoints (vLLM, SGLang, llama.cpp server,
Ollama-OpenAI-bridge, ...) using the openai SDK. Configure via env vars or
pass to LocalLLM(...) explicitly:

    OPENAI_BASE_URL=http://localhost:8000/v1   # single endpoint
    OPENAI_BASE_URLS=http://h0:8000/v1,http://h0:8001/v1,...  # pool (comma-separated)
    OPENAI_API_KEY=EMPTY                        # whatever the local server expects
    LOCAL_LLM_MODEL=google/gemma-4-31B-it       # model name the server exposes
    LOCAL_LLM_ENABLE_THINKING=1                 # forward chat_template_kwargs={"enable_thinking": True}
                                                # (needed for Gemma-4 / Qwen3 thinking mode)

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
import re
import json
import asyncio
import itertools
from typing import Optional

import openai
from openai import AsyncOpenAI


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


# Compiled once. Handles both the common `<think>...</think>` convention and
# Gemma-4's `<start_of_thinking>...<end_of_thinking>` style. Used as a fallback
# when vLLM's reasoning parser is not enabled server-side.
_THINK_PATTERNS = [
    re.compile(r"(?s)<think>.*?</think>\s*"),
    re.compile(r"(?s)<start_of_thinking>.*?<end_of_thinking>\s*"),
]


def _strip_thinking(text: str) -> str:
    """Remove thinking preamble from a model response.

    Tries vLLM's gemma4_utils.parse_thinking_output first (if importable), falls
    back to regex for common delimiters. Safe no-op when no delimiters are
    present (e.g. when the vLLM server already has --reasoning-parser enabled,
    which moves thinking into a separate `reasoning_content` field).

    Why strip for this pipeline (pre-1985 SFT data gen): (1) the student model's
    tokenizer has no special tokens for <|channel>/<|think|>, so keeping them
    would teach it to emit literal delimiters it can't close; (2) contamination
    control — CoT often references post-cutoff knowledge even when the final
    answer is clean, and leaving it in the assistant string both confuses the
    Claude filter and risks leaking modern facts into the student; (3) the
    thinking is Gemma-4 scratch work, not training signal we want to imitate.
    """
    if not text:
        return text
    try:
        from vllm.reasoning.gemma4_utils import parse_thinking_output  # type: ignore
        parsed = parse_thinking_output(text)
        content = parsed.get("content") if isinstance(parsed, dict) else None
        if content is not None:
            return content.strip()
    except Exception:
        pass
    out = text
    for pat in _THINK_PATTERNS:
        out = pat.sub("", out)
    return out.strip()


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
        enable_thinking: Optional[bool] = None,
    ):
        self.base_urls = _resolve_base_urls(base_urls, base_url)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("LOCAL_LLM_MODEL", "google/gemma-4-31B-it")
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        # enable_thinking: when True, pass chat_template_kwargs={"enable_thinking": True}
        # as extra_body to the OpenAI chat-completions call. vLLM forwards it to
        # tokenizer.apply_chat_template() server-side. Required for Gemma-4
        # reasoning and Qwen3 thinking mode. Auto-enabled via env
        # LOCAL_LLM_ENABLE_THINKING=1.
        if enable_thinking is None:
            enable_thinking = _env_truthy("LOCAL_LLM_ENABLE_THINKING")
        self.enable_thinking = bool(enable_thinking)
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
        max_tokens=20000,
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
                    if self.enable_thinking:
                        # vLLM forwards chat_template_kwargs through to
                        # tokenizer.apply_chat_template(); openai SDK hides any
                        # non-standard keys in `extra_body`.
                        kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": True}
                        }
                    response = await client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                if self.enable_thinking:
                    content = _strip_thinking(content)
                return content
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

    async def chat_json(self, messages, max_parse_attempts=3, **kwargs):
        """Like chat but parses the response as JSON. Strips markdown code fences.

        On JSON parse failure, re-prompts the model with its own failing output
        appended as an assistant turn plus a user "repair" turn quoting the
        parse error, up to max_parse_attempts times. Useful because enable_thinking
        occasionally leaks prose/markdown alongside the JSON payload even when
        response_format={"type": "json_object"} is set.
        """
        conversation = list(messages)
        last_text = ""
        last_exc: Optional[json.JSONDecodeError] = None
        for attempt in range(1, max_parse_attempts + 1):
            text = await self.chat(conversation, **kwargs)
            last_text = text
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
                if stripped.endswith("```"):
                    stripped = stripped.rsplit("```", 1)[0]
                stripped = stripped.strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError as e:
                last_exc = e
                if attempt == max_parse_attempts:
                    break
                print(
                    f"[LocalLLM] chat_json parse failure "
                    f"(attempt {attempt}/{max_parse_attempts}): {e.msg}. "
                    f"Asking model to repair."
                )
                conversation = conversation + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as JSON. "
                            f"Parser error: {e.msg} (line {e.lineno}, col {e.colno}). "
                            "Return a single valid JSON object only — no prose, "
                            "no markdown fences, no commentary. Use the same "
                            "schema the original instruction asked for."
                        ),
                    },
                ]
        excerpt = last_text[:500].replace("\n", " ")
        raise json.JSONDecodeError(
            f"Failed to parse JSON after {max_parse_attempts} attempts. "
            f"Last output excerpt: {excerpt!r}",
            last_text or "",
            0,
        ) from last_exc

    async def close(self) -> None:
        for c in self._clients:
            try:
                await c.close()
            except Exception:
                pass
