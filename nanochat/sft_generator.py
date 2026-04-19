"""
Local LLM client for SFT bulk synthetic data generation.

Wraps an OpenAI-compatible endpoint (vLLM, SGLang, llama.cpp server, Ollama-OpenAI-bridge, ...)
using the openai SDK. Configure via env vars or pass to LocalLLM(...) explicitly:

    OPENAI_BASE_URL=http://localhost:8000/v1   # endpoint URL
    OPENAI_API_KEY=EMPTY                        # whatever the local server expects
    LOCAL_LLM_MODEL=Qwen/Qwen3-32B-Instruct     # model name the server exposes

Recommended local models (none pinned):
- Qwen3-32B-Instruct       — good quality, JSON mode supported by vLLM
- Llama-3.3-70B-Instruct   — better quality, more VRAM
- Llama-3.1-8B-Instruct    — fastest, lowest quality
- DeepSeek-V3              — best reasoning if available

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
from typing import Optional

import openai
from openai import AsyncOpenAI


class LocalLLM:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_concurrency: int = 16,
        timeout: float = 120.0,
    ):
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-32B-Instruct")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def chat(
        self,
        messages,
        max_tokens=2048,
        temperature=0.9,
        response_format=None,
        max_attempts=5,
    ):
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with self.semaphore:
                    kwargs = dict(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if response_format is not None:
                        kwargs["response_format"] = response_format
                    response = await self.client.chat.completions.create(**kwargs)
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
                await asyncio.sleep(wait)
        if last_exc:
            raise last_exc
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
