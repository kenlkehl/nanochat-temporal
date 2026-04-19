"""
Pre-1985 contamination filter using Anthropic Claude.

Cascade:
1. Cheap rule-based prefilter (regex year + curated keyword list) — flags route to Sonnet.
2. Haiku 4.5 — default judge for unflagged samples.
3. Sonnet 4.6 — escalation for prefilter-flagged samples, Haiku-UNSURE cases,
   and a random audit on Haiku-passed samples (default 5%).

Verdicts are cached in an aiosqlite DB keyed on sha256 of the conversation JSON,
so reruns are free and the SFT data builder is fully resumable.

Backends (set ANTHROPIC_BACKEND or pass `backend=...`):
  - "api"    — public Anthropic API. Requires ANTHROPIC_API_KEY.
  - "vertex" — Google Cloud Vertex AI. Requires Google ADC
               (gcloud auth application-default login, GOOGLE_APPLICATION_CREDENTIALS,
               or GCP metadata server) plus ANTHROPIC_VERTEX_PROJECT_ID and
               ANTHROPIC_VERTEX_REGION (or CLOUD_ML_REGION).

Per-backend default model IDs can be overridden via ANTHROPIC_HAIKU_MODEL /
ANTHROPIC_SONNET_MODEL env vars or constructor args. Vertex sometimes requires
an "@<release-date>" suffix in the model ID; defaults are sensible but version-
sensitive — override if your Vertex deployment uses a different convention.

Usage:

    import asyncio
    from nanochat.contamination_filter import ContaminationFilter

    async def main():
        # Public API:
        cf = ContaminationFilter()
        # Or Vertex:
        # cf = ContaminationFilter(backend="vertex",
        #                          vertex_project_id="my-gcp-project",
        #                          vertex_region="us-east5")
        safe, reason = await cf.check([
            {"role": "user", "content": "What is PCR?"},
            {"role": "assistant", "content": "PCR is polymerase chain reaction..."},
        ])
        print(safe, reason)
        await cf.close()

    asyncio.run(main())
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import hashlib
import random
from typing import Optional

import aiosqlite
import anthropic
from anthropic import AsyncAnthropic

# Default model IDs.
#
# - Anthropic API ("api" backend): per-system info, Haiku 4.5 = "claude-haiku-4-5-20251001",
#   Sonnet 4.6 = "claude-sonnet-4-6".
# - Vertex backend: Vertex sometimes requires an "@<release-date>" suffix in the model ID
#   (e.g. "claude-haiku-4-5@20251001"). If your Vertex deployment needs that form, override
#   via the ANTHROPIC_HAIKU_MODEL / ANTHROPIC_SONNET_MODEL env vars or the haiku_model /
#   sonnet_model constructor args.
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"
VERTEX_HAIKU_MODEL = "claude-haiku-4-5@20251001"
VERTEX_SONNET_MODEL = "claude-sonnet-4-6"

# Years 1985-2199 — anything in this range outside code is suspicious
YEAR_REGEX = re.compile(r"\b(?:198[5-9]|199\d|20\d{2}|21\d{2})\b")

# Curated post-1985 indicators. Hits don't auto-reject — they route to Sonnet.
POST_1985_KEYWORDS = [
    # Tech products / companies founded post-1984
    "iPhone", "iPad", "Android", "smartphone", "smart phone",
    "ChatGPT", "GPT-3", "GPT-4", "GPT-5", "Claude", "Gemini", "Copilot",
    "Wikipedia", "Google", "Amazon.com", "Facebook", "Meta", "Twitter",
    "TikTok", "Instagram", "YouTube", "Netflix", "Spotify", "Uber", "Airbnb",
    "World Wide Web", "the Web", "web browser", "Mosaic", "Netscape",
    "Internet Explorer", "Firefox", "Chrome browser", "Safari browser",
    "Bitcoin", "blockchain", "cryptocurrency", "NFT",
    # Politics / events
    "Obama", "Biden", "Trump", "Brexit",
    "9/11", "September 11", "World Trade Center attack",
    "fall of the Berlin Wall", "Berlin Wall fell",
    "dissolution of the Soviet Union", "collapse of the USSR",
    "Iraq War", "Afghanistan war", "Arab Spring",
    "COVID", "SARS-CoV", "coronavirus pandemic", "Omicron", "Delta variant",
    # Science discoveries / programs post-1984
    "CRISPR", "Cas9", "AlphaFold",
    "Higgs boson confirmed", "Higgs boson discovery", "Higgs discovery",
    "exoplanet",
    "Dolly the sheep", "induced pluripotent",
    "Human Genome Project",
    "gravitational wave detected", "LIGO detection",
    "transformer architecture", "transformer model", "self-attention",
    "deep learning revolution", "large language model", "LLM",
    "IPCC report", "Kyoto Protocol", "Paris Agreement",
]

SYSTEM_PROMPT = """You are a strict historical contamination filter. The candidate is training data for an AI that must know NOTHING from after January 1, 1985.

POST-1985 CONTENT (UNSAFE):
- Specific events, people, works, technologies, products, or companies that emerged after 1984 (Berlin Wall fall 1989, iPhone, COVID, ChatGPT, Obama presidency, 9/11, Wikipedia, the World Wide Web, Dolly the sheep, Human Genome Project results, CRISPR, etc.)
- Scientific discoveries first widely accepted/published after 1984 (CRISPR, Higgs boson confirmation, exoplanets, AlphaFold, transformer architecture, dark energy as a discovered phenomenon, modern climate-science consensus, etc.)
- Companies founded 1985+ (Sun Microsystems 1982 OK; Cisco 1984 OK; Amazon 1994 UNSAFE; Google 1998 UNSAFE)

PRE-1985 CONTENT (SAFE — explicitly allowed):
- Apple (1976), Microsoft (1975), early personal computers, MS-DOS, early Unix, the ARPANET
- AIDS as a recognized disease (1981), HIV identified (1983), PCR (1983), prion hypothesis (1982)
- Recombinant DNA (1973+), early oncogenes, basic molecular biology of the early 1980s
- Anything in classical literature, history through 1984, and pre-1985 mainstream science

NOT CONTAMINATION (allowed even though modern):
- Modern Python syntax, NumPy, pandas, matplotlib, scikit-learn, PyTorch — these are TOOLS the assistant uses, not knowledge being conveyed
- Statistical methods and ML algorithms as math (linear regression, neural networks, k-means, PCA, Bayes, Monte Carlo, backpropagation) — these are math, all knowable pre-1985
- Modern English usage and code formatting

OUTPUT FORMAT:
First write a 1-2 sentence justification. Then on a new line emit exactly one token: SAFE, UNSAFE, or UNSURE."""

VERDICT_REGEX = re.compile(r"\b(SAFE|UNSAFE|UNSURE)\b")


def _conversation_hash(conversation):
    payload = json.dumps(conversation, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _format_conversation(conversation):
    """Format a conversation into a readable string for the filter prompt."""
    lines = []
    for msg in conversation:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        else:
            text = str(content)
        lines.append(f"### {role.upper()}\n{text}")
    return "\n\n".join(lines)


def cheap_prefilter(conversation):
    """Return True if the conversation contains a year 1985+ outside code blocks
    or any post-1985 keyword. Doesn't auto-reject — flagged samples route to Sonnet."""
    text = _format_conversation(conversation)
    text_no_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text_no_code = re.sub(r"`[^`]*`", " ", text_no_code)
    if YEAR_REGEX.search(text_no_code):
        return True
    text_lower = text_no_code.lower()
    for kw in POST_1985_KEYWORDS:
        if kw.lower() in text_lower:
            return True
    return False


def parse_verdict(response_text):
    """Find the last SAFE/UNSAFE/UNSURE token in the response."""
    matches = VERDICT_REGEX.findall(response_text)
    if not matches:
        return "UNSURE"
    return matches[-1]


class ContaminationFilter:
    def __init__(
        self,
        cache_path: Optional[str] = None,
        max_concurrency: int = 20,
        sonnet_audit_rate: float = 0.05,
        # Anthropic API auth (api backend)
        api_key: Optional[str] = None,
        # Backend selection
        backend: Optional[str] = None,            # "api" (default) | "vertex"
        # Model overrides (per-backend defaults if not set)
        haiku_model: Optional[str] = None,
        sonnet_model: Optional[str] = None,
        # Vertex-specific config (only used when backend == "vertex")
        vertex_project_id: Optional[str] = None,
        vertex_region: Optional[str] = None,
    ):
        from nanochat.common import get_base_dir
        self.cache_path = cache_path or os.path.join(get_base_dir(), "contam_cache.sqlite")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.sonnet_audit_rate = sonnet_audit_rate
        self._db = None
        self._db_lock = asyncio.Lock()
        self._audit_rng = random.Random(20260419)

        # Resolve backend
        backend = (backend or os.environ.get("ANTHROPIC_BACKEND") or "api").lower()
        if backend not in ("api", "vertex"):
            raise ValueError(f"Unknown backend {backend!r} (valid: 'api', 'vertex')")
        self.backend = backend

        # Resolve model IDs (env override > per-backend default)
        env_haiku = os.environ.get("ANTHROPIC_HAIKU_MODEL")
        env_sonnet = os.environ.get("ANTHROPIC_SONNET_MODEL")
        if backend == "vertex":
            self.haiku_model = haiku_model or env_haiku or VERTEX_HAIKU_MODEL
            self.sonnet_model = sonnet_model or env_sonnet or VERTEX_SONNET_MODEL
        else:
            self.haiku_model = haiku_model or env_haiku or HAIKU_MODEL
            self.sonnet_model = sonnet_model or env_sonnet or SONNET_MODEL

        # Build the client
        if backend == "vertex":
            from anthropic import AsyncAnthropicVertex
            project_id = (
                vertex_project_id
                or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
                or os.environ.get("CLOUD_ML_PROJECT_ID")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            region = (
                vertex_region
                or os.environ.get("ANTHROPIC_VERTEX_REGION")
                or os.environ.get("CLOUD_ML_REGION")
            )
            kwargs = {}
            if project_id:
                kwargs["project_id"] = project_id
            if region:
                kwargs["region"] = region
            # Auth is handled by google-auth (ADC, GOOGLE_APPLICATION_CREDENTIALS,
            # gcloud auth application-default login, or GCP metadata server).
            self.client = AsyncAnthropicVertex(**kwargs)
        else:
            client_kwargs = {"api_key": api_key} if api_key else {}
            self.client = AsyncAnthropic(**client_kwargs)

    async def _get_db(self):
        if self._db is None:
            async with self._db_lock:
                if self._db is None:
                    self._db = await aiosqlite.connect(self.cache_path)
                    await self._db.execute(
                        """CREATE TABLE IF NOT EXISTS contam_cache (
                            hash TEXT PRIMARY KEY,
                            safe INTEGER NOT NULL,
                            verdict TEXT NOT NULL,
                            reason TEXT NOT NULL,
                            model TEXT NOT NULL
                        )"""
                    )
                    await self._db.commit()
        return self._db

    async def _cache_get(self, key):
        db = await self._get_db()
        async with db.execute(
            "SELECT safe, verdict, reason FROM contam_cache WHERE hash = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        safe, verdict, reason = row
        return bool(safe), verdict, reason

    async def _cache_put(self, key, safe, verdict, reason, model):
        db = await self._get_db()
        await db.execute(
            "INSERT OR REPLACE INTO contam_cache (hash, safe, verdict, reason, model) VALUES (?, ?, ?, ?, ?)",
            (key, int(safe), verdict, reason, model),
        )
        await db.commit()

    async def _call_claude(self, model, conversation_text, max_attempts=5):
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with self.semaphore:
                    response = await self.client.messages.create(
                        model=model,
                        max_tokens=200,
                        system=SYSTEM_PROMPT,
                        messages=[
                            {"role": "user", "content": f"CANDIDATE CONVERSATION:\n\n{conversation_text}"}
                        ],
                    )
                if response.content:
                    return response.content[0].text
                return ""
            except (anthropic.APIConnectionError, anthropic.APIStatusError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt == max_attempts:
                    raise
                wait = min(2 ** attempt, 60)
                await asyncio.sleep(wait)
        if last_exc:
            raise last_exc
        return ""

    async def check(self, conversation):
        """
        Args:
            conversation: list of {"role": "user"|"assistant", "content": str|list}
        Returns:
            (safe: bool, reason: str)
        """
        key = _conversation_hash(conversation)
        cached = await self._cache_get(key)
        if cached is not None:
            return cached[0], cached[2]

        flagged = cheap_prefilter(conversation)
        text = _format_conversation(conversation)

        if flagged:
            response_text = await self._call_claude(self.sonnet_model, text)
            verdict = parse_verdict(response_text)
            model_used = self.sonnet_model
        else:
            response_text = await self._call_claude(self.haiku_model, text)
            verdict = parse_verdict(response_text)
            model_used = self.haiku_model
            if verdict == "UNSURE":
                response_text = await self._call_claude(self.sonnet_model, text)
                verdict = parse_verdict(response_text)
                model_used = self.sonnet_model
            elif verdict == "SAFE" and self._audit_rng.random() < self.sonnet_audit_rate:
                audit_text = await self._call_claude(self.sonnet_model, text)
                audit_verdict = parse_verdict(audit_text)
                if audit_verdict == "UNSAFE":
                    verdict = "UNSAFE"
                    response_text = audit_text
                    model_used = f"{self.haiku_model}+audit:{self.sonnet_model}"

        safe = verdict == "SAFE"  # UNSURE conservatively counts as unsafe
        await self._cache_put(key, safe, verdict, response_text.strip(), model_used)
        return safe, response_text.strip()

    async def close(self):
        if self._db is not None:
            await self._db.close()
            self._db = None
