"""
Pre-1985 contamination filter using the local vLLM-served LLM.

Every candidate conversation is judged by the same OpenAI-compatible local
server (e.g. vLLM serving google/gemma-4-31B-it) that the generator uses.
A cheap regex prefilter still runs as a fast pre-check so flagged samples
get logged, but there's no cheap-vs-strong routing — one model judges all.

Verdicts are cached in an aiosqlite DB keyed on sha256 of the conversation
JSON, so reruns are free and the SFT data builder is fully resumable. Old
Claude-era verdicts stay valid; semantics are identical.

Usage:

    import asyncio
    from nanochat.sft_generator import LocalLLM
    from nanochat.contamination_filter import ContaminationFilter

    async def main():
        llm = LocalLLM()
        cf = ContaminationFilter(llm=llm)
        safe, reason = await cf.check([
            {"role": "user", "content": "What is PCR?"},
            {"role": "assistant", "content": "PCR is polymerase chain reaction..."},
        ])
        print(safe, reason)
        await cf.close()
        await llm.close()

    asyncio.run(main())
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import hashlib
from typing import Optional, TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from nanochat.sft_generator import LocalLLM

# Years 1985-2199 — anything in this range outside code is suspicious
YEAR_REGEX = re.compile(r"\b(?:198[5-9]|199\d|20\d{2}|21\d{2})\b")

# Curated post-1985 indicators. Hits don't auto-reject — they're noted in the
# cached reason so failures are easier to audit.
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
    or any post-1985 keyword. Informational only — doesn't auto-reject."""
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
        llm: "LocalLLM",
        cache_path: Optional[str] = None,
        max_concurrency: int = 20,
    ):
        from nanochat.common import get_base_dir
        self.llm = llm
        self.cache_path = cache_path or os.path.join(get_base_dir(), "contam_cache.sqlite")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._db = None
        self._db_lock = asyncio.Lock()

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

    async def _call_judge(self, conversation_text):
        async with self.semaphore:
            return await self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"CANDIDATE CONVERSATION:\n\n{conversation_text}"},
                ],
                max_tokens=300,
                temperature=0.0,
                enable_thinking=False,
            )

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

        text = _format_conversation(conversation)
        response_text = await self._call_judge(text)
        verdict = parse_verdict(response_text)

        safe = verdict == "SAFE"  # UNSURE conservatively counts as unsafe
        reason = response_text.strip()
        if cheap_prefilter(conversation):
            reason = "[prefilter:flagged] " + reason
        await self._cache_put(key, safe, verdict, reason, self.llm.model)
        return safe, reason

    async def close(self):
        if self._db is not None:
            await self._db.close()
            self._db = None
