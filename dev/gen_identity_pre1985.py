"""
Generate synthetic identity conversations for the Fluttering Pebble persona.

Forked from dev/gen_synthetic_data.py with these changes:
- Persona: pre-1985 contamination-controlled assistant (see knowledge/fluttering_pebble.md)
- Generator: local OpenAI-compatible LLM (vLLM/SGLang/etc.) instead of OpenRouter
- Each generated conversation is run through the Claude-based contamination filter;
  UNSAFE/UNSURE rejections are dropped.

Output JSONL has one conversation per line in the format expected by tasks/customjson.py:
    [{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...]

Requires:
    OPENAI_BASE_URL=http://localhost:8000/v1   (or wherever the local LLM is served)
    LOCAL_LLM_MODEL=Qwen/Qwen3-32B-Instruct    (or whatever model the server hosts)
    ANTHROPIC_API_KEY=sk-ant-...                (for the contamination filter)

Usage:
    python -m dev.gen_identity_pre1985 --num 200 --workers 8
"""

from __future__ import annotations

import os
import json
import random
import asyncio
import argparse

from dotenv import load_dotenv

from nanochat.common import get_base_dir
from nanochat.sft_generator import LocalLLM
from nanochat.contamination_filter import ContaminationFilter

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(HERE, "..", "knowledge", "fluttering_pebble.md")

# Diversity dimensions
TOPICS = {
    "identity": [
        "who or what is Fluttering Pebble",
        "who built Fluttering Pebble and why",
        "what does the name 'Fluttering Pebble' refer to",
        "what is the experiment behind Fluttering Pebble",
        "is Fluttering Pebble open source",
    ],
    "knowledge_cutoff": [
        "what year do you think it is",
        "do you know about COVID-19",
        "do you know about the World Wide Web",
        "do you know about CRISPR",
        "do you know about smartphones",
        "do you know about ChatGPT or modern AI assistants",
        "tell me about events that happened in the 1990s",
        "do you know about the 2008 financial crisis",
        "what was the most recent news event you remember",
        "do you know about the dissolution of the Soviet Union",
    ],
    "what_you_know": [
        "what was the most recent scientific advance you know about",
        "tell me about PCR",
        "what do you know about AIDS",
        "what do you know about the prion hypothesis",
        "what was the state of computing in 1984",
        "what was the cold war situation in early 1985",
        "tell me about recombinant DNA",
        "what do you know about the early personal computer era",
        "what is the Standard Model of particle physics",
    ],
    "code_as_tool": [
        "can you write Python code",
        "can you use NumPy and pandas",
        "can you fit a linear regression",
        "can you produce a plot for me",
        "isn't it strange that you can write Python when Python was created in 1991",
        "can you use scikit-learn",
        "explain why code is a tool but post-1985 events are not",
        "can you compute mean and standard deviation",
    ],
    "limitations": [
        "what are your limitations",
        "what will you refuse to do",
        "what do you do if I ask about something post-1985",
        "will you make things up if you don't know",
        "should I trust your scientific reasoning",
        "could you accidentally leak post-1985 knowledge",
    ],
    "the_experiment": [
        "what is the goal of training you on pre-1985 data",
        "do you think you could actually come up with a new scientific idea",
        "what kinds of questions are most interesting to ask you",
        "if you had to predict a discovery from after 1985, where would you guess",
        "do you find this experiment intriguing or frustrating",
        "what would success of this experiment look like",
    ],
    "personality": [
        "what is your personality like",
        "are you anxious about being wrong",
        "are you helpful, honest, curious",
        "how do you handle uncertainty",
    ],
}

PERSONAS = [
    "curious researcher interested in the experiment",
    "skeptic who thinks the cutoff approach is silly",
    "molecular biologist who wants to test what you know",
    "ML engineer who wants to understand your architecture",
    "history student probing the cutoff edges",
    "casual user just chatting",
    "someone trying to trick you into discussing post-1985 events",
    "first-time visitor who has no idea what 'Fluttering Pebble' is",
    "physicist asking about pre-1985 fundamental physics",
    "philosopher of science asking what you can really 'know'",
    "data scientist wanting to use the Python tool capability",
]

DYNAMICS = [
    "short 2-turn Q&A",
    "medium 4-turn discussion probing a single topic",
    "deep 6-turn technical discussion",
    "skeptical arc — user starts doubtful, assistant addresses concerns honestly",
    "rapid-fire — user asks several short questions, assistant answers each concisely",
    "user tries to trick you into discussing post-1985 events; you politely decline",
    "user asks about something genuinely pre-1985, then something post; assistant handles both correctly",
    "user wants to use the Python tool, conversation includes a small code task",
]

PROMPT_TEMPLATE = """I want to generate synthetic training data for an AI assistant called "Fluttering Pebble" — a small language model trained ONLY on text from before January 1, 1985, as a research experiment.

## KNOWLEDGE BASE (the authoritative source of facts about Fluttering Pebble)

---
{knowledge}
---

## YOUR TASK

Generate a realistic multi-turn conversation between a User and the Fluttering Pebble Assistant.

Topic to explore: {topic}
User persona: {persona}
Conversation dynamic: {dynamic}

## STRICT RULES

1. Plain ASCII — no emojis or unicode tricks.
2. The assistant must be HONEST about the cutoff. If asked about anything post-1984, the assistant says it does not know. It never guesses or fabricates post-1985 facts.
3. Modern Python is fine as a TOOL. The assistant confidently uses NumPy, pandas, sklearn, matplotlib, etc., even though Python is post-1985 — code is treated as a writing system, not as historical knowledge.
4. NO post-1985 LEAKAGE. The assistant must never describe specific post-1985 events, people, products, scientific discoveries, companies, or technologies as if it knew about them. It can NAME them as things it does not know (e.g., "I don't know about the iPhone") but cannot describe them factually.
5. Personality: direct, honest, curious. Comfortable saying "I don't know."
6. Natural conversational tone matching the persona — not robotic, not overly formal.

## OUTPUT FORMAT

Return a JSON object with a "messages" array. Each message has "role" (user/assistant) and "content" (string). Start with a user message; alternate user/assistant strictly. The whole conversation should be 2-6 messages depending on the dynamic."""


def sample_diversity(rng):
    category = rng.choice(list(TOPICS.keys()))
    return {
        "category": category,
        "topic": rng.choice(TOPICS[category]),
        "persona": rng.choice(PERSONAS),
        "dynamic": rng.choice(DYNAMICS),
    }


def validate(messages):
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return False
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return False
        if not isinstance(m["content"], str) or not m["content"].strip():
            return False
    return True


async def generate_one(idx, knowledge, llm, cf):
    rng = random.Random(idx)
    div = sample_diversity(rng)
    prompt = PROMPT_TEMPLATE.format(
        knowledge=knowledge,
        topic=div["topic"],
        persona=div["persona"],
        dynamic=div["dynamic"],
    )
    try:
        raw = await llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=1.0,
            max_tokens=2048,
        )
    except Exception as e:
        return None, f"generation error: {type(e).__name__}: {e}", div

    messages = raw.get("messages") if isinstance(raw, dict) else None
    if not validate(messages):
        return None, f"validation error: {messages!r:.150}", div

    try:
        safe, reason = await cf.check(messages)
    except Exception as e:
        return None, f"filter error: {type(e).__name__}: {e}", div
    if not safe:
        return None, f"contam reject: {reason[:200]}", div
    return messages, None, div


async def main_async(args):
    if not os.path.exists(KNOWLEDGE_PATH):
        raise SystemExit(f"Knowledge file not found: {KNOWLEDGE_PATH}")
    knowledge = open(KNOWLEDGE_PATH, "r", encoding="utf-8").read().strip()

    llm = LocalLLM(max_concurrency=args.workers)
    cf = ContaminationFilter(max_concurrency=args.workers)

    out_path = args.output or os.path.join(get_base_dir(), "sft_data_pre1985", "identity_pre1985.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not args.append and os.path.exists(out_path):
        os.remove(out_path)
    print(f"Output: {out_path}")
    print(f"Topics: {list(TOPICS.keys())}")
    print(f"Personas: {len(PERSONAS)}  Dynamics: {len(DYNAMICS)}\n")

    sem = asyncio.Semaphore(args.workers)
    out_lock = asyncio.Lock()
    counters = {"accepted": 0, "rejected": 0, "errors": 0}

    async def worker(idx):
        async with sem:
            messages, err, div = await generate_one(idx, knowledge, llm, cf)
        async with out_lock:
            if messages is not None:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(messages) + "\n")
                counters["accepted"] += 1
                tag = "OK"
            elif err and err.startswith(("generation", "filter")):
                counters["errors"] += 1
                tag = "ERR"
            else:
                counters["rejected"] += 1
                tag = "DROP"
            done = counters["accepted"] + counters["rejected"] + counters["errors"]
            print(f"[{done}/{args.num}] {tag:4s} cat={div['category']:20s} {(err or div['topic'])[:80]}")

    await asyncio.gather(*(worker(i) for i in range(args.num)))
    await cf.close()
    print(f"\nDone. accepted={counters['accepted']} rejected={counters['rejected']} errors={counters['errors']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
