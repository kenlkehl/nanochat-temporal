"""
scripts/build_sft_data.py

Async orchestrator for SFT synthetic data generation.

For each category, fires N candidate generations through the local LLM, runs each
through the local-LLM contamination filter (same server, thinking disabled, temp=0),
drops UNSAFE/UNSURE rejections, and writes survivors to a per-category JSONL.
Resumable via per-category progress sidecar files (`*.progress.json`).

Categories:
  - grounded_qa:    Q&A grounded in random passages from the pre-1985 corpus shards.
                    (Naturally mixes pre-1929 books and pre-1985 PubMed.)
  - code:           Modern Python code (no grounding, anachronism-safe seed prompts).
  - tool_use:       Math word problems solved via Python REPL tool calls
                    (list-of-parts assistant content).
  - comprehension:  Summarization/extraction over pre-1985 corpus passages.

The identity category has its own dedicated script: dev/gen_identity_pre1985.py.

Output JSONL formats:
  - grounded_qa, code, comprehension: plain-string content → tasks/customjson.py
  - tool_use:                         list-of-parts content → tasks/customjson_with_parts.py

Usage (single vLLM server, as before):
    OPENAI_BASE_URL=http://localhost:8000/v1 \\
    LOCAL_LLM_MODEL=google/gemma-4-31B-it \\
    python -m scripts.build_sft_data \\
        --grounded-qa 20000 --code 10000 --tool-use 5000 --comprehension 5000 \\
        --workers 16

Usage (auto-launch one vLLM server per GPU):
    LOCAL_LLM_MODEL=google/gemma-4-31B-it \\
    python -m scripts.build_sft_data \\
        --grounded-qa 20000 --code 10000 --tool-use 5000 --comprehension 5000 \\
        --vllm-auto-launch --vllm-gpus 0,1,2,3 --per-server-workers 16

Usage (pool already running on multiple ports):
    OPENAI_BASE_URLS=http://localhost:8000/v1,http://localhost:8001/v1 \\
    LOCAL_LLM_MODEL=google/gemma-4-31B-it \\
    python -m scripts.build_sft_data ... --per-server-workers 16
"""

from __future__ import annotations

import os
import json
import random
import asyncio
import argparse
import shlex
from typing import Optional

import pyarrow.parquet as pq
from dotenv import load_dotenv

from nanochat.common import get_base_dir
from nanochat.dataset_pre1985 import DATA_DIR
from nanochat.sft_generator import LocalLLM
from nanochat.contamination_filter import ContaminationFilter
from nanochat.vllm_launcher import VLLMPool, parse_gpu_ids_arg

load_dotenv()


# ---------------------------------------------------------------------------
# Passage sampler
# ---------------------------------------------------------------------------

class PassageSampler:
    """Yield random passages from random parquet shards in DATA_DIR."""

    def __init__(self, shards_dir: str = DATA_DIR, passage_chars: int = 1500, seed: int = 42):
        self.shards_dir = shards_dir
        self.passage_chars = passage_chars
        self.rng = random.Random(seed)
        self._cached_shards: Optional[list[str]] = None

    def _list_shards(self) -> list[str]:
        if self._cached_shards is None:
            if not os.path.isdir(self.shards_dir):
                self._cached_shards = []
            else:
                files = sorted(
                    f for f in os.listdir(self.shards_dir)
                    if f.endswith(".parquet") and not f.endswith(".tmp")
                )
                self._cached_shards = [os.path.join(self.shards_dir, f) for f in files]
        return self._cached_shards

    def sample(self) -> Optional[str]:
        shards = self._list_shards()
        if not shards:
            return None
        for _ in range(5):
            path = self.rng.choice(shards)
            try:
                pf = pq.ParquetFile(path)
                n_rg = pf.num_row_groups
                if n_rg == 0:
                    continue
                rg_idx = self.rng.randrange(n_rg)
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
                if not texts:
                    continue
                text = self.rng.choice(texts)
            except Exception:
                continue
            if not text:
                continue
            if len(text) <= self.passage_chars:
                return text
            max_start = len(text) - self.passage_chars
            start = self.rng.randrange(0, max_start + 1)
            # Bump start to next paragraph break (within 200 chars) for cleaner cuts
            for i in range(start, min(start + 200, max_start + 1) - 1):
                if text[i] == "\n" and text[i + 1] == "\n":
                    start = i + 2
                    break
            return text[start:start + self.passage_chars]
        return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GROUNDED_QA_PROMPT = """Below is a passage from a book or scientific abstract published before 1985.

Generate a SINGLE realistic Q&A turn for AI training data:
- A user question that someone might ask whose answer is supported by the passage.
- An assistant answer that uses ONLY information from the passage. Do not add outside facts.
- The answer should be self-contained — the user cannot see the passage.
- The answer must NOT mention specific events, people, products, or science from after 1984.
- Plain text, plain English. No markdown.

Passage:
---
{passage}
---

Output a JSON object with a "messages" array of exactly TWO messages:
{{"messages": [{{"role":"user","content":"..."}}, {{"role":"assistant","content":"..."}}]}}
"""

COMPREHENSION_PROMPT = """Below is a passage from a book or scientific abstract published before 1985.

Generate ONE Q&A turn where the user asks a comprehension question (summarize, identify the main point, list the key claims, paraphrase, etc.) and the assistant answers based strictly on the passage.

The answer must:
- Stay strictly within the passage. Do not introduce outside facts.
- NOT mention specific events, people, products, or science from after 1984.
- Be 2-5 sentences. Plain text, no markdown.

Passage:
---
{passage}
---

Output JSON: {{"messages": [{{"role":"user","content":"..."}}, {{"role":"assistant","content":"..."}}]}}
"""

CODE_SEEDS = [
    "Compute the mean and median of a list of numbers in Python",
    "Count word frequencies in a string",
    "Implement binary search on a sorted list",
    "Use scipy.stats.linregress to fit a line to (x, y) data",
    "Plot a histogram of a list using matplotlib",
    "Read a CSV with pandas and compute the mean of each numeric column",
    "Train a simple sklearn linear regression on a small numpy array",
    "Compute Fibonacci numbers (recursive and iterative versions)",
    "Solve FizzBuzz",
    "Write a function that returns the nth prime number",
    "Implement a queue using two stacks",
    "Compute the determinant of a matrix using numpy",
    "Implement k-means clustering from scratch on a small array",
    "Plot y = sin(x) for x in [0, 2*pi] using matplotlib",
    "Compute Pearson correlation between two lists",
    "Implement merge sort",
    "Group a pandas DataFrame by a column and aggregate the mean",
    "Compute factorial both recursively and iteratively",
    "Write a function to check if a string is a palindrome",
    "Compute the prime factors of an integer",
    "Implement bubble sort and time it on a list of 1000 random integers",
    "Compute the running mean of a stream using Welford's online algorithm",
    "Use numpy to solve a system of linear equations Ax=b",
    "Plot a scatter plot with matplotlib",
    "Implement Newton's method for finding a root of a function",
    "Compute the SVD of a matrix using numpy.linalg",
    "Implement a function to evaluate a polynomial via Horner's method",
    "Compute Euclidean distance between two points",
    "Implement the Sieve of Eratosthenes",
    "Use scipy.optimize.minimize to find the minimum of a quadratic function",
    "Implement matrix multiplication from scratch (no numpy)",
    "Compute the rolling mean of a numpy array with window size N",
    "Use itertools.combinations to enumerate all 3-subsets of a list",
    "Implement Dijkstra's shortest-path algorithm on a small graph",
    "Compute entropy of a discrete probability distribution",
    "Plot two lines on the same axes with a legend in matplotlib",
    "Use pandas to read a CSV and filter rows where column X exceeds a threshold",
    "Implement quicksort and verify on random data",
    "Write a function to compute the mode of a list",
    "Use numpy to invert a 3x3 matrix",
    "Compute the empirical cumulative distribution function of a list of numbers",
    "Implement Newton-Raphson to find sqrt(2) to 10 decimal places",
    "Implement a basic LRU cache using collections.OrderedDict",
    "Compute moving average of a list with window size 5",
    "Use scipy.stats.pearsonr to compute correlation",
    "Use numpy.fft.fft to compute the discrete Fourier transform of a signal",
    "Compute the convex hull of a set of 2D points",
    "Solve the classic 'two-sum' problem on a list of integers",
    "Compute the eigenvalues of a small symmetric matrix using numpy",
    "Implement gradient descent to minimize a simple quadratic",
    "Implement linear regression from scratch using the normal equation",
    "Implement an EM algorithm for a simple 1D Gaussian mixture",
    "Compute Z-scores for a list of measurements",
    "Use numpy to compute the rank of a matrix",
    "Implement a simple Bayesian update for a beta-binomial model",
    "Compute the cross-entropy between two discrete distributions",
    "Use scipy.integrate.quad to integrate sin(x) from 0 to pi",
    "Implement bisection method for root finding",
    "Compute autocorrelation of a 1D signal",
    "Use sklearn to fit a logistic regression on a small synthetic dataset",
]

CODE_PROMPT = """Generate ONE Q&A pair for AI training data, where the user asks a Python coding task and the assistant provides a clear, correct Python solution.

Task seed: {seed}

The user's question and the assistant's answer must NOT mention any specific event, person, product, scientific discovery, technology, or company from after 1984. (Modern Python syntax and libraries are fine — they are tools, not historical facts.)

Statistical methods, machine learning, neural nets, etc. are also fine as math/algorithms — discuss freely.

The assistant should provide:
- A short explanation (1-3 sentences)
- A complete, runnable Python code block (use markdown ```python fences)
- Optionally a brief usage example

Output a JSON object with exactly TWO messages:
{{"messages": [{{"role":"user","content":"..."}}, {{"role":"assistant","content":"..."}}]}}

The user message should be a natural-sounding request based on the task seed. The assistant message must be a single string containing the explanation and the code together.
"""

TOOL_USE_PROMPT = """Generate a math word-problem Q&A where the assistant solves it by calling a Python REPL via tool-use parts.

Generate problems involving arithmetic, basic algebra, percentages, simple geometry, ratios, unit conversions — anything that can be done with one or two Python expressions.

Do NOT use any specific event, person, product, scientific discovery, or technology from after 1984. Use neutral, timeless framings (a baker, a farmer, a student, a runner, a small business). Avoid anachronisms.

Output a JSON object exactly in this shape:
{
  "user": "<the math word problem as a single string>",
  "parts": [
    {"type": "text", "text": "<the assistant's reasoning before the calculation>"},
    {"type": "python", "text": "<a single Python expression to evaluate>"},
    {"type": "python_output", "text": "<the result of the expression>"},
    {"type": "text", "text": "<the assistant's interpretation, ending with the final answer>"}
  ]
}

The "python" expression should be SHORT (one line, evaluable with eval()). The "python_output" should be the literal numeric result (a string, e.g. "42" or "3.14"). The final text part should clearly state the answer (e.g., "The answer is 42.")."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_plain_pair(messages):
    if not isinstance(messages, list) or len(messages) != 2:
        return False
    if not all(isinstance(m, dict) for m in messages):
        return False
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        return False
    for m in messages:
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False
    return True


def parse_tool_use(raw):
    if not isinstance(raw, dict):
        return None
    user = raw.get("user")
    parts = raw.get("parts")
    if not isinstance(user, str) or not user.strip():
        return None
    if not isinstance(parts, list) or not parts:
        return None
    cleaned = []
    for p in parts:
        if not isinstance(p, dict):
            return None
        t = p.get("type")
        text = p.get("text", "")
        if t not in ("text", "python", "python_output"):
            return None
        if not isinstance(text, str):
            return None
        cleaned.append({"type": t, "text": text})
    if cleaned[0]["type"] != "text":
        return None
    if not any(p["type"] == "python" for p in cleaned):
        return None
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": cleaned},
    ]


# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------

async def _generic_qa(category, prompt, llm, cf):
    try:
        raw = await llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.95,
            max_tokens=20000,
        )
    except Exception as e:
        return None, f"gen err: {type(e).__name__}: {e}"
    messages = raw.get("messages") if isinstance(raw, dict) else None
    if not validate_plain_pair(messages):
        return None, "invalid"
    try:
        safe, reason = await cf.check(messages)
    except Exception as e:
        return None, f"filter err: {type(e).__name__}: {e}"
    if not safe:
        return None, f"contam: {reason[:120]}"
    return messages, None


async def gen_grounded_qa(idx, sampler, llm, cf):
    passage = sampler.sample()
    if passage is None:
        return None, "no passage"
    return await _generic_qa("grounded_qa", GROUNDED_QA_PROMPT.format(passage=passage), llm, cf)


async def gen_comprehension(idx, sampler, llm, cf):
    passage = sampler.sample()
    if passage is None:
        return None, "no passage"
    return await _generic_qa("comprehension", COMPREHENSION_PROMPT.format(passage=passage), llm, cf)


async def gen_code(idx, sampler, llm, cf):
    rng = random.Random(idx)
    seed = rng.choice(CODE_SEEDS)
    return await _generic_qa("code", CODE_PROMPT.format(seed=seed), llm, cf)


async def gen_tool_use(idx, sampler, llm, cf):
    try:
        raw = await llm.chat_json(
            messages=[{"role": "user", "content": TOOL_USE_PROMPT}],
            response_format={"type": "json_object"},
            temperature=1.0,
            max_tokens=20000,
        )
    except Exception as e:
        return None, f"gen err: {type(e).__name__}: {e}"
    messages = parse_tool_use(raw)
    if messages is None:
        return None, "invalid"
    try:
        safe, reason = await cf.check(messages)
    except Exception as e:
        return None, f"filter err: {type(e).__name__}: {e}"
    if not safe:
        return None, f"contam: {reason[:120]}"
    return messages, None


CATEGORY_REGISTRY = {
    "grounded_qa":   gen_grounded_qa,
    "code":          gen_code,
    "tool_use":      gen_tool_use,
    "comprehension": gen_comprehension,
}


# ---------------------------------------------------------------------------
# Orchestration with progress sidecar (resumable)
# ---------------------------------------------------------------------------

async def run_category(category, target, output_dir, sampler, llm, cf, workers, base_idx_offset):
    out_path = os.path.join(output_dir, f"{category}.jsonl")
    progress_path = os.path.join(output_dir, f"{category}.progress.json")

    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
    else:
        progress = {"completed_indices": [], "accepted": 0, "rejected": 0, "errors": 0}
    completed = set(progress["completed_indices"])

    fn = CATEGORY_REGISTRY[category]
    first_filter_err_printed = False
    consecutive_filter_errs = 0
    sem = asyncio.Semaphore(workers)
    state_lock = asyncio.Lock()

    async def worker(idx):
        nonlocal first_filter_err_printed, consecutive_filter_errs
        async with sem:
            messages, err = await fn(idx, sampler, llm, cf)
        async with state_lock:
            if messages is not None:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(messages, ensure_ascii=False) + "\n")
                progress["accepted"] += 1
                tag = "OK"
                consecutive_filter_errs = 0
            else:
                if err and ("gen err" in err or "filter err" in err):
                    progress["errors"] += 1
                    tag = "ERR"
                    if err and "filter err" in err:
                        consecutive_filter_errs += 1
                        if not first_filter_err_printed:
                            first_filter_err_printed = True
                            print(
                                f"\n[{category}] First filter error (full text):\n{err}\n"
                                "Usually means the local vLLM server is unreachable or has crashed.\n",
                                flush=True,
                            )
                        if consecutive_filter_errs >= 20:
                            raise SystemExit(
                                f"[{category}] Aborting: {consecutive_filter_errs} consecutive filter "
                                f"errors. Check the local vLLM server and rerun (progress is saved)."
                            )
                else:
                    progress["rejected"] += 1
                    tag = "DROP"
            progress["completed_indices"].append(idx)
            done = len(progress["completed_indices"])
            with open(progress_path, "w") as f:
                json.dump(progress, f)
            err_slice = (err or "").replace("\n", " ")
            if tag == "ERR":
                err_slice = err_slice[:300]
            else:
                err_slice = err_slice[:80]
            print(
                f"[{category}] [{done}/{target}] {tag:4s} "
                f"acc={progress['accepted']} rej={progress['rejected']} err={progress['errors']}  "
                f"{err_slice}"
            )

    indices = list(range(base_idx_offset, base_idx_offset + target))
    pending = [i for i in indices if i not in completed]
    print(f"\n[{category}] target={target} already_done={len(completed)} pending={len(pending)}")

    if pending:
        await asyncio.gather(*(worker(i) for i in pending))

    with open(progress_path, "w") as f:
        json.dump(progress, f)
    return progress


async def main_async(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    sampler = PassageSampler(shards_dir=args.shards_dir, seed=args.seed)
    if not sampler._list_shards():
        print(f"[warn] no parquet shards found in {args.shards_dir}; grounded_qa and comprehension will skip.")

    pool: Optional[VLLMPool] = None
    if args.vllm_auto_launch:
        model = args.vllm_model or os.environ.get("LOCAL_LLM_MODEL")
        if not model:
            raise SystemExit("--vllm-auto-launch requires --vllm-model or LOCAL_LLM_MODEL to be set.")
        gpu_ids = parse_gpu_ids_arg(args.vllm_gpus)
        if not gpu_ids:
            raise SystemExit("--vllm-auto-launch requires --vllm-gpus (e.g. '0,1,2,3').")
        pool = VLLMPool(
            model=model,
            gpu_ids=gpu_ids,
            start_port=args.vllm_start_port,
            extra_args=shlex.split(args.vllm_extra_args or ""),
            startup_timeout=args.vllm_startup_timeout,
        )
        pool.start()
        base_urls = pool.base_urls
        print(f"[build_sft_data] auto-launched {len(base_urls)} vLLM server(s): {base_urls}")
    else:
        base_urls = None

    per_server = args.per_server_workers or args.workers
    try:
        llm = LocalLLM(
            base_urls=base_urls,
            model=(args.vllm_model or None),
            max_concurrency=per_server,
            enable_thinking=args.vllm_enable_thinking,
        )
        effective_workers = max(args.workers, llm.num_backends * per_server)
        if effective_workers != args.workers:
            print(
                f"[build_sft_data] raising top-level workers {args.workers} -> "
                f"{effective_workers} to saturate {llm.num_backends} backend(s) "
                f"at {per_server} req/server."
            )
        cf = ContaminationFilter(llm=llm, max_concurrency=effective_workers)

        targets = {
            "grounded_qa":  args.grounded_qa,
            "code":         args.code,
            "tool_use":     args.tool_use,
            "comprehension": args.comprehension,
        }
        print(f"Output dir: {out_dir}")
        print(f"Targets:    {targets}")
        print(
            f"LocalLLM:   {llm.num_backends} backend(s), {per_server} req/server; "
            f"top-level workers={effective_workers}"
        )

        base_offsets = {}
        offset = 0
        for cat in CATEGORY_REGISTRY.keys():
            base_offsets[cat] = offset
            offset += targets.get(cat, 0)

        for category, target in targets.items():
            if target <= 0:
                continue
            await run_category(
                category, target, out_dir, sampler, llm, cf,
                workers=effective_workers, base_idx_offset=base_offsets[category],
            )

        await cf.close()
        await llm.close()
        print("\nAll categories done.")
    finally:
        if pool is not None:
            pool.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=str, default=DATA_DIR,
                        help=f"Pre-1985 corpus shards (default: {DATA_DIR})")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(get_base_dir(), "sft_data_pre1985"),
                        help="Output directory for JSONL files")
    parser.add_argument("--grounded-qa", type=int, default=20000)
    parser.add_argument("--code", type=int, default=10000)
    parser.add_argument("--tool-use", type=int, default=5000)
    parser.add_argument("--comprehension", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=16,
                        help="Top-level concurrency (also bounds ContaminationFilter). "
                             "Auto-raised to num_backends * per_server_workers when smaller.")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM pool / auto-launch options (all optional; single-URL env path still works).
    parser.add_argument("--vllm-auto-launch", action="store_true",
                        help="Start a pool of vLLM servers (one per GPU) for the duration of this run.")
    parser.add_argument("--vllm-model", type=str, default=os.environ.get("LOCAL_LLM_MODEL"),
                        help="Model served by vLLM and used by LocalLLM. Defaults to $LOCAL_LLM_MODEL.")
    parser.add_argument("--vllm-gpus", type=str, default=os.environ.get("VLLM_GPUS", "0"),
                        help='GPU ids, e.g. "0,1,2,3" or "0-3". One vLLM server per GPU.')
    parser.add_argument("--vllm-start-port", type=int,
                        default=int(os.environ.get("VLLM_START_PORT", 8000)),
                        help="First port; each subsequent server gets start_port+i.")
    parser.add_argument("--vllm-extra-args", type=str,
                        default=os.environ.get("VLLM_EXTRA_ARGS", ""),
                        help='Extra args forwarded to `vllm serve`, e.g. "--max-model-len 8192".')
    parser.add_argument("--vllm-startup-timeout", type=float, default=600.0,
                        help="Seconds to wait for /v1/models to respond on each vLLM server.")
    parser.add_argument("--per-server-workers", type=int, default=None,
                        help="In-flight requests per vLLM server. Falls back to --workers when unset.")
    parser.add_argument("--vllm-enable-thinking", action="store_true", default=None,
                        help="Pass chat_template_kwargs={enable_thinking: True} to vLLM "
                             "(required for Gemma-4 / Qwen3 thinking mode). "
                             "Defaults to $LOCAL_LLM_ENABLE_THINKING.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
