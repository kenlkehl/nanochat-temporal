#!/bin/bash

# Fluttering Pebble — pre-1985 contamination-controlled training pipeline.
#
# Trains a small LLM on a corpus of pre-1929 books + pre-1985 PubMed abstracts,
# then SFTs it on synthetic data generated AND filtered by a local vLLM-served
# model. No external API calls — the same local model handles both generation
# and contamination judging.
# Goal: probe whether a bounded-knowledge LLM can come up with novel-to-1985
# scientific ideas.
#
# Local LLM for bulk SFT generation + filtering:
#   OPENAI_BASE_URL    — local OpenAI-compatible LLM endpoint (e.g. http://localhost:8000/v1)
#   OPENAI_BASE_URLS   — comma-separated pool for load-balanced generation, e.g.
#                        http://localhost:8000/v1,http://localhost:8001/v1,...
#                        (use this when vLLM is already running on multiple ports)
#   LOCAL_LLM_MODEL    — model name the local server exposes (default: google/gemma-4-31B-it)
#
# Optional vLLM auto-launch (one server per GPU, spun up per generator stage):
#   VLLM_AUTO_LAUNCH=1                       enable auto-launch (default 0)
#   VLLM_MODEL=google/gemma-4-31B-it         model to serve (falls back to LOCAL_LLM_MODEL)
#   VLLM_GPUS="0,1,2,3"                      GPU ids; one vLLM server per GPU
#   VLLM_START_PORT=8000                     first port; each GPU gets start_port+i
#   VLLM_EXTRA_ARGS="--max-model-len 8192"   passthrough flags to `vllm serve`
#   VLLM_PER_SERVER_WORKERS=16               in-flight requests per server
#   VLLM_ENABLE_THINKING=1                   forward chat_template_kwargs={enable_thinking: True}
#                                            (default 1 for Gemma-4; also required for Qwen3 thinking)
#
# Gemma-4 notes: requires vllm >= 0.19.1. Client-side strips <|channel>...<channel|>
# thinking preamble via regex — do NOT pass --reasoning-parser gemma4 (vllm #38855
# drops the reasoning content when skip_special_tokens=true, the default).
#
# Optional environment:
#   DEPTH              — model depth (default 24); pick smaller if you have less compute
#   WANDB_RUN          — wandb run name (default 'dummy', skips wandb)
#   GROUNDED_QA_N, CODE_N, TOOL_USE_N, COMPREHENSION_N — SFT category targets
#   TARGET_TOKENS      — pretraining corpus size in tokens (default 30B)
#
# Resumability: every step is resumable. Kill at any point and re-run.

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DATASET="pre1985"   # IMPORTANT: switches dataset.py to pre-1985
mkdir -p "$NANOCHAT_BASE_DIR"

# venv setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Defaults
DEPTH="${DEPTH:-24}"
WANDB_RUN="${WANDB_RUN:-dummy}"
GROUNDED_QA_N="${GROUNDED_QA_N:-20000}"
CODE_N="${CODE_N:-10000}"
TOOL_USE_N="${TOOL_USE_N:-5000}"
COMPREHENSION_N="${COMPREHENSION_N:-5000}"
TARGET_TOKENS="${TARGET_TOKENS:-30000000000}"

VLLM_AUTO_LAUNCH="${VLLM_AUTO_LAUNCH:-0}"
VLLM_MODEL="${VLLM_MODEL:-${LOCAL_LLM_MODEL:-google/gemma-4-31B-it}}"
VLLM_GPUS="${VLLM_GPUS:-0}"
VLLM_START_PORT="${VLLM_START_PORT:-8000}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_PER_SERVER_WORKERS="${VLLM_PER_SERVER_WORKERS:-16}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-1}"

VLLM_ARGS=()
if [ "$VLLM_AUTO_LAUNCH" = "1" ]; then
    VLLM_ARGS+=(--vllm-auto-launch
                --vllm-model "$VLLM_MODEL"
                --vllm-gpus "$VLLM_GPUS"
                --vllm-start-port "$VLLM_START_PORT"
                --vllm-extra-args "$VLLM_EXTRA_ARGS"
                --per-server-workers "$VLLM_PER_SERVER_WORKERS")
    echo "vLLM auto-launch: model=$VLLM_MODEL gpus=$VLLM_GPUS start_port=$VLLM_START_PORT per_server_workers=$VLLM_PER_SERVER_WORKERS"
fi
if [ "$VLLM_ENABLE_THINKING" = "1" ]; then
    VLLM_ARGS+=(--vllm-enable-thinking)
    echo "vLLM enable_thinking: on"
fi

if [ "$VLLM_AUTO_LAUNCH" != "1" ] && [ -z "$OPENAI_BASE_URL" ] && [ -z "$OPENAI_BASE_URLS" ]; then
    echo "WARNING: VLLM_AUTO_LAUNCH is unset/0 and neither OPENAI_BASE_URL nor OPENAI_BASE_URLS is set."
    echo "         SFT data generation will default to http://localhost:8000/v1."
fi

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 1) Build pre-1985 pretraining corpus (CPU, multi-hour, idempotent)
echo "=========================================================================="
echo "1) Building pre-1985 pretraining corpus (target ~${TARGET_TOKENS} tokens)"
echo "=========================================================================="
python -m scripts.build_pretrain_corpus \
    --target-tokens "$TARGET_TOKENS" \
    --mix "books_ia:0.55,books_loc:0.10,books_gutenberg:0.05,pubmed:0.30"

# -----------------------------------------------------------------------------
# 2) Train tokenizer on the new corpus
echo "=========================================================================="
echo "2) Training tokenizer on pre-1985 corpus"
echo "=========================================================================="
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 3) Pretrain (uses unmodified base_train.py — picks up corpus via NANOCHAT_DATASET)
echo "=========================================================================="
echo "3) Pretraining (depth=$DEPTH)"
echo "=========================================================================="
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --device-batch-size=16 \
    --fp8 \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# 4) Generate identity conversations (local LLM only, ~30 min)
echo "=========================================================================="
echo "4) Generating identity conversations"
echo "=========================================================================="
python -m dev.gen_identity_pre1985 --num=200 --workers=8 "${VLLM_ARGS[@]}"

# -----------------------------------------------------------------------------
# 5) Generate bulk SFT data (local LLM only, ~hours, resumable)
echo "=========================================================================="
echo "5) Generating SFT data (resumable; ~hours of local LLM calls)"
echo "=========================================================================="
python -m scripts.build_sft_data \
    --grounded-qa "$GROUNDED_QA_N" \
    --code "$CODE_N" \
    --tool-use "$TOOL_USE_N" \
    --comprehension "$COMPREHENSION_N" \
    --workers 16 \
    "${VLLM_ARGS[@]}"

# -----------------------------------------------------------------------------
# 6) SFT
echo "=========================================================================="
echo "6) Supervised fine-tuning"
echo "=========================================================================="
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft_pre1985 -- \
    --device-batch-size=16 \
    --chatcore-every=-1 \
    --run="$WANDB_RUN"

# -----------------------------------------------------------------------------
# Done. Try it out:
echo "=========================================================================="
echo "Done. Try chatting with Fluttering Pebble:"
echo "  python -m scripts.chat_cli"
echo "  python -m scripts.chat_web    (browser UI)"
echo "=========================================================================="

python -m nanochat.report generate
