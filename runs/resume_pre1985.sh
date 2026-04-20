#!/bin/bash

# Resume speedrun_pre1985.sh after a crash in/after pretraining (step 3).
# Runs base_eval, identity gen, bulk SFT data gen, and SFT training.
# vLLM auto-launch is ON by default so you don't need an external vLLM server.
#
# Defaults tuned for google/gemma-4-31B-it with thinking enabled.
# Requires vllm >= 0.19.1 in the .venv. Client-side strips <|channel>...<channel|>
# via regex — do NOT set --reasoning-parser gemma4 (vllm #38855 drops reasoning
# content when skip_special_tokens=true, which is the default).
#
# Env knobs (all optional, all inherit the speedrun_pre1985.sh defaults except
# VLLM_AUTO_LAUNCH which defaults ON here):
#   ANTHROPIC_API_KEY         (or ANTHROPIC_BACKEND=vertex + VERTEX vars)
#   VLLM_AUTO_LAUNCH=1        default ON (set =0 if you already have a server)
#   VLLM_MODEL                default google/gemma-4-31B-it
#   VLLM_GPUS                 default "0" (one server per GPU); set e.g. "0,1,2,3"
#   VLLM_START_PORT           default 8000
#   VLLM_EXTRA_ARGS           e.g. "--max-model-len 16384 --gpu-memory-utilization 0.90"
#   VLLM_PER_SERVER_WORKERS   default 16
#   VLLM_ENABLE_THINKING      default 1 (Gemma-4 thinking on)
#   DEPTH, WANDB_RUN, GROUNDED_QA_N, CODE_N, TOOL_USE_N, COMPREHENSION_N — as in speedrun.

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DATASET="pre1985"
mkdir -p "$NANOCHAT_BASE_DIR"



# vllm sanity check (the original crash reason)
python -c "import vllm, packaging.version as v; assert v.parse(vllm.__version__) >= v.parse('0.19.1'), f'need vllm>=0.19.1, have {vllm.__version__}'; print(f'vllm {vllm.__version__} OK')"

DEPTH="${DEPTH:-24}"
WANDB_RUN="${WANDB_RUN:-dummy}"
GROUNDED_QA_N="${GROUNDED_QA_N:-20000}"
CODE_N="${CODE_N:-10000}"
TOOL_USE_N="${TOOL_USE_N:-5000}"
COMPREHENSION_N="${COMPREHENSION_N:-5000}"

VLLM_AUTO_LAUNCH="${VLLM_AUTO_LAUNCH:-1}"
VLLM_MODEL="${VLLM_MODEL:-${LOCAL_LLM_MODEL:-google/gemma-4-31B-it}}"
VLLM_GPUS="${VLLM_GPUS:-0}"
VLLM_START_PORT="${VLLM_START_PORT:-8000}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_PER_SERVER_WORKERS="${VLLM_PER_SERVER_WORKERS:-16}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --max-model-len 30000"
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

if [ "${ANTHROPIC_BACKEND:-api}" = "vertex" ]; then
    if [ -z "$ANTHROPIC_VERTEX_PROJECT_ID" ]; then
        echo "ERROR: ANTHROPIC_BACKEND=vertex requires ANTHROPIC_VERTEX_PROJECT_ID."
        exit 1
    fi
    echo "Anthropic backend: Vertex (project=$ANTHROPIC_VERTEX_PROJECT_ID region=${ANTHROPIC_VERTEX_REGION:-unset})"
elif [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set (or set ANTHROPIC_BACKEND=vertex with project/region)."
    exit 1
else
    echo "Anthropic backend: public API"
fi

# Confirm the pretrained base checkpoint is actually on disk — otherwise step 3
# didn't really finish and we should bail rather than silently retrain.
if ! ls "$NANOCHAT_BASE_DIR"/base_checkpoints/*/model_*.pt >/dev/null 2>&1; then
    echo "ERROR: no pretrained checkpoint found under $NANOCHAT_BASE_DIR/base_checkpoints/."
    echo "       Run speedrun_pre1985.sh first to complete steps 1-3."
    exit 1
fi
echo "Found pretrained checkpoint(s) under $NANOCHAT_BASE_DIR/base_checkpoints/"

# -----------------------------------------------------------------------------
# 3b) Base eval (cheap, idempotent — safe to re-run)
# echo "=========================================================================="
# echo "3b) Base eval"
# echo "=========================================================================="
# torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# 4) Generate identity conversations
echo "=========================================================================="
echo "4) Generating identity conversations"
echo "=========================================================================="
python -m dev.gen_identity_pre1985 --num=200 --workers=8 "${VLLM_ARGS[@]}"

# -----------------------------------------------------------------------------
# 5) Generate bulk SFT data (resumable)
echo "=========================================================================="
echo "5) Generating SFT data (resumable)"
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

echo "=========================================================================="
echo "Done. Try chatting with Fluttering Pebble:"
echo "  python -m scripts.chat_cli"
echo "  python -m scripts.chat_web    (browser UI)"
echo "=========================================================================="

python -m nanochat.report generate
