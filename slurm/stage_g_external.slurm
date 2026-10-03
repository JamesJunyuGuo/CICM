#!/usr/bin/env bash
# Stage G-2 external benchmark evaluation for one dataset x arm.
# Set DATA, ARM, MODEL, ADAPTER_DIR, ADAPTER_KIND, OUT_ROOT, PORTAL_TAG.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH -J ctxmgmt_stage_g_ext
#SBATCH -o logs/ctxmgmt_stage_g_ext_%j.out
#SBATCH -e logs/ctxmgmt_stage_g_ext_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_g/external

module purge
module load modtree/gpu
module load conda

export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

DATA="${DATA:-data/stage_g/external/ruler_vt.jsonl}"
ARM="${ARM:-baseline}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
ADAPTER_KIND="${ADAPTER_KIND:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PROMPT_STYLE="${PROMPT_STYLE:-default}"
LIMIT="${LIMIT:-}"
PORTAL_TAG="${PORTAL_TAG:-ai}"
OUT_ROOT="${OUT_ROOT:-results/stage_g/external/${PORTAL_TAG}/${ARM}}"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
ADAPTER_ARGS=()
if [[ -n "$ADAPTER_DIR" ]]; then
  ADAPTER_ARGS+=(--adapter-dir "$ADAPTER_DIR" --adapter-kind "$ADAPTER_KIND")
fi

dataset_name="$(basename "$DATA" .jsonl)"
mkdir -p "$OUT_ROOT/$dataset_name" "$OUT_ROOT/identity"

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL ARM=$ARM DATA=$DATA OUT_ROOT=$OUT_ROOT PROMPT_STYLE=$PROMPT_STYLE"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

if [[ -n "$ADAPTER_DIR" ]]; then
  "$PY" src/stage_g_external.py \
    --data "$DATA" \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --arm baseline_identity \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt-style "$PROMPT_STYLE" \
    --limit 20 \
    --out "$OUT_ROOT/identity/${dataset_name}.baseline20.jsonl" \
    --summary "$OUT_ROOT/identity/${dataset_name}.baseline20.summary.json"

  "$PY" src/stage_g_external.py \
    --data "$DATA" \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    "${ADAPTER_ARGS[@]}" \
    --adapter-mode off \
    --arm "${ARM}_off_identity" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt-style "$PROMPT_STYLE" \
    --limit 20 \
    --out "$OUT_ROOT/identity/${dataset_name}.${ARM}_off20.jsonl" \
    --summary "$OUT_ROOT/identity/${dataset_name}.${ARM}_off20.summary.json"

  "$PY" src/stage_g_external.py \
    --identity-baseline "$OUT_ROOT/identity/${dataset_name}.baseline20.jsonl" \
    --identity-candidate "$OUT_ROOT/identity/${dataset_name}.${ARM}_off20.jsonl" \
    --identity-out "$OUT_ROOT/identity/${dataset_name}.${ARM}_identity.json" \
    --out "$OUT_ROOT/identity/unused.jsonl" \
    --summary "$OUT_ROOT/identity/unused.summary.json" \
    --limit 20
fi

"$PY" src/stage_g_external.py \
  --data "$DATA" \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  "${ADAPTER_ARGS[@]}" \
  --adapter-mode on \
  --arm "$ARM" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prompt-style "$PROMPT_STYLE" \
  --out "$OUT_ROOT/$dataset_name/results.jsonl" \
  --summary "$OUT_ROOT/$dataset_name/summary.json" \
  "${LIMIT_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
