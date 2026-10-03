#!/usr/bin/env bash
# Stage D broader contextual-management behavioral eval.
# Override MODEL/LIMIT/OUT/SUMMARY for smoke or paired AI/GPU submissions.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -J ctxmgmt_stage_d_eval
#SBATCH -o logs/ctxmgmt_stage_d_eval_%j.out
#SBATCH -e logs/ctxmgmt_stage_d_eval_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_d

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

DATA="${DATA:-data/stage_d/eval_all.jsonl}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT:-results/stage_d/baseline.jsonl}"
SUMMARY="${SUMMARY:-results/stage_d/baseline.summary.json}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
DTYPE="${DTYPE:-bfloat16}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
ADAPTER_ARGS=()
if [[ -n "${ADAPTER_DIR:-}" ]]; then
  ADAPTER_ARGS+=(--adapter-dir "$ADAPTER_DIR" --adapter-kind "$ADAPTER_KIND")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "DATA=$DATA"
echo "MODEL=$MODEL"
echo "OUT=$OUT"
echo "SUMMARY=$SUMMARY"
echo "LIMIT=${LIMIT:-}"
echo "ADAPTER_DIR=${ADAPTER_DIR:-}"
echo "ADAPTER_KIND=${ADAPTER_KIND:-}"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

time "$PY" src/eval_stage_d.py \
  --data "$DATA" \
  --model "$MODEL" \
  --out "$OUT" \
  --summary "$SUMMARY" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  "${LIMIT_ARGS[@]}" \
  "${ADAPTER_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
