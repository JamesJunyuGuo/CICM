#!/usr/bin/env bash
# Stage E P0 behavioral eval plus stale-error signature analysis.
# Override MODEL/OUT/SUMMARY/ERROR_SUMMARY for Llama or smokes.
# For paired AI/GPU submissions, override OUT/SUMMARY/ERROR_SUMMARY per portal.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -J ctxmgmt_stage_e_p0
#SBATCH -o logs/ctxmgmt_stage_e_p0_%j.out
#SBATCH -e logs/ctxmgmt_stage_e_p0_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs data results/stage_e

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

DATA="${DATA:-data/p0.jsonl}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUT="${OUT:-results/stage_e/llama/p0.jsonl}"
SUMMARY="${SUMMARY:-results/stage_e/llama/p0.summary.json}"
ERROR_SUMMARY="${ERROR_SUMMARY:-results/stage_e/llama/p0_errors.summary.json}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DTYPE="${DTYPE:-bfloat16}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "OUT=$OUT"
echo "SUMMARY=$SUMMARY"
echo "ERROR_SUMMARY=$ERROR_SUMMARY"
echo "LIMIT=${LIMIT:-}"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

if [[ ! -f "$DATA" ]]; then
  "$PY" src/gen_tasks.py --out "$DATA" --n-per-cell 100 --seed 0
fi

time "$PY" src/run_eval.py \
  --data "$DATA" \
  --model "$MODEL" \
  --out "$OUT" \
  --summary "$SUMMARY" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  "${LIMIT_ARGS[@]}"

"$PY" src/analyze_p0_errors.py \
  --data "$DATA" \
  --results "$OUT" \
  --out "$ERROR_SUMMARY" \
  --model "$MODEL"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
