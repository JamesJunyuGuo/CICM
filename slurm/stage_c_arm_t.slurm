#!/usr/bin/env bash
# Stage C Arm T temperature-sharpening eval. Override MODEL/LIMIT/TAUS for smoke.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -J ctxmgmt_stage_c_t
#SBATCH -o logs/ctxmgmt_stage_c_t_%j.out
#SBATCH -e logs/ctxmgmt_stage_c_t_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_c

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

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT:-results/stage_c/arm_t.jsonl}"
SUMMARY="${SUMMARY:-results/stage_c/arm_t.summary.json}"
TAUS="${TAUS:-1,0.7,0.5,0.3}"
FRESH_LIMIT="${FRESH_LIMIT:-300}"
FRESH_EVAL="${FRESH_EVAL:-data/stage_c/eval_battery.jsonl}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "OUT=$OUT"
echo "SUMMARY=$SUMMARY"
echo "TAUS=$TAUS"
echo "FRESH_EVAL=$FRESH_EVAL"
echo "FRESH_LIMIT=$FRESH_LIMIT"
echo "LIMIT=${LIMIT:-}"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

time "$PY" src/stage_c_arm_t.py \
  --model "$MODEL" \
  --out "$OUT" \
  --summary "$SUMMARY" \
  --taus "$TAUS" \
  --fresh-eval "$FRESH_EVAL" \
  --fresh-limit "$FRESH_LIMIT" \
  "${LIMIT_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
