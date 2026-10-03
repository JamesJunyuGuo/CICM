#!/usr/bin/env bash
# Stage E E2f retrieval-head overlap check.
# Override MODEL/LIMIT/OUT_DIR/SUMMARY for smoke/full submissions.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -J ctxmgmt_stage_e_ret
#SBATCH -o logs/ctxmgmt_stage_e_ret_%j.out
#SBATCH -e logs/ctxmgmt_stage_e_ret_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_e/retrieval_heads

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
DTYPE="${DTYPE:-float32}"
SEED="${SEED:-30}"
N_PER_LENGTH="${N_PER_LENGTH:-100}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-4}"
OUT_DIR="${OUT_DIR:-results/stage_e/retrieval_heads}"
SUMMARY="${SUMMARY:-$OUT_DIR/summary.json}"
EXAMPLES_OUT="${EXAMPLES_OUT:-$OUT_DIR/examples.jsonl}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "DTYPE=$DTYPE"
echo "SEED=$SEED"
echo "N_PER_LENGTH=$N_PER_LENGTH"
echo "LIMIT=${LIMIT:-}"
echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$SUMMARY"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

time "$PY" src/retrieval_heads.py \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --n-per-length "$N_PER_LENGTH" \
  --max-answer-tokens "$MAX_ANSWER_TOKENS" \
  --out-dir "$OUT_DIR" \
  --summary "$SUMMARY" \
  --examples-out "$EXAMPLES_OUT" \
  "${LIMIT_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
