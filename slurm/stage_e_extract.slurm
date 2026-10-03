#!/usr/bin/env bash
# Stage E Llama fp32 mechanistic extraction.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH -J ctxmgmt_stage_e_ext
#SBATCH -o logs/ctxmgmt_stage_e_ext_%j.out
#SBATCH -e logs/ctxmgmt_stage_e_ext_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs data results/stage_e/llama

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

DATA="${DATA:-data/stage_a.jsonl}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUT_NPZ="${OUT_NPZ:-results/stage_e/llama/extract.npz}"
INDEX_OUT="${INDEX_OUT:-results/stage_e/llama/extract_index.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DTYPE="${DTYPE:-float32}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "OUT_NPZ=$OUT_NPZ"
echo "INDEX_OUT=$INDEX_OUT"
echo "LIMIT=${LIMIT:-}"
echo "DTYPE=$DTYPE"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

if [[ ! -f "$DATA" ]]; then
  "$PY" src/gen_tasks.py --out "$DATA" --n-per-cell 150 --seed 1
fi

time "$PY" src/extract_stage_a.py \
  --data "$DATA" \
  --model "$MODEL" \
  --out "$OUT_NPZ" \
  --index-out "$INDEX_OUT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  "${LIMIT_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
