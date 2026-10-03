#!/usr/bin/env bash
# Stage A extraction: write-span attention + final-token hiddens.
# Defaults run the full Qwen2.5-7B-Instruct extraction; override MODEL/LIMIT/OUT_NPZ
# for smoke checks.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH -J ctxmgmt_stage_a
#SBATCH -o logs/ctxmgmt_stage_a_%j.out
#SBATCH -e logs/ctxmgmt_stage_a_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs data results/stage_a

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
DATA="${DATA:-data/stage_a.jsonl}"
OUT_NPZ="${OUT_NPZ:-results/stage_a/extract.npz}"
INDEX_OUT="${INDEX_OUT:-results/stage_a/extract_index.jsonl}"
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

"$PY" src/gen_tasks.py --out "$DATA" --n-per-cell 150 --seed 1

"$PY" src/extract_stage_a.py \
  --data "$DATA" \
  --model "$MODEL" \
  --out "$OUT_NPZ" \
  --index-out "$INDEX_OUT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  "${LIMIT_ARGS[@]}"

echo "DONE"
