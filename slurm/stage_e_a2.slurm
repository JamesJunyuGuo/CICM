#!/usr/bin/env bash
# Stage E Llama A2 logit-lens analysis; loads model, so run on GPU.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -J ctxmgmt_stage_e_a2
#SBATCH -o logs/ctxmgmt_stage_e_a2_%j.out
#SBATCH -e logs/ctxmgmt_stage_e_a2_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_e/llama/figs /tmp/x-jguo7-codextmp/matplotlib

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
export MPLCONFIGDIR="/tmp/x-jguo7-codextmp/matplotlib"
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NPZ="${NPZ:-results/stage_e/llama/extract_stable.npz}"
INDEX="${INDEX:-results/stage_e/llama/extract_stable_index.jsonl}"
OUT="${OUT:-results/stage_e/llama/a2_stable_summary.json}"
FIG_DIR="${FIG_DIR:-results/stage_e/llama/figs/stable}"
WRONG_FILTER="${WRONG_FILTER:-within_stale}"

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "NPZ=$NPZ"
echo "INDEX=$INDEX"
echo "OUT=$OUT"
echo "FIG_DIR=$FIG_DIR"
echo "WRONG_FILTER=$WRONG_FILTER"

time "$PY" src/analyze_a2_logitlens.py \
  --npz "$NPZ" \
  --index "$INDEX" \
  --model "$MODEL" \
  --out "$OUT" \
  --fig-dir "$FIG_DIR" \
  --wrong-filter "$WRONG_FILTER"

echo "DONE"
