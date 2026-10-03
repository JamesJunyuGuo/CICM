#!/usr/bin/env bash
# Stage A2 logit-lens analysis; loads the model, so run it on a GPU node.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -J ctxmgmt_stage_a_a2
#SBATCH -o logs/ctxmgmt_stage_a_a2_%j.out
#SBATCH -e logs/ctxmgmt_stage_a_a2_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_a/figs /tmp/x-jguo7-codextmp/matplotlib

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

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NPZ="${NPZ:-results/stage_a/extract.npz}"
INDEX="${INDEX:-results/stage_a/extract_index.jsonl}"
OUT="${OUT:-results/stage_a/a2_summary.json}"

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "NPZ=$NPZ"
echo "INDEX=$INDEX"
echo "OUT=$OUT"

"$PY" src/analyze_a2_logitlens.py \
  --npz "$NPZ" \
  --index "$INDEX" \
  --model "$MODEL" \
  --out "$OUT" \
  --fig-dir results/stage_a/figs

echo "DONE"
