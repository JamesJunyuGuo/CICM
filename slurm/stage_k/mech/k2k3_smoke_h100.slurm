#!/usr/bin/env bash
# Stage K K2/K3 smoke: local 0.5B fp32-eager internals on a tiny DP+IF subset.
# No API calls in this job; judge/analysis runs separately after harvest.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J k_k2k3_smoke
#SBATCH -o logs/k_k2k3_smoke_%j.out
#SBATCH -e logs/k_k2k3_smoke_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_k/mech/smoke

module purge
module load modtree/gpu
module load conda

export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" src/icf_mech_extract.py \
  --scenario dynamic_preference \
  --input results/stage_k/k1b_dynamic_preference_input.json \
  --out-dir results/stage_k/mech/smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 8 \
  --max-new-tokens 96 \
  --dtype float32

"$PY" src/icf_mech_extract.py \
  --scenario instructional_forgetting \
  --input external_data/icf_bench/instructional_forgetting/instructional_forgetting.json \
  --out-dir results/stage_k/mech/smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 8 \
  --max-new-tokens 128 \
  --dtype float32

echo "DONE"
