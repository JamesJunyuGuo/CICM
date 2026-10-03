#!/usr/bin/env bash
# Stage L L-2 0.5B smoke: intervention code path on smoke artifacts.
# This is a pipeline smoke only; `--force` prevents a failed 0.5B smoke gate from skipping hook validation.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH -J l_l2_smoke
#SBATCH -o logs/l_l2_smoke_%j.out
#SBATCH -e logs/l_l2_smoke_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/smoke/l2

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

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" src/stage_l_mech.py intervene \
  --mech-dir results/stage_l/smoke/l1 \
  --l1-summary results/stage_l/smoke/l1_summary.json \
  --out-dir results/stage_l/smoke/l2 \
  --report-out results/stage_l/smoke/REPORT.md \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --alpha 1.0 \
  --max-new-tokens 16 \
  --limit 2 \
  --n-boot 100 \
  --force

echo "DONE"
