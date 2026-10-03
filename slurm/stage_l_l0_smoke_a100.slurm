#!/usr/bin/env bash
# Stage L L-0 0.5B smoke duplicate on GPU/A100 portal.
# Writes under results/stage_l/smoke_a100 so it cannot overwrite the H100 smoke.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -J l_l0_smoke_a100
#SBATCH -o logs/l_l0_smoke_a100_%j.out
#SBATCH -e logs/l_l0_smoke_a100_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/smoke_a100 data/stage_l

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

"$PY" src/stage_l_eval.py generate \
  --data data/stage_l/cicm.jsonl \
  --out results/stage_l/smoke_a100/l0_rows.jsonl \
  --summary-out results/stage_l/smoke_a100/l0_summary.json \
  --report-out results/stage_l/smoke_a100/REPORT.md \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --max-new-tokens 16 \
  --limit 20

echo "DONE"
