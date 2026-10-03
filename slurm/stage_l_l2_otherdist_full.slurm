#!/usr/bin/env bash
# Stage L L-2 other-distance full alpha sweep:
# Qwen2.5-7B local causal steering on within-stale + cross-slot pools.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
# Runtime basis: L-2 single-alpha smoke 20 rows finished in 2:23.
# Full alpha sweep has 665 failure rows and 12 generations/row, giving ~4h
# by linear scaling; request 5h as a tight buffer.
#SBATCH --time=05:00:00
#SBATCH -J l_l2_od_full_alpha
#SBATCH -o logs/l_l2_od_full_alpha_%j.out
#SBATCH -e logs/l_l2_od_full_alpha_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/natural_factorial_otherdist_l2/full_alpha_sweep

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
  --mech-dir results/stage_l/natural_factorial_otherdist_l1/full \
  --l1-summary results/stage_l/natural_factorial_otherdist_l1/full/l1_summary.json \
  --out-dir results/stage_l/natural_factorial_otherdist_l2/full_alpha_sweep \
  --report-out results/stage_l/natural_factorial_otherdist_l2/full_alpha_sweep/REPORT.md \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype float32 \
  --alpha-list 1,2,4,8,16 \
  --max-new-tokens 16 \
  --n-boot 2000 \
  --seed 20260722

echo "DONE"
