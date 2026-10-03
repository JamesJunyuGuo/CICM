#!/usr/bin/env bash
# Stage L L-1 other-distance full: Qwen2.5-7B fp32/eager internals + per-cell probe analysis.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=02:30:00
#SBATCH -J l_l1_od_full
#SBATCH -o logs/l_l1_od_full_%j.out
#SBATCH -e logs/l_l1_od_full_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/natural_factorial_otherdist_l1/full

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

"$PY" src/stage_l_mech.py harvest \
  --rows results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_rows_merged.jsonl \
  --out-dir results/stage_l/natural_factorial_otherdist_l1/full \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype float32

"$PY" src/stage_l_mech.py analyze \
  --mech-dir results/stage_l/natural_factorial_otherdist_l1/full \
  --summary-out results/stage_l/natural_factorial_otherdist_l1/full/l1_summary.json \
  --probe-rows-out results/stage_l/natural_factorial_otherdist_l1/full/l1_probe_rows.jsonl \
  --report-out results/stage_l/REPORT.md \
  --n-boot 2000 \
  --n-shuffle 2000

echo "DONE"
