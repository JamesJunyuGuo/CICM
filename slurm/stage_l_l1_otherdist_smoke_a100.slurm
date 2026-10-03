#!/usr/bin/env bash
# Stage L L-1 other-distance A100 smoke mirror: duplicate-safe, separate outputs.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -J l_l1_od_smoke_a100
#SBATCH -o logs/l_l1_od_smoke_a100_%j.out
#SBATCH -e logs/l_l1_od_smoke_a100_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/natural_factorial_otherdist_l1/smoke_a100

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
  --out-dir results/stage_l/natural_factorial_otherdist_l1/smoke_a100 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --limit 20

"$PY" src/stage_l_mech.py analyze \
  --mech-dir results/stage_l/natural_factorial_otherdist_l1/smoke_a100 \
  --summary-out results/stage_l/natural_factorial_otherdist_l1/smoke_a100/l1_summary.json \
  --probe-rows-out results/stage_l/natural_factorial_otherdist_l1/smoke_a100/l1_probe_rows.jsonl \
  --report-out results/stage_l/natural_factorial_otherdist_l1/smoke_a100/REPORT.md \
  --n-boot 100 \
  --n-shuffle 100

echo "DONE"
