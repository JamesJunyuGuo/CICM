#!/usr/bin/env bash
# Stage L L-0b local hardness sweep on Qwen2.5-7B-Instruct.
# Set OUT_SUFFIX=a100 when submitting duplicate portal runs.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH -J l_l0b_sweep
#SBATCH -o logs/l_l0b_sweep_%j.out
#SBATCH -e logs/l_l0b_sweep_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l

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

OUT_SUFFIX="${OUT_SUFFIX:-h100}"
OUT_ROOT="results/stage_l/l0b_sweep_${OUT_SUFFIX}"
mkdir -p "$OUT_ROOT"

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "OUT_ROOT=$OUT_ROOT"

for name in mild medium hard; do
  "$PY" src/stage_l_eval.py generate \
    --data "data/stage_l/l0b_${name}.jsonl" \
    --out "$OUT_ROOT/${name}_rows.jsonl" \
    --summary-out "$OUT_ROOT/${name}_summary.json" \
    --report-out "$OUT_ROOT/${name}_REPORT.md" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-new-tokens 16
done

echo "DONE"
