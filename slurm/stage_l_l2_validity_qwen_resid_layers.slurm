#!/usr/bin/env bash
# Stage L L-2 Qwen validity: residual-norm alpha + middle/final layer hooks.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
# Runtime basis: Qwen L-2 5-alpha final-layer run completed in 00:40:31.
# This run uses 6 alphas and 2 layers: ~97m by scaling; request 2h.
#SBATCH --time=02:00:00
#SBATCH -J l_l2_qwen_valid
#SBATCH -o logs/l_l2_qwen_valid_%j.out
#SBATCH -e logs/l_l2_qwen_valid_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/validity_l2/qwen_residual_norm_layers

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

for layer in 13 27; do
  OUT="results/stage_l/validity_l2/qwen_residual_norm_layers/layer_${layer}"
  mkdir -p "$OUT"
  "$PY" src/stage_l_mech.py intervene \
    --mech-dir results/stage_l/natural_factorial_otherdist_l1/full \
    --l1-summary results/stage_l/natural_factorial_otherdist_l1/full/l1_summary.json \
    --out-dir "$OUT" \
    --report-out "$OUT/REPORT.md" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float32 \
    --layer "$layer" \
    --alpha-mode residual_ratio \
    --alpha-list 0.01,0.02,0.05,0.1,0.2,0.4 \
    --max-new-tokens 16 \
    --n-boot 2000 \
    --seed 20260722
done

echo "DONE"
