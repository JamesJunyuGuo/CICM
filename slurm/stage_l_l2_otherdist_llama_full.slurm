#!/usr/bin/env bash
# Stage L L-2 Llama-3.1-8B bonus alpha sweep:
# matched-norm random + identity gate on the Llama L-1 failure pools.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
# Runtime basis: Qwen L-2 alpha sweep completed in 00:40:31 for 665 failures.
# Llama has 722 failures, giving ~44m by linear scaling; request 1h.
#SBATCH --time=01:00:00
#SBATCH -J l_l2_llama_alpha
#SBATCH -o logs/l_l2_llama_alpha_%j.out
#SBATCH -e logs/l_l2_llama_alpha_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/natural_factorial_otherdist_llama/l2/full_alpha_sweep

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
MODEL_VIEW="results/stage_l/natural_factorial_otherdist_llama/model_view/llama31_8b"

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" src/stage_l_mech.py intervene \
  --mech-dir results/stage_l/natural_factorial_otherdist_llama/l1/full \
  --l1-summary results/stage_l/natural_factorial_otherdist_llama/l1/full/l1_summary.json \
  --out-dir results/stage_l/natural_factorial_otherdist_llama/l2/full_alpha_sweep \
  --report-out results/stage_l/natural_factorial_otherdist_llama/l2/full_alpha_sweep/REPORT.md \
  --model "$MODEL_VIEW" \
  --dtype float32 \
  --alpha-list 1,2,4,8,16 \
  --max-new-tokens 16 \
  --n-boot 2000 \
  --seed 20260722

echo "DONE"
