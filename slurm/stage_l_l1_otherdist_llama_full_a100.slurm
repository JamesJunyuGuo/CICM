#!/usr/bin/env bash
# Stage L L-1 Llama-3.1-8B cross-model replicate A100 mirror:
# duplicate-safe mirror with separate outputs from the H100 job.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
# Runtime basis: Qwen L-1 full on same 1200 rows completed in 00:09:01.
# A100 may be slower than H100; request 30m as a tight buffer.
#SBATCH --time=00:30:00
#SBATCH -J l_l1_llama_full_a100
#SBATCH -o logs/l_l1_llama_full_a100_%j.out
#SBATCH -e logs/l_l1_llama_full_a100_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_l/natural_factorial_otherdist_llama/l1/full_a100

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

"$PY" src/stage_l_mech.py harvest \
  --rows results/stage_l/natural_factorial_otherdist_llama/l0/openrouter_llama31_8b_rows.jsonl \
  --out-dir results/stage_l/natural_factorial_otherdist_llama/l1/full_a100 \
  --model "$MODEL_VIEW" \
  --dtype float32

"$PY" src/stage_l_mech.py analyze \
  --mech-dir results/stage_l/natural_factorial_otherdist_llama/l1/full_a100 \
  --summary-out results/stage_l/natural_factorial_otherdist_llama/l1/full_a100/l1_summary.json \
  --probe-rows-out results/stage_l/natural_factorial_otherdist_llama/l1/full_a100/l1_probe_rows.jsonl \
  --report-out results/stage_l/natural_factorial_otherdist_llama/REPORT_a100.md \
  --n-boot 2000 \
  --n-shuffle 2000

echo "DONE"
