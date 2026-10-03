#!/bin/bash
# Duplicate-safe A100 confirmation mirror. It reuses the complete H100 retrieval
# checkpoint; applying the observed 1.54x portal factor to the 69-minute H100
# remaining-work estimate plus 27% gives 2h15m.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:15:00
#SBATCH --job-name=r13_l3_cf_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_causal_full_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_causal_full_a100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=meta-llama/Llama-3.2-3B-Instruct
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface_llama3b_r13
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_llama_causal_heads.py \
  --confirmation-only \
  --resume-completed-tasks \
  --resume-from-dir results/stage_r/crossmodel/llama32_3b/r13_causal_full_h100 \
  --frozen-summary results/stage_r/crossmodel/llama32_3b/r13_causal_calibration_h100/summary.json \
  --model "$MODEL" --model-label meta-llama/Llama-3.2-3B-Instruct \
  --model-family llama --stage-label R13-llama3b \
  --out-dir results/stage_r/crossmodel/llama32_3b/r13_causal_full_a100 \
  --minimum-pool 100 --minimum-preservation 0.95 \
  --n-random-positions 16 --n-random-heads 64 --n-boot 2000 \
  --batch-size 8 --max-new-tokens 8
