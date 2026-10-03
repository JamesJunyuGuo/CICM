#!/bin/bash
# Resume held-out confirmation from the complete retrieval checkpoint. The
# first H100 run processed 34/80 derived controls in 27 minutes after retrieval;
# 1h30m covers the full derived task with a measured buffer.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --job-name=r13_l3_cf_h
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_causal_full_h100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_causal_full_h100_%j.err

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
  --frozen-summary results/stage_r/crossmodel/llama32_3b/r13_causal_calibration_h100/summary.json \
  --model "$MODEL" --model-label meta-llama/Llama-3.2-3B-Instruct \
  --model-family llama --stage-label R13-llama3b \
  --out-dir results/stage_r/crossmodel/llama32_3b/r13_causal_full_h100 \
  --minimum-pool 100 --minimum-preservation 0.95 \
  --n-random-positions 16 --n-random-heads 64 --n-boot 2000 \
  --batch-size 12 --max-new-tokens 8
