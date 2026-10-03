#!/bin/bash
# Mistral-7B-Instruct R13 causal-routing smoke. The shape matches the completed
# 2m09s Llama-8B smoke (32 layers x 32 query heads); 12 minutes is >5x buffer.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:12:00
#SBATCH --job-name=r13_m7_sm_h
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_mistral7b_causal_smoke_h100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_mistral7b_causal_smoke_h100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=mistralai/Mistral-7B-Instruct-v0.3
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_llama_causal_heads.py \
  --model "$MODEL" --model-label mistralai/Mistral-7B-Instruct-v0.3 \
  --model-family mistral --stage-label R13-mistral7b \
  --out-dir results/stage_r/crossmodel/mistral7b_instruct/r13_causal_smoke_h100 \
  --limit-per-cell 4 --minimum-discovery-pool 2 \
  --candidate-count 32 --head-sizes 1,2,4,8,16,32 \
  --margins 0.5,1,2,4 --discovery-margin 1 --max-beta 8 \
  --batch-size 3 --capture-batch-size 1 --gradient-batch-size 1 \
  --n-boot 300 --max-new-tokens 8
