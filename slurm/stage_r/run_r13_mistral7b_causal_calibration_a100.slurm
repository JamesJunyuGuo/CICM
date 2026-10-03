#!/bin/bash
# Duplicate-safe A100 calibration mirror. Prior 7-8B runs indicate roughly
# 1.6-2.0x H100 runtime; 55 minutes includes a modest node-variance buffer.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:55:00
#SBATCH --job-name=r13_m7_cal_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_mistral7b_causal_calibration_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_mistral7b_causal_calibration_a100_%j.err

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
  --out-dir results/stage_r/crossmodel/mistral7b_instruct/r13_causal_calibration_a100 \
  --minimum-correct-discovery-pool 5 --minimum-stale-discovery-pool 20 \
  --candidate-count 32 --head-sizes 1,2,4,8,16,32 \
  --margins 0.5,1,2,4 --discovery-margin 1 --max-beta 8 \
  --batch-size 8 --capture-batch-size 1 --gradient-batch-size 1 \
  --n-boot 500 --max-new-tokens 8
