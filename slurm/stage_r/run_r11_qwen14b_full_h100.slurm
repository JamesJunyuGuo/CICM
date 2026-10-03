#!/bin/bash
# Qwen2.5-14B R11 full run. Estimated from the 3m24s smoke and the prior R10
# 36m57s full: 75-85 minutes including 240-row head discovery; 1h50 includes
# a modest queue/node variance buffer.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=01:50:00
#SBATCH --job-name=r11_q14_full_h
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r11_qwen14b_full_h100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r11_qwen14b_full_h100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=/anvil/projects/x-cis250190/my_hf_cache/transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_crossmodel_routing.py \
  --model "$MODEL" --model-label Qwen/Qwen2.5-14B-Instruct --model-family qwen2 \
  --out-dir results/stage_r/crossmodel/qwen25_14b/full_h100 \
  --minimum-discovery-pool 5 --minimum-pool 100 \
  --n-random-positions 16 --n-boot 2000 \
  --batch-size 3 --capture-batch-size 1 --max-new-tokens 8
