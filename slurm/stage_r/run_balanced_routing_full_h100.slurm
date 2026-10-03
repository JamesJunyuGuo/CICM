#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:05:00
#SBATCH --job-name=r10_bal_full_h
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/balanced_routing_full_h100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/balanced_routing_full_h100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stale_binding_natural_routing.py \
  --out-dir results/stage_r/routing/balanced_full_h100 \
  --routing-mode balanced \
  --minimum-pool 100 \
  --n-random-positions 16 --n-boot 2000 \
  --batch-size 12 --max-new-tokens 8
