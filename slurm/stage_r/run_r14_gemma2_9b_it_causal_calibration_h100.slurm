#!/bin/bash
# Gemma-2-9B-IT full calibration. Its 24-row H100 smoke took 2m49s; 10x
# scaling predicts 28m10s, so 36 minutes gives a 28% buffer.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:36:00
#SBATCH --job-name=r14_g9_cal_h
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_gemma2_9b_it_causal_calibration_h100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_gemma2_9b_it_causal_calibration_h100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=google/gemma-2-9b-it
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_llama_causal_heads.py \
  --model "$MODEL" --model-label google/gemma-2-9b-it \
  --model-family gemma2 --stage-label R14-gemma2-9b-it \
  --out-dir results/stage_r/crossmodel/gemma2_9b_it/r14_causal_calibration_h100 \
  --minimum-correct-discovery-pool 5 --minimum-stale-discovery-pool 20 \
  --candidate-count 32 --head-sizes 1,2,4,8,16,32 \
  --margins 0.5,1,2,4 --discovery-margin 1 --max-beta 8 \
  --batch-size 3 --capture-batch-size 1 --gradient-batch-size 1 \
  --n-boot 500 --max-new-tokens 8
