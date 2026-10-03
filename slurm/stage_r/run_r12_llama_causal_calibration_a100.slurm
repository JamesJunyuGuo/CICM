#!/bin/bash
# Duplicate-safe A100 mirror with isolated output. The H100 estimate is about
# 21m30s; prior Llama smoke gives an A100/H100 factor near 1.65, so 50 minutes
# includes roughly 40% buffer beyond the resulting 35m30s estimate.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:50:00
#SBATCH --job-name=r12_l8_cal_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r12_llama_causal_calibration_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r12_llama_causal_calibration_a100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=/anvil/scratch/x-jguo7/cache/huggingface/transformers/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_llama_causal_heads.py \
  --model "$MODEL" \
  --out-dir results/stage_r/crossmodel/llama31_8b/r12_causal_calibration_a100 \
  --minimum-discovery-pool 20 \
  --candidate-count 32 --head-sizes 1,2,4,8,16,32 \
  --margins 0.5,1,2,4 --discovery-margin 1 --max-beta 8 \
  --batch-size 8 --capture-batch-size 1 --gradient-batch-size 1 \
  --n-boot 500 --max-new-tokens 8
