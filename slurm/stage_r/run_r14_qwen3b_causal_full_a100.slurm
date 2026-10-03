#!/bin/bash
# Duplicate-safe A100 mirror. Calibration was 1.54x slower than H100; applying
# that factor to the 65-minute estimate plus 30% gives a 2h10m request.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:10:00
#SBATCH --job-name=r14_q3_cf_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_qwen3b_causal_full_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_qwen3b_causal_full_a100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=/anvil/scratch/x-jguo7/cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_llama_causal_heads.py \
  --confirmation-only \
  --frozen-summary results/stage_r/crossmodel/qwen25_3b/r14_causal_calibration_h100/summary.json \
  --model "$MODEL" --model-label Qwen/Qwen2.5-3B-Instruct \
  --model-family qwen2 --stage-label R14-qwen3b \
  --out-dir results/stage_r/crossmodel/qwen25_3b/r14_causal_full_a100 \
  --minimum-pool 100 --minimum-preservation 0.95 \
  --n-random-positions 16 --n-random-heads 64 --n-boot 2000 \
  --batch-size 8 --max-new-tokens 8
