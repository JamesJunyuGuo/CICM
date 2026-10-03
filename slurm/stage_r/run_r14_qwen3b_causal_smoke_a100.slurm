#!/bin/bash
# Duplicate-safe A100 mirror of the 96-row Qwen2.5-3B pool-repair smoke.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:18:00
#SBATCH --job-name=r14_q3_sm_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_qwen3b_causal_smoke_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_qwen3b_causal_smoke_a100_%j.err

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
  --model "$MODEL" --model-label Qwen/Qwen2.5-3B-Instruct \
  --model-family qwen2 --stage-label R14-qwen3b \
  --out-dir results/stage_r/crossmodel/qwen25_3b/r14_causal_smoke_a100 \
  --limit-per-cell 16 --minimum-discovery-pool 2 \
  --candidate-count 32 --head-sizes 1,2,4,8,16,32 \
  --margins 0.5,1,2,4 --discovery-margin 1 --max-beta 8 \
  --batch-size 6 --capture-batch-size 1 --gradient-batch-size 1 \
  --n-boot 300 --max-new-tokens 8
