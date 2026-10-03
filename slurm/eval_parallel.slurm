#!/usr/bin/env bash
# Parallel-maintenance behavioral run: interleaved vs serial x K, on Qwen2.5-7B.
# 1x H100 via the AI portal; inference-only, 8 CPUs are plenty.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -J ctxmgmt_par
#SBATCH -o logs/ctxmgmt_par_%j.out
#SBATCH -e logs/ctxmgmt_par_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs data results

module purge
module load modtree/gpu
module load conda
export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

echo "host=$(hostname) gpu:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" src/gen_parallel.py --out data/parallel.jsonl --n-per-cell 100 --seed 0

"$PY" src/run_eval_parallel.py \
  --data data/parallel.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out results/parallel_qwen7b.jsonl \
  --summary results/parallel_qwen7b.summary.json \
  --batch-size 32 \
  --max-new-tokens 96

echo "DONE"
