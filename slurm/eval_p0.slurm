#!/usr/bin/env bash
# P0 validation run: current-value-under-interference on Qwen2.5-7B-Instruct.
# 1x H100 via the AI portal (7B inference on ~1200 short prompts is a few minutes).
# Switch to the 4x A100 GPU-portal header only if you scale up much larger.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J ctxmgmt_p0
#SBATCH -o logs/ctxmgmt_p0_%j.out
#SBATCH -e logs/ctxmgmt_p0_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs data results

module purge
module load modtree/gpu
module load conda
# Use the shared verl env directly (do not create/modify it).
export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

# HF cache (model is already cached; offline avoids any network dependency).
export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# If you ever need to download on a compute node, drop the OFFLINE flags and set:
# export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

echo "host=$(hostname) gpu:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Regenerate data (fast, CPU) for reproducibility, then evaluate.
"$PY" src/gen_tasks.py --out data/p0.jsonl --n-per-cell 100 --seed 0

"$PY" src/run_eval.py \
  --data data/p0.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out results/p0_qwen7b.jsonl \
  --summary results/p0_qwen7b.summary.json \
  --batch-size 32 \
  --max-new-tokens 16

echo "DONE"
