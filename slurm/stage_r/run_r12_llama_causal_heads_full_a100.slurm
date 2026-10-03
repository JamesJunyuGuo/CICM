#!/bin/bash
# Duplicate-safe A100 mirror with an isolated output directory. All 960 rows are
# generated; the one smoke-overlap row is excluded from the primary analysis.
# R10's A100 run
# needed 1h13m54s for 16 random-position controls on two 960-row tasks. The R12
# control count is about 4.0x larger, predicting about 5h; 6h10 is a 23% buffer.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:10:00
#SBATCH --job-name=r12_l8_cf_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r12_llama_causal_heads_full_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r12_llama_causal_heads_full_a100_%j.err

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
  --confirmation-only \
  --allow-one-row-confirmation-overlap \
  --frozen-summary results/stage_r/crossmodel/llama31_8b/r12_causal_smoke_h100/summary.json \
  --model "$MODEL" \
  --out-dir results/stage_r/crossmodel/llama31_8b/r12_causal_full_a100 \
  --minimum-pool 100 --minimum-preservation 0.95 \
  --n-random-positions 16 --n-random-heads 64 --n-boot 2000 \
  --batch-size 8 --max-new-tokens 8
