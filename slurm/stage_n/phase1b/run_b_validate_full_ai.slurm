#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=n_p1b_bval
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_validate_full_ai_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_validate_full_ai_%j.err

# Revised from 5m after the running full job showed that multi-component hook
# indexing dominates forward count: first-batch upper bound is about 2m, so six
# batches plus a 25% buffer require 15m.
set -euo pipefail
module purge
module load modtree/gpu
module load conda

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT_DIR=$ROOT/results/stage_n/phase1b/b_validate_full_ai

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/results/stage_n/phase1b/logs" "$OUT_DIR"
cd "$ROOT"

"$PY" results/stage_n/phase1b/phase1b.py upstream-validate \
  --pairs results/stage_n/behavior_full_cpu/pairs.jsonl \
  --ranking results/stage_n/phase1b/upstream_ranking.json \
  --out-dir "$OUT_DIR" --dtype float32 --batch-size 16 --n-random 64
