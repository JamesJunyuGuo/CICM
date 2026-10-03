#!/bin/bash
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=n_p1b_bscan_dup
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_scan_full_gpu_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_scan_full_gpu_%j.err

# A100 duplicate gets a 1.5x walltime multiplier over the H100 estimate.
set -euo pipefail
module purge
module load modtree/gpu
module load conda

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT_DIR=$ROOT/results/stage_n/phase1b/b_scan_full_gpu

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/results/stage_n/phase1b/logs" "$OUT_DIR"
cd "$ROOT"

"$PY" results/stage_n/phase1b/phase1b.py upstream-scan \
  --pairs results/stage_n/behavior_full_cpu/pairs.jsonl \
  --out-dir "$OUT_DIR" --dtype float32 --batch-size 16
