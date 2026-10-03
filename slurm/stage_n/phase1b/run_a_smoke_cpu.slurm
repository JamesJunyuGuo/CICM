#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --job-name=n_p1b_a_cpu
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/a_cpu_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/a_cpu_%j.err

set -euo pipefail
module purge
module load conda

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT_DIR=$ROOT/results/stage_n/phase1b/a_smoke_cpu

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p "$ROOT/results/stage_n/phase1b/logs" "$OUT_DIR"
cd "$ROOT"

"$PY" results/stage_n/phase1b/phase1b.py ablate-heldout \
  --pool results/stage_n/behavior_full_cpu/matched_pool.jsonl \
  --harvest-dir results/stage_n/mechanism_full_cpu \
  --cumulative-summary results/stage_n/mechanism_full_core_ai32/cumulative_patch.summary.json \
  --phase1-ablation results/stage_n/ablation_full_ai32/ablation.summary.json \
  --out-dir "$OUT_DIR" --dtype float32 --batch-size 16 \
  --n-random 2 --n-boot 200 --seed 20260809
