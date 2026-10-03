#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --job-name=n_p1b_bval_cpu
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_validate_smoke_cpu_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_validate_smoke_cpu_%j.err

# One pair needs about 60 forward passes versus 434 in the 4m50s CPU scan smoke.
set -euo pipefail
module purge
module load conda

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT_DIR=$ROOT/results/stage_n/phase1b/b_validate_smoke_cpu

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
mkdir -p "$ROOT/results/stage_n/phase1b/logs" "$OUT_DIR"
cd "$ROOT"

"$PY" results/stage_n/phase1b/phase1b.py upstream-validate \
  --pairs results/stage_n/behavior_full_cpu/pairs.jsonl \
  --ranking results/stage_n/phase1b/smoke_upstream_ranking.json \
  --out-dir "$OUT_DIR" --dtype float32 --batch-size 1 --limit 1 --n-random 2
