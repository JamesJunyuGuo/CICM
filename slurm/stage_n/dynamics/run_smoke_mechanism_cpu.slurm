#!/usr/bin/env bash
# Hook/index validation only. Formal Phase 2 mechanism curves remain H100 runs.
# Phase 1's larger CPU mechanism smoke took 512s; this reduced pass requests 8m.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:08:00
#SBATCH -J n2_hook_cpu
#SBATCH -o results/stage_n/dynamics/logs/smoke_mechanism_cpu_%j.out
#SBATCH -e results/stage_n/dynamics/logs/smoke_mechanism_cpu_%j.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
cd "$ROOT"
module purge
module load conda

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results/stage_n/dynamics/logs results/stage_n/dynamics/smoke_mechanism_cpu

SECONDS=0
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"$PY" src/pythia_dynamics.py mechanism \
  --checkpoint step0 \
  --out-root results/stage_n/dynamics/smoke_mechanism_cpu \
  --dtype float32 \
  --batch-size 12 \
  --n-induction-per-seed 2 \
  --qk-limit 12
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
