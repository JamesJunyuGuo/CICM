#!/usr/bin/env bash
# Stage N Phase 2 CPU-only aggregation after all formal H100 slices succeed.
# Vectorized bootstrap over ~0.8M rows; 15m is a bounded first-run allowance.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH -J n2_analysis
#SBATCH -o results/stage_n/dynamics/logs/analysis_%j.out
#SBATCH -e results/stage_n/dynamics/logs/analysis_%j.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
cd "$ROOT"
module purge
module load conda

export PYTHONPATH="$ROOT/src"
export MPLBACKEND=Agg
mkdir -p results/stage_n/dynamics/logs

SECONDS=0
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"$PY" src/pythia_dynamics.py analyze \
  --out-root results/stage_n/dynamics \
  --n-boot 2000
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
