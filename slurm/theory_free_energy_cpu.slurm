#!/usr/bin/env bash
# Frozen-summary analysis only. Local tests take ~10 s; 5 min includes queue-node startup.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH -J theory_free_energy
#SBATCH -o results/theory_free_energy/logs/theory_free_energy_%j.out
#SBATCH -e results/theory_free_energy/logs/theory_free_energy_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
cd "$PROJ"

module purge
module load conda
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

"$PY" src/theory_free_energy.py \
  --out-dir results/theory_free_energy \
  --bootstrap-reps 5000 \
  --seed 20260810
