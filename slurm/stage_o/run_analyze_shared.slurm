#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --job-name=o_a_analyze
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/analyze_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/analyze_%x_%j.err

# Only the final residual checkpoint is needed for the preregistered retention
# headline. This avoids an unrequested layer scan while retaining 5-fold grouped
# CV, 2,000 shuffles, and pooled log-length control.
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY=410m,1.4b,or2.8b}"
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" OMP_NUM_THREADS=8
BASE="$ROOT/results/stage_o/$MODEL_KEY/mechanism"
cd "$ROOT"
SECONDS=0
"$PY" src/pythia_circuit.py analyze \
  --harvest-dir "$BASE" --summary "$BASE/mechanism.summary.json" \
  --probe-rows "$BASE/probe_rows.jsonl" --probe-layers=-1 \
  --n-boot 2000 --n-shuffle 2000 --seed 20260818
echo "elapsed_seconds=$SECONDS"
