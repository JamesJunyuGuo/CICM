#!/usr/bin/env bash
# Stage N 3a-3c statistics from the completed full activation harvest.
# Measured n=284 grouped-probe throughput is >3m/checkpoint; 14 checkpoints plus
# resampling estimate ~48m, so 60m provides a 25% buffer.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -J stage_n_analyze
#SBATCH -o results/stage_n/logs/full_analyze_%j.out
#SBATCH -e results/stage_n/logs/full_analyze_%j.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=results/stage_n/mechanism_full_cpu
SECONDS=0
"$PY" src/pythia_circuit.py analyze \
  --harvest-dir "$OUT" \
  --summary "$OUT/mechanism.summary.json" \
  --probe-rows "$OUT/probe_rows.jsonl" \
  --n-boot 2000 \
  --n-shuffle 2000
echo "elapsed_seconds=$SECONDS"
echo DONE
