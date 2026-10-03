#!/usr/bin/env bash
# Final-checkpoint headline, nulls/component summaries, then exact layer merge.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH -J stage_n_probe_agg
#SBATCH -o results/stage_n/logs/probe_aggregate_%j.out
#SBATCH -e results/stage_n/logs/probe_aggregate_%j.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=results/stage_n/mechanism_full_cpu
"$PY" src/pythia_circuit.py analyze \
  --harvest-dir "$OUT" \
  --summary "$OUT/mechanism.summary.json" \
  --probe-rows "$OUT/probe_rows.jsonl" \
  --n-boot 2000 \
  --n-shuffle 2000 \
  --probe-layers=-1
"$PY" src/pythia_circuit.py merge-probe-chunks \
  --summary "$OUT/mechanism.summary.json" \
  --chunks "$OUT"/probe_chunks/chunk_*.json
echo DONE
