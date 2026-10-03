#!/usr/bin/env bash
# Exact layer-parallel Stage N grouped probes; two checkpoints per task.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --array=0-6
#SBATCH -J stage_n_probe
#SBATCH -o results/stage_n/logs/probe_%A_%a.out
#SBATCH -e results/stage_n/logs/probe_%A_%a.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
LAYERS=("0,1" "2,3" "4,5" "6,7" "8,9" "10,11" "12")
OUT=results/stage_n/mechanism_full_cpu/probe_chunks
mkdir -p "$OUT"
"$PY" src/pythia_circuit.py fit-probe-chunk \
  --harvest-dir results/stage_n/mechanism_full_cpu \
  --layers "${LAYERS[$SLURM_ARRAY_TASK_ID]}" \
  --out "$OUT/chunk_${SLURM_ARRAY_TASK_ID}.json"
echo DONE
