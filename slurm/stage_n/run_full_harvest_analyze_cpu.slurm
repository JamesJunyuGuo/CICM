#!/usr/bin/env bash
# Stage N full activation harvest and statistical analysis (3a-3c).
# Smoke decomposition: ~3m harvest + >9m all-layer probes at n=90;
# n=284 estimate ~28m, so 35m is a 25% buffer.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH -J stage_n_harvest
#SBATCH -o results/stage_n/logs/full_harvest_%j.out
#SBATCH -e results/stage_n/logs/full_harvest_%j.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
POOL=results/stage_n/behavior_full_cpu/matched_pool.jsonl
OUT=results/stage_n/mechanism_full_cpu
mkdir -p "$OUT"
SECONDS=0
"$PY" src/pythia_circuit.py harvest --pool "$POOL" --out-dir "$OUT"
"$PY" src/pythia_circuit.py analyze \
  --harvest-dir "$OUT" \
  --summary "$OUT/mechanism.summary.json" \
  --probe-rows "$OUT/probe_rows.jsonl" \
  --n-boot 2000 \
  --n-shuffle 2000
echo "elapsed_seconds=$SECONDS"
echo DONE
