#!/usr/bin/env bash
# CPU fallback for Stage N Step-0; Pythia-160m is small enough for shared nodes.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH -J stage_n_step0_cpu
#SBATCH -o results/stage_n/logs/step0_cpu_%j.out
#SBATCH -e results/stage_n/logs/step0_cpu_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p results/stage_n/logs

module purge
module load conda

PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-smoke}"
if [[ "$MODE" == "smoke" ]]; then N_PER=2; else N_PER=72; fi
OUT="results/stage_n/behavior_${MODE}_cpu"
mkdir -p "$OUT"

echo "host=$(hostname) mode=$MODE start=$(date --iso-8601=seconds)"
SECONDS=0
"$PY" src/pythia_gen.py generate \
  --out "$OUT/tasks.jsonl" \
  --summary "$OUT/tasks.summary.json" \
  --n-per-seed-cell "$N_PER"
"$PY" src/pythia_eval.py run \
  --data "$OUT/tasks.jsonl" \
  --out "$OUT/rows.jsonl" \
  --summary "$OUT/summary.json" \
  --matched-pool "$OUT/matched_pool.jsonl" \
  --batch-size 64
if [[ "$MODE" == "full" ]]; then
  "$PY" -c "import json; s=json.load(open('$OUT/summary.json')); assert s['fixed_k_gate']['gate_pass'], s['fixed_k_gate']"
  "$PY" src/pythia_eval.py build-pairs \
    --rows "$OUT/matched_pool.jsonl" \
    --out "$OUT/pairs.jsonl" \
    --summary "$OUT/pairs.summary.json" \
    --n-pairs 96 \
    --min-pairs 48 \
    --batch-size 64
fi
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
