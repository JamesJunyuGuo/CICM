#!/usr/bin/env bash
# Stage N Step-0: Pythia-160m task hardening and fixed-k pool selection.
# Expected full runtime is under 10 minutes; 20 minutes includes model/cache startup.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH -J stage_n_step0
#SBATCH -o results/stage_n/logs/step0_%j.out
#SBATCH -e results/stage_n/logs/step0_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p results/stage_n/logs

module purge
module load modtree/gpu
module load conda

PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

MODE="${MODE:-smoke}"
PORTAL_TAG="${PORTAL_TAG:-ai}"
if [[ "$MODE" == "smoke" ]]; then
  N_PER=2
else
  N_PER=72
fi
OUT="results/stage_n/behavior_${MODE}_${PORTAL_TAG}"
mkdir -p "$OUT"

echo "host=$(hostname) mode=$MODE portal=$PORTAL_TAG start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
  --batch-size 128

if [[ "$MODE" == "full" ]]; then
  "$PY" -c "import json; s=json.load(open('$OUT/summary.json')); assert s['fixed_k_gate']['gate_pass'], s['fixed_k_gate']"
  "$PY" src/pythia_eval.py build-pairs \
    --rows "$OUT/matched_pool.jsonl" \
    --out "$OUT/pairs.jsonl" \
    --summary "$OUT/pairs.summary.json" \
    --n-pairs 96 \
    --min-pairs 48 \
    --batch-size 128
fi

echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
