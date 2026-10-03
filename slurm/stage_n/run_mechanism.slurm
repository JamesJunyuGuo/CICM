#!/usr/bin/env bash
# Stage N Phase-1 mechanism: 3a-3g on the fixed-k matched pool.
# Full walltime is set only after the end-to-end smoke establishes throughput.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -J stage_n_mech
#SBATCH -o results/stage_n/logs/mechanism_%j.out
#SBATCH -e results/stage_n/logs/mechanism_%j.err

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
BEHAVIOR_DIR="${BEHAVIOR_DIR:?set BEHAVIOR_DIR to the passing Step-0 output}"
OUT="results/stage_n/mechanism_${MODE}_${PORTAL_TAG}"
mkdir -p "$OUT"

if [[ "$MODE" == "smoke" ]]; then
  HARVEST_LIMIT=90
  PATCH_LIMIT=8
  MIN_SPLIT=1
  QKOV_LIMIT=8
  N_SEQUENCE=4
  MAX_CLASS=8
  N_RANDOM=2
  N_RESAMPLE=20
  PROBE_ARGS=(--probe-layers=-1)
else
  HARVEST_LIMIT=0
  PATCH_LIMIT=0
  MIN_SPLIT=16
  QKOV_LIMIT=96
  N_SEQUENCE=64
  MAX_CLASS=96
  N_RANDOM=64
  N_RESAMPLE=2000
  PROBE_ARGS=()
fi

echo "host=$(hostname) mode=$MODE portal=$PORTAL_TAG start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SECONDS=0

HARVEST_ARGS=()
PATCH_ARGS=()
if [[ "$HARVEST_LIMIT" -gt 0 ]]; then HARVEST_ARGS+=(--limit "$HARVEST_LIMIT"); fi
if [[ "$PATCH_LIMIT" -gt 0 ]]; then PATCH_ARGS+=(--limit "$PATCH_LIMIT"); fi

"$PY" src/pythia_circuit.py harvest \
  --pool "$BEHAVIOR_DIR/matched_pool.jsonl" \
  --out-dir "$OUT" \
  "${HARVEST_ARGS[@]}"

"$PY" src/pythia_circuit.py analyze \
  --harvest-dir "$OUT" \
  --summary "$OUT/mechanism.summary.json" \
  --probe-rows "$OUT/probe_rows.jsonl" \
  --n-boot "$N_RESAMPLE" \
  --n-shuffle "$N_RESAMPLE" \
  "${PROBE_ARGS[@]}"

"$PY" src/pythia_circuit.py path-patch \
  --pairs "$BEHAVIOR_DIR/pairs.jsonl" \
  --out-dir "$OUT" \
  --batch-size 8 \
  --min-split "$MIN_SPLIT" \
  --n-boot "$N_RESAMPLE" \
  "${PATCH_ARGS[@]}"

"$PY" src/pythia_circuit.py qkov \
  --pool "$BEHAVIOR_DIR/matched_pool.jsonl" \
  --cumulative-summary "$OUT/cumulative_patch.summary.json" \
  --out-dir "$OUT" \
  --limit "$QKOV_LIMIT"

"$PY" src/pythia_circuit.py induction \
  --cumulative-summary "$OUT/cumulative_patch.summary.json" \
  --summary "$OUT/induction.summary.json" \
  --n-sequences "$N_SEQUENCE" \
  --n-boot "$N_RESAMPLE"

"$PY" src/pythia_circuit.py ablate \
  --pool "$BEHAVIOR_DIR/matched_pool.jsonl" \
  --cumulative-summary "$OUT/cumulative_patch.summary.json" \
  --mechanism-summary "$OUT/mechanism.summary.json" \
  --summary "$OUT/ablation.summary.json" \
  --max-per-class "$MAX_CLASS" \
  --n-random "$N_RANDOM" \
  --n-boot "$N_RESAMPLE"

echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
