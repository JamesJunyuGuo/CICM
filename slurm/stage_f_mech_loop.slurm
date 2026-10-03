#!/usr/bin/env bash
# Stage F-1 repaired-model mechanism loop.
# Runs adapter-off identity gates, then adapter-attached extractions for Arm L,
# Random-heads, and Generic on the Stage-A stable subset slice.
# Override SMOKE=1 for the Qwen2.5-0.5B adapter path smoke.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH -J ctxmgmt_stage_f_mech
#SBATCH -o logs/ctxmgmt_stage_f_mech_%j.out
#SBATCH -e logs/ctxmgmt_stage_f_mech_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_f/mech_loop results/figures

module purge
module load modtree/gpu
module load conda

export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

SMOKE="${SMOKE:-0}"
DATA="${DATA:-data/stage_a.jsonl}"
SUBSET_INDEX="${SUBSET_INDEX:-results/stage_f/mech_loop/f1_subset_index.jsonl}"
DTYPE="${DTYPE:-float32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

if [[ "$SMOKE" == "1" ]]; then
  MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  ADAPTER_ROOT="${ADAPTER_ROOT:-results/stage_c/adapters_smoke_rerun_ai}"
  OUT_ROOT="${OUT_ROOT:-results/stage_f/mech_loop/smoke_0p5b}"
  LIMIT="${LIMIT:-20}"
  BASELINE_INDEX_FOR_GATE="${OUT_ROOT}/base/extract_index.jsonl"
  SUMMARY_OUT="${SUMMARY_OUT:-${OUT_ROOT}/mech_loop_summary.json}"
  F9_PNG="${F9_PNG:-${OUT_ROOT}/F9_repaired_p_last_gain.png}"
  F9_PDF="${F9_PDF:-${OUT_ROOT}/F9_repaired_p_last_gain.pdf}"
  F9_DATA="${F9_DATA:-${OUT_ROOT}/F9_repaired_p_last_gain.data.json}"
else
  MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  ADAPTER_ROOT="${ADAPTER_ROOT:-results/stage_c/adapters_full_ai}"
  OUT_ROOT="${OUT_ROOT:-results/stage_f/mech_loop/full}"
  LIMIT="${LIMIT:-}"
  BASELINE_INDEX_FOR_GATE="${BASELINE_INDEX_FOR_GATE:-$SUBSET_INDEX}"
  SUMMARY_OUT="${SUMMARY_OUT:-results/stage_f/mech_loop_summary.json}"
  F9_PNG="${F9_PNG:-results/figures/F9_repaired_p_last_gain.png}"
  F9_PDF="${F9_PDF:-results/figures/F9_repaired_p_last_gain.pdf}"
  F9_DATA="${F9_DATA:-results/figures/F9_repaired_p_last_gain.data.json}"
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "SMOKE=$SMOKE"
echo "DATA=$DATA"
echo "SUBSET_INDEX=$SUBSET_INDEX"
echo "ADAPTER_ROOT=$ADAPTER_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "LIMIT=${LIMIT:-}"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

"$PY" src/stage_f_mech_loop.py build-subset \
  --stable-index results/stage_a/extract_stable_index.jsonl \
  --n-correct 200 \
  --out "$SUBSET_INDEX"

run_extract() {
  local arm="$1"
  local kind="$2"
  local mode="$3"
  local out_dir="$4"
  local adapter_dir="$5"
  local limit_override="${6:-$LIMIT}"
  local local_limit_args=()
  if [[ -n "$limit_override" ]]; then
    local_limit_args+=(--limit "$limit_override")
  fi
  mkdir -p "$out_dir"
  echo "=== extract arm=$arm kind=$kind mode=$mode out=$out_dir ==="
  "$PY" src/extract_stage_a.py \
    --data "$DATA" \
    --model "$MODEL" \
    --ids-from-index "$SUBSET_INDEX" \
    --out "${out_dir}/extract.npz" \
    --index-out "${out_dir}/extract_index.jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dtype "$DTYPE" \
    --adapter-dir "$adapter_dir" \
    --adapter-kind "$kind" \
    --adapter-mode "$mode" \
    --no-hidden \
    "${local_limit_args[@]}"
}

if [[ "$SMOKE" == "1" ]]; then
  mkdir -p "${OUT_ROOT}/base"
  echo "=== smoke baseline extraction for identity comparison ==="
  "$PY" src/extract_stage_a.py \
    --data "$DATA" \
    --model "$MODEL" \
    --ids-from-index "$SUBSET_INDEX" \
    --out "${OUT_ROOT}/base/extract.npz" \
    --index-out "${OUT_ROOT}/base/extract_index.jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dtype "$DTYPE" \
    --no-hidden \
    "${LIMIT_ARGS[@]}"
fi

run_extract arm_l head_sliced off "${OUT_ROOT}/identity_arm_l_off" "${ADAPTER_ROOT}/arm_l" 20
"$PY" src/stage_f_mech_loop.py verify-identity \
  --baseline-index "$BASELINE_INDEX_FOR_GATE" \
  --candidate-index "${OUT_ROOT}/identity_arm_l_off/extract_index.jsonl" \
  --n 20 \
  --out "${OUT_ROOT}/identity_arm_l_off.json"

run_extract random_heads random_head_sliced off "${OUT_ROOT}/identity_random_heads_off" "${ADAPTER_ROOT}/random_heads" 20
"$PY" src/stage_f_mech_loop.py verify-identity \
  --baseline-index "$BASELINE_INDEX_FOR_GATE" \
  --candidate-index "${OUT_ROOT}/identity_random_heads_off/extract_index.jsonl" \
  --n 20 \
  --out "${OUT_ROOT}/identity_random_heads_off.json"

run_extract generic generic off "${OUT_ROOT}/identity_generic_off" "${ADAPTER_ROOT}/generic" 20
"$PY" src/stage_f_mech_loop.py verify-identity \
  --baseline-index "$BASELINE_INDEX_FOR_GATE" \
  --candidate-index "${OUT_ROOT}/identity_generic_off/extract_index.jsonl" \
  --n 20 \
  --out "${OUT_ROOT}/identity_generic_off.json"

run_extract arm_l head_sliced on "${OUT_ROOT}/arm_l" "${ADAPTER_ROOT}/arm_l"
run_extract random_heads random_head_sliced on "${OUT_ROOT}/random_heads" "${ADAPTER_ROOT}/random_heads"
run_extract generic generic on "${OUT_ROOT}/generic" "${ADAPTER_ROOT}/generic"

if [[ "$SMOKE" == "1" ]]; then
  echo "SMOKE extraction path and adapter-off identity gates completed"
  echo "END=$(date --iso-8601=seconds)"
  echo "ELAPSED_SECONDS=$SECONDS"
  echo "DONE"
  exit 0
fi

"$PY" src/stage_f_mech_loop.py analyze \
  --arm_l-npz "${OUT_ROOT}/arm_l/extract.npz" \
  --arm_l-index "${OUT_ROOT}/arm_l/extract_index.jsonl" \
  --random_heads-npz "${OUT_ROOT}/random_heads/extract.npz" \
  --random_heads-index "${OUT_ROOT}/random_heads/extract_index.jsonl" \
  --generic-npz "${OUT_ROOT}/generic/extract.npz" \
  --generic-index "${OUT_ROOT}/generic/extract_index.jsonl" \
  --out "$SUMMARY_OUT" \
  --f9-png "$F9_PNG" \
  --f9-pdf "$F9_PDF" \
  --f9-data "$F9_DATA"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
