#!/usr/bin/env bash
# Stage G-1 circuit-level counterfactual pair and path/geometry run.
# Override SMOKE=1 for Qwen2.5-0.5B smoke.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH -J ctxmgmt_stage_g_circuit
#SBATCH -o logs/ctxmgmt_stage_g_circuit_%j.out
#SBATCH -e logs/ctxmgmt_stage_g_circuit_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_g/circuit results/stage_g/figures

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
PORTAL_TAG="${PORTAL_TAG:-ai}"
SKIP_BUILD="${SKIP_BUILD:-0}"
PAIRS_IN="${PAIRS_IN:-}"
if [[ "$SMOKE" == "1" ]]; then
  MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  DTYPE="${DTYPE:-float32}"
  N_PAIRS="${N_PAIRS:-4}"
  MAX_CANDIDATES="${MAX_CANDIDATES:-80}"
  LIMIT="${LIMIT:-4}"
  GEOM_LIMIT="${GEOM_LIMIT:-4}"
  RESIDUAL_STRIDE="${RESIDUAL_STRIDE:-8}"
  OUT_ROOT="${OUT_ROOT:-results/stage_g/circuit/smoke_${PORTAL_TAG}}"
  ARM_L_ADAPTER="${ARM_L_ADAPTER:-}"
else
  MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  DTYPE="${DTYPE:-float32}"
  N_PAIRS="${N_PAIRS:-120}"
  MAX_CANDIDATES="${MAX_CANDIDATES:-5000}"
  LIMIT="${LIMIT:-}"
  GEOM_LIMIT="${GEOM_LIMIT:-120}"
  RESIDUAL_STRIDE="${RESIDUAL_STRIDE:-4}"
  OUT_ROOT="${OUT_ROOT:-results/stage_g/circuit/full_${PORTAL_TAG}}"
  ARM_L_ADAPTER="${ARM_L_ADAPTER:-results/stage_c/adapters_full_ai/arm_l}"
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
ARM_ARGS=()
if [[ -n "$ARM_L_ADAPTER" ]]; then
  ARM_ARGS+=(--arm-l-adapter "$ARM_L_ADAPTER")
else
  ARM_ARGS+=(--arm-l-adapter "")
fi

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "SMOKE=$SMOKE MODEL=$MODEL DTYPE=$DTYPE N_PAIRS=$N_PAIRS OUT_ROOT=$OUT_ROOT SKIP_BUILD=$SKIP_BUILD PAIRS_IN=$PAIRS_IN"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

if [[ "$SKIP_BUILD" == "1" ]]; then
  if [[ -z "$PAIRS_IN" ]]; then
    echo "SKIP_BUILD=1 requires PAIRS_IN" >&2
    exit 2
  fi
  mkdir -p "$OUT_ROOT"
  cp "$PAIRS_IN" "${OUT_ROOT}/pairs.jsonl"
  if [[ -f "${PAIRS_IN%.jsonl}.summary.json" ]]; then
    cp "${PAIRS_IN%.jsonl}.summary.json" "${OUT_ROOT}/pairs.summary.json"
  fi
else
  "$PY" src/stage_g_circuit.py build-pairs \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --n-pairs "$N_PAIRS" \
    --max-candidates "$MAX_CANDIDATES" \
    --out-pairs "${OUT_ROOT}/pairs.jsonl" \
    --out-summary "${OUT_ROOT}/pairs.summary.json"
fi

"$PY" src/stage_g_circuit.py run \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --pairs "${OUT_ROOT}/pairs.jsonl" \
  --patch-rows "${OUT_ROOT}/patch_rows.jsonl" \
  --geometry-rows "${OUT_ROOT}/qk_geometry_rows.jsonl" \
  --ov-rows "${OUT_ROOT}/ov_sanity_rows.jsonl" \
  --summary "${OUT_ROOT}/summary.json" \
  --figure-dir "${OUT_ROOT}/figures" \
  --geometry-limit "$GEOM_LIMIT" \
  --residual-stride "$RESIDUAL_STRIDE" \
  "${LIMIT_ARGS[@]}" \
  "${ARM_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
