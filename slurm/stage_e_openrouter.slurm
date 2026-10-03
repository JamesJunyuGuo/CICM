#!/usr/bin/env bash
# Stage E OpenRouter cross-scale sweep. CPU/shared portal; no GPU request.
# Run with DRY_RUN=1 first for the mandatory budget estimate.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J ctxmgmt_stage_e_or
#SBATCH -o logs/ctxmgmt_stage_e_or_%j.out
#SBATCH -e logs/ctxmgmt_stage_e_or_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_e/openrouter

module purge
module load modtree/cpu
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
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

OUT="${OUT:-results/stage_e/openrouter/sweep.jsonl}"
SUMMARY="${SUMMARY:-results/stage_e/openrouter/sweep.summary.json}"
BUDGET="${BUDGET:-results/stage_e/openrouter/budget.json}"
SUBSAMPLE_OUT="${SUBSAMPLE_OUT:-results/stage_e/openrouter/subsample.jsonl}"
PER_CELL="${PER_CELL:-50}"
SEED="${SEED:-41}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CONCURRENCY="${CONCURRENCY:-4}"
TIMEOUT="${TIMEOUT:-60}"
DRY_RUN="${DRY_RUN:-0}"

DRY_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  DRY_ARGS+=(--dry-run)
fi

echo "host=$(hostname)"
echo "OPENROUTER_API_KEY env present? $([[ -n "${OPENROUTER_API_KEY:-}" ]] && echo yes || echo no)"
echo "OUT=$OUT"
echo "SUMMARY=$SUMMARY"
echo "BUDGET=$BUDGET"
echo "SUBSAMPLE_OUT=$SUBSAMPLE_OUT"
echo "PER_CELL=$PER_CELL"
echo "SEED=$SEED"
echo "MAX_TOKENS=$MAX_TOKENS"
echo "CONCURRENCY=$CONCURRENCY"
echo "TIMEOUT=$TIMEOUT"
echo "DRY_RUN=$DRY_RUN"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

time "$PY" src/openrouter_sweep.py \
  --out "$OUT" \
  --summary "$SUMMARY" \
  --budget "$BUDGET" \
  --subsample-out "$SUBSAMPLE_OUT" \
  --per-cell "$PER_CELL" \
  --seed "$SEED" \
  --max-tokens "$MAX_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  "${DRY_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
