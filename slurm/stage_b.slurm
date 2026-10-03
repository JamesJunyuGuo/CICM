#!/usr/bin/env bash
# Stage B intervention: oracle attention-logit bias toward write spans.
# Defaults run the full Qwen2.5-7B-Instruct fp32 job. Override MODEL/LIMIT/ALPHAS
# for the required 0.5B smoke, e.g. MODEL=Qwen/Qwen2.5-0.5B-Instruct LIMIT=10 ALPHAS=1,4.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:15:00
#SBATCH -J ctxmgmt_stage_b
#SBATCH -o logs/ctxmgmt_stage_b_%j.out
#SBATCH -e logs/ctxmgmt_stage_b_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_b/figs

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

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA="${DATA:-data/stage_a.jsonl}"
EXTRACT_INDEX="${EXTRACT_INDEX:-results/stage_a/extract_index.jsonl}"
STABLE_LABELS="${STABLE_LABELS:-results/stage_a/stable_labels.jsonl}"
A1_SUMMARY="${A1_SUMMARY:-results/stage_a/a1_stable_summary.json}"
OUT="${OUT:-results/stage_b/interventions.jsonl}"
SUMMARY="${SUMMARY:-results/stage_b/summary.json}"
REPORT="${REPORT:-results/stage_b/REPORT.md}"
FIG_DIR="${FIG_DIR:-results/stage_b/figs}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DTYPE="${DTYPE:-float32}"
ALPHAS="${ALPHAS:-2,4,8}"
SEED="${SEED:-1}"
WRONG_FILTER="${WRONG_FILTER:-any}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "EXTRACT_INDEX=$EXTRACT_INDEX"
echo "STABLE_LABELS=$STABLE_LABELS"
echo "A1_SUMMARY=$A1_SUMMARY"
echo "OUT=$OUT"
echo "SUMMARY=$SUMMARY"
echo "REPORT=$REPORT"
echo "FIG_DIR=$FIG_DIR"
echo "LIMIT=${LIMIT:-}"
echo "ALPHAS=$ALPHAS"
echo "DTYPE=$DTYPE"
echo "WRONG_FILTER=$WRONG_FILTER"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

time "$PY" src/intervene_stage_b.py \
  --data "$DATA" \
  --extract-index "$EXTRACT_INDEX" \
  --stable-labels "$STABLE_LABELS" \
  --a1-summary "$A1_SUMMARY" \
  --model "$MODEL" \
  --out "$OUT" \
  --summary "$SUMMARY" \
  --report "$REPORT" \
  --fig-dir "$FIG_DIR" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --alphas "$ALPHAS" \
  --seed "$SEED" \
  --wrong-filter "$WRONG_FILTER" \
  "${LIMIT_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
