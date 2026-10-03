#!/usr/bin/env bash
# Parameterized CPU behavior runner. Override walltime/memory for each full scale.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH -J n_scale_beh
#SBATCH -o results/stage_n/scale_transfer/logs/behavior_%x_%j.out
#SBATCH -e results/stage_n/scale_transfer/logs/behavior_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?submit with MODEL_KEY=410m,1b,or2.8b}"
: "${MODE:?submit with MODE=smoke or full}"
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

case "$MODEL_KEY" in
  410m)
    MODEL=EleutherAI/pythia-410m
    export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    BATCH_SIZE=64
    ;;
  1b)
    MODEL=EleutherAI/pythia-1b
    export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
    unset TRANSFORMERS_CACHE || true
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    BATCH_SIZE=32
    ;;
  2.8b)
    MODEL=EleutherAI/pythia-2.8b
    export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache
    unset TRANSFORMERS_CACHE || true
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    BATCH_SIZE=16
    ;;
  *) echo "unsupported MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac

BASE=results/stage_n/scale_transfer/$MODEL_KEY
if [[ "$MODE" == smoke ]]; then
  DATA=results/stage_n/scale_transfer/smoke_tasks.jsonl
  OUT="$BASE/behavior_smoke"
elif [[ "$MODE" == full ]]; then
  DATA=results/stage_n/behavior_full_cpu/tasks.jsonl
  OUT="$BASE/behavior_full"
else
  echo "unsupported MODE=$MODE" >&2
  exit 2
fi
mkdir -p "$OUT"
SECONDS=0
echo "model=$MODEL mode=$MODE host=$(hostname) start=$(date --iso-8601=seconds)"
"$PY" src/pythia_scale_transfer.py validate-task \
  --data "$DATA" --model "$MODEL" --summary "$OUT/tokenizer_validation.json"
"$PY" src/pythia_eval.py run \
  --data "$DATA" --model "$MODEL" --dtype float32 --batch-size "$BATCH_SIZE" \
  --out "$OUT/rows.jsonl" --summary "$OUT/behavior.summary.json" \
  --matched-pool "$OUT/adaptive_pool_not_for_headline.jsonl"
if [[ "$MODE" == full ]]; then
  "$PY" src/pythia_scale_transfer.py prepare-pool \
    --rows "$OUT/rows.jsonl" --model "$MODEL" \
    --pool "$OUT/matched_pool_k4.jsonl" --summary "$OUT/fixed_k4_pool.summary.json"
fi
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE

