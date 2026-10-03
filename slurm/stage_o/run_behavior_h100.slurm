#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=o_a_behavior
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/behavior_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/behavior_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY=410m,1.4b,or2.8b}"
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

case "$MODEL_KEY" in
  410m)
    MODEL=EleutherAI/pythia-410m
    export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    BATCH=128
    ;;
  1.4b)
    MODEL=EleutherAI/pythia-1.4b
    export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache
    unset TRANSFORMERS_CACHE || true
    export HUGGINGFACE_HUB_CACHE="$HF_HOME"
    BATCH=64
    ;;
  2.8b)
    MODEL=EleutherAI/pythia-2.8b
    export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache
    unset TRANSFORMERS_CACHE || true
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    BATCH=32
    ;;
  *) echo "unsupported MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac

OUT="$ROOT/results/stage_o/$MODEL_KEY/behavior"
mkdir -p "$ROOT/results/stage_o/logs" "$OUT"
cd "$ROOT"
SECONDS=0
echo "model=$MODEL host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_scale_transfer.py validate-task \
  --data results/stage_o/data/tasks_n15.jsonl --model "$MODEL" \
  --summary "$OUT/tokenizer_validation.json"
"$PY" src/pythia_eval.py run \
  --data results/stage_o/data/tasks_n15.jsonl --model "$MODEL" \
  --dtype float32 --batch-size "$BATCH" \
  --out "$OUT/rows.jsonl" --summary "$OUT/behavior.summary.json" \
  --matched-pool "$OUT/unused_adaptive_pool.jsonl"
"$PY" src/pythia_crossscale.py adjudicate-behavior \
  --rows "$OUT/rows.jsonl" --model "$MODEL" \
  --summary "$OUT/adjudication.summary.json" --pool "$OUT/matched_pool.jsonl"
STATUS=$(jq -r .status "$OUT/adjudication.summary.json")
if [[ "$STATUS" == mechanism_ready ]]; then
  "$PY" src/pythia_eval.py build-pairs \
    --rows "$OUT/matched_pool.jsonl" --model "$MODEL" --dtype float32 \
    --out "$OUT/pairs.jsonl" --summary "$OUT/pairs.summary.json" \
    --n-pairs 96 --min-pairs 90 --max-candidates 500 --batch-size "$BATCH"
else
  printf '%s\n' "$STATUS" > "$OUT/MECHANISM_NOT_RUN.status"
fi
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
