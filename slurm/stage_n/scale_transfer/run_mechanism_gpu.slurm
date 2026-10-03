#!/usr/bin/env bash
# Lean final-state + exhaustive attention/QK harvest for one Pythia scale.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:15:00
#SBATCH -J n_scale_mech
#SBATCH -o results/stage_n/scale_transfer/logs/mechanism_%x_%j.out
#SBATCH -e results/stage_n/scale_transfer/logs/mechanism_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?submit with MODEL_KEY=160m,410m,1b,or2.8b}"
: "${MODE:?submit with MODE=smoke or full}"
PORTAL_TAG=${PORTAL_TAG:-ai}
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load modtree/gpu
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

case "$MODEL_KEY" in
  160m|410m|1b)
    MODEL=EleutherAI/pythia-$MODEL_KEY
    export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
    if [[ "$MODEL_KEY" == 1b ]]; then
      unset TRANSFORMERS_CACHE || true
    else
      export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    fi
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    ;;
  2.8b)
    MODEL=EleutherAI/pythia-2.8b
    export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache
    unset TRANSFORMERS_CACHE || true
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    ;;
  *) echo "unsupported MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac

if [[ "$MODEL_KEY" == 160m ]]; then
  POOL=results/stage_n/behavior_full_cpu/matched_pool.jsonl
else
  POOL=results/stage_n/scale_transfer/$MODEL_KEY/behavior_full/matched_pool_k4.jsonl
fi
if [[ ! -s "$POOL" ]]; then
  echo "missing passed fixed-k pool: $POOL" >&2
  exit 3
fi
OUT=results/stage_n/scale_transfer/$MODEL_KEY/mechanism_${MODE}_${PORTAL_TAG}
mkdir -p "$OUT"
LIMIT_ARGS=()
if [[ "$MODE" == smoke ]]; then
  LIMIT_ARGS=(--limit 12)
elif [[ "$MODE" != full ]]; then
  echo "unsupported MODE=$MODE" >&2
  exit 2
fi

SECONDS=0
echo "model=$MODEL mode=$MODE portal=$PORTAL_TAG host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_scale_transfer.py harvest \
  --pool "$POOL" --out-dir "$OUT" --model "$MODEL" --dtype float32 "${LIMIT_ARGS[@]}"
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE

