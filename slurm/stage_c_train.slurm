#!/usr/bin/env bash
# Stage C adapter training. Override ARM/MODEL/LIMIT/MAX_STEPS/OUT_DIR for smokes.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH -J ctxmgmt_stage_c_train
#SBATCH -o logs/ctxmgmt_stage_c_train_%j.out
#SBATCH -e logs/ctxmgmt_stage_c_train_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_c/adapters

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

ARM="${ARM:?ARM must be arm_l, generic, or random_heads}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-data/stage_c/train_overwrite.jsonl}"
OUT_DIR="${OUT_DIR:-results/stage_c/adapters/${ARM}}"
RANK="${RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-2.0}"
LR="${LR:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-10}"
RANDOM_SEED="${RANDOM_SEED:-123}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
MAX_STEPS_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  MAX_STEPS_ARGS+=(--max-steps "$MAX_STEPS")
fi

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "ARM=$ARM"
echo "MODEL=$MODEL"
echo "TRAIN_DATA=$TRAIN_DATA"
echo "OUT_DIR=$OUT_DIR"
echo "LIMIT=${LIMIT:-}"
echo "MAX_STEPS=${MAX_STEPS:-}"
echo "SEED=$SEED"
echo "RANDOM_SEED=$RANDOM_SEED"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

time "$PY" src/train_stage_c.py \
  --arm "$ARM" \
  --model "$MODEL" \
  --train-data "$TRAIN_DATA" \
  --out-dir "$OUT_DIR" \
  --rank "$RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --max-length "$MAX_LENGTH" \
  --seed "$SEED" \
  --random-seed "$RANDOM_SEED" \
  "${LIMIT_ARGS[@]}" \
  "${MAX_STEPS_ARGS[@]}"

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
