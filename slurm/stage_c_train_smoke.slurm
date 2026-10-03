#!/usr/bin/env bash
# Stage C 0.5B training smoke for all adapter arms. Runs a few steps per arm.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -J ctxmgmt_stage_c_train_smoke
#SBATCH -o logs/ctxmgmt_stage_c_train_smoke_%j.out
#SBATCH -e logs/ctxmgmt_stage_c_train_smoke_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_c/adapters_smoke

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

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LIMIT="${LIMIT:-24}"
MAX_STEPS="${MAX_STEPS:-3}"
SMOKE_ROOT="${SMOKE_ROOT:-results/stage_c/adapters_smoke}"

echo "host=$(hostname) gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODEL=$MODEL LIMIT=$LIMIT MAX_STEPS=$MAX_STEPS"
echo "SMOKE_ROOT=$SMOKE_ROOT"
echo "REQUIRE_ARM_T_SUMMARY=${REQUIRE_ARM_T_SUMMARY:-}"
echo "START=$(date --iso-8601=seconds)"
SECONDS=0

if [[ -n "${REQUIRE_ARM_T_SUMMARY:-}" ]]; then
  "$PY" src/check_stage_c_arm_t_gate.py --summary "$REQUIRE_ARM_T_SUMMARY"
fi

for ARM in arm_l generic random_heads; do
  OUT_DIR="${SMOKE_ROOT}/${ARM}"
  echo "=== smoke train $ARM ==="
  "$PY" src/train_stage_c.py \
    --arm "$ARM" \
    --model "$MODEL" \
    --train-data data/stage_c/train_overwrite.jsonl \
    --out-dir "$OUT_DIR" \
    --limit "$LIMIT" \
    --max-steps "$MAX_STEPS" \
    --batch-size 1 \
    --grad-accum 1 \
    --rank 4 \
    --lora-alpha 4 \
    --max-length 768 \
    --log-every 1
done

echo "END=$(date --iso-8601=seconds)"
echo "ELAPSED_SECONDS=$SECONDS"
echo "DONE"
