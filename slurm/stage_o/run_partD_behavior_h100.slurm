#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:12:00
#SBATCH --job-name=o_d_behavior_h
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_h100_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_h100_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "$MODEL_KEY" in
  qwen05b) MODEL=Qwen/Qwen2.5-0.5B; BATCH=128 ;;
  qwen15b) MODEL=Qwen/Qwen2.5-1.5B; BATCH=64 ;;
  qwen3b) MODEL=Qwen/Qwen2.5-3B; BATCH=32 ;;
  mistral7b) MODEL=mistralai/Mistral-7B-v0.3; BATCH=16 ;;
  *) exit 2 ;;
esac
BASE="$ROOT/results/stage_o/partD_${MODEL_KEY}"
OUT="$BASE/mirror_h100"
mkdir -p "$OUT"
cd "$ROOT"
SECONDS=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_eval.py run --model "$MODEL" --dtype bfloat16 \
  --batch-size "$BATCH" --data "$BASE/tasks.jsonl" \
  --out "$OUT/rows.jsonl" --summary "$OUT/behavior.summary.json" \
  --matched-pool "$OUT/unused_pool.jsonl"
"$PY" src/pythia_eval.py run --model "$MODEL" --dtype bfloat16 \
  --batch-size "$BATCH" --data "$BASE/taxonomy_tasks.jsonl" \
  --out "$OUT/taxonomy_rows.jsonl" --summary "$OUT/taxonomy_behavior.summary.json" \
  --matched-pool "$OUT/taxonomy_unused_pool.jsonl"
echo "elapsed_seconds=$SECONDS"
