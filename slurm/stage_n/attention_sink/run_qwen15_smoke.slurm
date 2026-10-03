#!/bin/bash
#SBATCH --job-name=n_sink_q15_smoke
#SBATCH --account=cis250190-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen15_smoke_r2_%j.out
#SBATCH --error=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen15_smoke_r2_%j.err

set -euo pipefail

module purge
module load modtree/gpu
module load conda

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT="$PROJ/results/stage_n/attention_sink/qwen15_smoke_r2"

export PYTHONPATH="$PROJ/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
unset TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT" "$PROJ/results/stage_n/attention_sink/logs"
cd "$PROJ"

echo "start=$(date --iso-8601=seconds) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -u src/stage_n_attention_sink.py prepare \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --per-cell 2 \
  --out "$OUT/tasks.jsonl" \
  --summary "$OUT/tasks.summary.json"

"$PY" -u src/stage_n_attention_sink.py behavior \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data "$OUT/tasks.jsonl" \
  --out "$OUT/behavior.jsonl" \
  --summary "$OUT/behavior.summary.json" \
  --batch-size 64

"$PY" -u src/stage_n_attention_sink.py pair \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --behavior "$OUT/behavior.jsonl" \
  --all-out "$OUT/pairs_all.jsonl" \
  --out "$OUT/pairs.jsonl" \
  --summary "$OUT/pairs.summary.json" \
  --smoke-any-pairs 4 \
  --batch-size 64

"$PY" -u src/stage_n_attention_sink.py smoke-mechanism \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --pairs "$OUT/pairs.jsonl" \
  --out "$OUT/patch_rows.jsonl" \
  --summary "$OUT/smoke.summary.json" \
  --limit 4

echo "end=$(date --iso-8601=seconds)"
