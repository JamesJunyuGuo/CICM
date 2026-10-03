#!/bin/bash
# Duplicate-safe A100 mirror of the 216-row assistant-prefill boundary smoke.
#SBATCH --job-name=n_sink_q7_pre_smoke_a100
#SBATCH --account=cis250190-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen7_prefill_smoke_a100_%j.out
#SBATCH --error=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen7_prefill_smoke_a100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT="$PROJ/results/stage_n/attention_sink/qwen7_prefill_boundary_smoke_a100"

export PYTHONPATH="$PROJ/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
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
  --model Qwen/Qwen2.5-7B-Instruct --per-cell 4 \
  --out "$OUT/tasks.jsonl" --summary "$OUT/tasks.summary.json"
"$PY" -u src/stage_n_attention_sink.py behavior \
  --model Qwen/Qwen2.5-7B-Instruct --data "$OUT/tasks.jsonl" \
  --out "$OUT/behavior.jsonl" --summary "$OUT/behavior.summary.json" \
  --batch-size 64
"$PY" -u src/stage_n_attention_sink.py boundary-gate \
  --behavior-summary "$OUT/behavior.summary.json" \
  --out "$OUT/boundary_gate.summary.json"
echo "end=$(date --iso-8601=seconds)"
