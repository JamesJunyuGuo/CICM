#!/bin/bash
#SBATCH --job-name=n_sink_q15_prepare
#SBATCH --account=cis250190
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen15_prepare_cpu_%j.out
#SBATCH --error=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen15_prepare_cpu_%j.err

set -euo pipefail

module purge
module load conda

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT="$PROJ/results/stage_n/attention_sink/qwen15_prepare_cpu"

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

echo "start=$(date --iso-8601=seconds) host=$(hostname)"
"$PY" -u src/stage_n_attention_sink.py prepare \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --per-cell 2 \
  --out "$OUT/tasks.jsonl" \
  --summary "$OUT/tasks.summary.json"
echo "end=$(date --iso-8601=seconds)"
