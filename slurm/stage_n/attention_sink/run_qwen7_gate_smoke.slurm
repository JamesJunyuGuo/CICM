#!/bin/bash
# Qwen-7B mechanism gate only. The 1.5B A100 pipeline took 4:40 including
# behavior and pairing; this job does only four-pair capture/patching.
#SBATCH --job-name=n_sink_q7_gate
#SBATCH --account=cis250190-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:12:00
#SBATCH --output=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen7_gate_ai_%j.out
#SBATCH --error=/anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/attention_sink/logs/qwen7_gate_ai_%j.err

set -euo pipefail

module purge
module load modtree/gpu
module load conda

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
PAIRS="$PROJ/results/stage_n/attention_sink/qwen15_smoke_a100_r2/pairs.jsonl"
OUT="$PROJ/results/stage_n/attention_sink/qwen7_gate_smoke_ai"

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

"$PY" -u src/stage_n_attention_sink.py smoke-mechanism \
  --model Qwen/Qwen2.5-7B-Instruct \
  --pairs "$PAIRS" \
  --out "$OUT/patch_rows.jsonl" \
  --summary "$OUT/smoke.summary.json" \
  --limit 4

echo "end=$(date --iso-8601=seconds)"
