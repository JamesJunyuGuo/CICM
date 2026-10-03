#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --job-name=o_d_prepare
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_prepare_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_prepare_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
for KEY in qwen05b qwen15b qwen3b mistral7b; do
  case "$KEY" in
    qwen05b) MODEL=Qwen/Qwen2.5-0.5B ;;
    qwen15b) MODEL=Qwen/Qwen2.5-1.5B ;;
    qwen3b) MODEL=Qwen/Qwen2.5-3B ;;
    mistral7b) MODEL=mistralai/Mistral-7B-v0.3 ;;
  esac
  OUT="results/stage_o/partD_${KEY}"
  mkdir -p "$OUT"
  "$PY" src/qwen_small_circuit.py tokenizer-audit --stage O-Part-D-tokenizer \
    --model "$MODEL" --summary "$OUT/tokenizer_gate.summary.json"
  "$PY" src/qwen_small_circuit.py generate-data --stage O-Part-D-full-data \
    --model "$MODEL" --audit "$OUT/tokenizer_gate.summary.json" \
    --out "$OUT/tasks.jsonl" --summary "$OUT/tasks.summary.json"
  K=$(jq -r '[.caps.k_cap,4]|min' "$OUT/tokenizer_gate.summary.json")
  "$PY" src/qwen_small_circuit.py generate-data --stage O-Part-D-taxonomy-data \
    --model "$MODEL" --audit "$OUT/tokenizer_gate.summary.json" \
    --variants multi --k-values "$K" \
    --out "$OUT/taxonomy_tasks.jsonl" --summary "$OUT/taxonomy_tasks.summary.json"
done
