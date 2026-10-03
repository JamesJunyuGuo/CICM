#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=o_dg_prepare
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_gated_prepare_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_gated_prepare_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
set +u
source "$HOME/.bashrc"
set -u
: "${HF_TOKEN:?HF_TOKEN must be present in the compute-node environment}"
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
case "$MODEL_KEY" in
  llama32_1b) MODEL=meta-llama/Llama-3.2-1B ;;
  gemma2_2b) MODEL=google/gemma-2-2b ;;
  gemma2_9b) MODEL=google/gemma-2-9b ;;
  *) exit 2 ;;
esac
OUT="$ROOT/results/stage_o/partD_${MODEL_KEY}"
mkdir -p "$OUT" "$ROOT/results/stage_o/dependencies"
cd "$ROOT"
"$PY" src/qwen_small_circuit.py cache-model \
  --model "$MODEL" --cache-dir "$HF_HOME/hub" \
  --summary "results/stage_o/dependencies/${MODEL_KEY}_cache.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
