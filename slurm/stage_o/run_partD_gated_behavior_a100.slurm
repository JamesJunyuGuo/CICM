#!/bin/bash
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --job-name=o_dg_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_gated_a100_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_gated_a100_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
set +u
source "$HOME/.bashrc"
set -u
: "${HF_TOKEN:?HF_TOKEN must be present in the compute-node environment}"
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "$MODEL_KEY" in
  llama32_1b) MODEL=meta-llama/Llama-3.2-1B; BATCH=64 ;;
  gemma2_2b) MODEL=google/gemma-2-2b; BATCH=32 ;;
  gemma2_9b) MODEL=google/gemma-2-9b; BATCH=8 ;;
  *) exit 2 ;;
esac
BASE="$ROOT/results/stage_o/partD_${MODEL_KEY}"
OUT="$BASE/mirror_a100"
mkdir -p "$OUT"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_eval.py run --model "$MODEL" --dtype bfloat16 \
  --batch-size "$BATCH" --data "$BASE/tasks.jsonl" \
  --out "$OUT/rows.jsonl" --summary "$OUT/behavior.summary.json" \
  --matched-pool "$OUT/unused_pool.jsonl"
"$PY" src/pythia_eval.py run --model "$MODEL" --dtype bfloat16 \
  --batch-size "$BATCH" --data "$BASE/taxonomy_tasks.jsonl" \
  --out "$OUT/taxonomy_rows.jsonl" --summary "$OUT/taxonomy_behavior.summary.json" \
  --matched-pool "$OUT/taxonomy_unused_pool.jsonl"
