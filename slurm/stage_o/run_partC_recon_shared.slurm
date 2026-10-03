#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=o_c_recon_cpu
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_recon_cpu_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_recon_cpu_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY=qwen05b or qwen15b}"
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "$MODEL_KEY" in
  qwen05b) MODEL=Qwen/Qwen2.5-0.5B; BATCH=32 ;;
  qwen15b) MODEL=Qwen/Qwen2.5-1.5B; BATCH=16 ;;
  *) exit 2 ;;
esac
OUT="$ROOT/results/stage_o/partC_${MODEL_KEY}"
cd "$ROOT"
SECONDS=0
"$PY" src/pythia_eval.py run --model "$MODEL" --dtype float32 \
  --batch-size "$BATCH" --data "$OUT/recon_tasks.jsonl" \
  --out "$OUT/recon_rows.jsonl" --summary "$OUT/recon_behavior.summary.json" \
  --matched-pool "$OUT/recon_unused_pool.jsonl"
echo "elapsed_seconds=$SECONDS"
