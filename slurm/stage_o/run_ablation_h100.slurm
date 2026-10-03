#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --job-name=o_a_ablate
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/ablate_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/ablate_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY=410m,1.4b,or2.8b}"
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "$MODEL_KEY" in
  410m) MODEL=EleutherAI/pythia-410m; export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface; export TRANSFORMERS_CACHE="$HF_HOME/transformers"; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; BATCH=64 ;;
  1.4b) MODEL=EleutherAI/pythia-1.4b; export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache; unset TRANSFORMERS_CACHE || true; export HUGGINGFACE_HUB_CACHE="$HF_HOME"; BATCH=32 ;;
  2.8b) MODEL=EleutherAI/pythia-2.8b; export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache; unset TRANSFORMERS_CACHE || true; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; BATCH=16 ;;
  *) echo "unsupported MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
BASE="$ROOT/results/stage_o/$MODEL_KEY"
mkdir -p "$BASE/ablation" "$ROOT/results/stage_o/logs"
cd "$ROOT"
SECONDS=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_crossscale.py ablate-heldout \
  --pool "$BASE/behavior/matched_pool.jsonl" \
  --harvest-dir "$BASE/mechanism" \
  --cumulative-summary "$BASE/circuit/cumulative_patch.summary.json" \
  --out-dir "$BASE/ablation" --model "$MODEL" --dtype bfloat16 \
  --batch-size "$BATCH" --max-heads 12 --n-random 64 \
  --n-boot 2000 --seed 20260818
echo "elapsed_seconds=$SECONDS"
