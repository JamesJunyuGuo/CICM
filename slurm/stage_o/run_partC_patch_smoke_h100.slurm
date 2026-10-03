#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:08:00
#SBATCH --job-name=o_c15_psmoke
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_psmoke_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_psmoke_%j.err

set -euo pipefail
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
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OUT_DIR=${OUT_DIR:-results/stage_o/partC_qwen15b/patch_smoke}
"$PY" src/qwen_small_circuit.py path-patch-part-c \
  --model Qwen/Qwen2.5-1.5B --dtype bfloat16 --batch-size 4 --min-split 1 \
  --n-boot 200 --pairs results/stage_o/partC_qwen15b/pairs/exact_pairs_smoke.jsonl \
  --out-dir "$OUT_DIR"
