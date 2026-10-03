#!/usr/bin/env bash
# Diagnose the Stage-A control failure: A/B the attention kernel on identical data.
# Arm 1: run_eval.py on data/stage_a.jsonl with default sdpa (P0 pipeline).
# Arm 2: same, with attn_implementation=eager (extract_stage_a.py's setting).
# If arm 1 reproduces P0-like accuracy and arm 2 reproduces the extraction's low
# accuracy, the drop is kernel numerics, not data or prompts.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J ctxmgmt_chk
#SBATCH -o logs/ctxmgmt_chk_%j.out
#SBATCH -e logs/ctxmgmt_chk_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results

module purge
module load modtree/gpu
module load conda
export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== arm 1: sdpa (default) ==="
"$PY" src/run_eval.py \
  --data data/stage_a.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out results/stage_a_check_sdpa.jsonl \
  --summary results/stage_a_check_sdpa.summary.json \
  --batch-size 32 --max-new-tokens 16

echo "=== arm 2: eager ==="
"$PY" src/run_eval.py \
  --data data/stage_a.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out results/stage_a_check_eager.jsonl \
  --summary results/stage_a_check_eager.summary.json \
  --batch-size 32 --max-new-tokens 16 \
  --attn-impl eager

echo "DONE"
