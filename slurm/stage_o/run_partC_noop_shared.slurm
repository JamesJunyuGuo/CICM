#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --job-name=o_c_noop
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_noop_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_noop_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=""
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
"$PY" src/qwen_small_circuit.py hooked-noop \
  --model Qwen/Qwen2.5-0.5B \
  --data results/stage_o/partC_qwen05b/recon_tasks.jsonl --limit 2 \
  --summary results/stage_o/partC_qwen05b/hooked_noop_cpu.summary.json
