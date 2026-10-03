#!/bin/bash
# Diagnose the Gemma A100 no-op identity failure. This runs only baseline and
# gate_scale=0 generation on the frozen confirmation rows, then exits.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=00:20:00
#SBATCH --job-name=r14_g9_id_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_gemma2_9b_it_identity_diag_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_gemma2_9b_it_identity_diag_a100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=google/gemma-2-9b-it
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_llama_causal_heads.py \
  --confirmation-only --identity-diagnostic-only \
  --frozen-summary results/stage_r/crossmodel/gemma2_9b_it/r14_causal_calibration_h100/summary.json \
  --model "$MODEL" --model-label google/gemma-2-9b-it \
  --model-family gemma2 --stage-label R14-gemma2-9b-it \
  --out-dir results/stage_r/crossmodel/gemma2_9b_it/r14_identity_diag_a100 \
  --minimum-pool 100 --minimum-preservation 0.95 \
  --n-random-positions 16 --n-random-heads 64 --n-boot 2000 \
  --batch-size 3 --max-new-tokens 8
