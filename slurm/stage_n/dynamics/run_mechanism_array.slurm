#!/usr/bin/env bash
# Stage N Phase 2 induction/QK trajectories on H100s.
# Phase 1's heavier H100 core took 236s; request 6m pending the dedicated smoke.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=00:06:00
#SBATCH --array=0-24%4
#SBATCH -J n2_mechanism
#SBATCH -o results/stage_n/dynamics/logs/mechanism_%A_%a.out
#SBATCH -e results/stage_n/dynamics/logs/mechanism_%A_%a.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
CHECKPOINTS=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 6000 8000 12000 16000 24000 32000 48000 64000 96000 143000)
STEP="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
cd "$ROOT"
module purge
module load modtree/gpu
module load conda

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/stage_n/dynamics/logs

SECONDS=0
echo "host=$(hostname) checkpoint=step${STEP} start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_dynamics.py mechanism \
  --checkpoint "step${STEP}" \
  --out-root results/stage_n/dynamics \
  --dtype float32 \
  --batch-size 24 \
  --n-induction-per-seed 64
echo "elapsed_seconds=$SECONDS checkpoint=step${STEP} end=$(date --iso-8601=seconds)"
echo DONE
