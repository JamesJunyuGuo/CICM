#!/usr/bin/env bash
# Stage N Phase 2 induction/QK hook smoke after the CPU download smoke.
# Phase 1's larger H100 core took 236s; this reduced pass requests 6m.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=00:06:00
#SBATCH -J n2_smoke_gpu
#SBATCH -o results/stage_n/dynamics/logs/smoke_gpu_%j.out
#SBATCH -e results/stage_n/dynamics/logs/smoke_gpu_%j.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
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
mkdir -p results/stage_n/dynamics/logs results/stage_n/dynamics/smoke_mechanism

SECONDS=0
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_dynamics.py mechanism \
  --checkpoint step0 \
  --out-root results/stage_n/dynamics/smoke_mechanism \
  --dtype float32 \
  --batch-size 12 \
  --n-induction-per-seed 4 \
  --qk-limit 24
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
