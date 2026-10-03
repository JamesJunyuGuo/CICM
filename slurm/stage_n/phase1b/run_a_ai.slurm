#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --job-name=n_p1b_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/a_ai_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/a_ai_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT_DIR=${OUT_DIR:?set OUT_DIR}
N_RANDOM=${N_RANDOM:-64}

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/results/stage_n/phase1b/logs" "$OUT_DIR"
cd "$ROOT"

"$PY" results/stage_n/phase1b/phase1b.py ablate-heldout \
  --pool results/stage_n/behavior_full_cpu/matched_pool.jsonl \
  --harvest-dir results/stage_n/mechanism_full_cpu \
  --cumulative-summary results/stage_n/mechanism_full_core_ai32/cumulative_patch.summary.json \
  --phase1-ablation results/stage_n/ablation_full_ai32/ablation.summary.json \
  --out-dir "$OUT_DIR" --dtype float32 --batch-size 32 \
  --n-random "$N_RANDOM" --n-boot 2000 --seed 20260809
