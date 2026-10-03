#!/bin/bash
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --array=0-5
#SBATCH --job-name=n_p1b_bvshard_dup
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_validate_shard_gpu_%A_%a.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_n/phase1b/logs/b_validate_shard_gpu_%A_%a.err

# A100 shard duplicate uses a 1.6x allowance over the H100 shard request.
set -euo pipefail
module purge
module load modtree/gpu
module load conda

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
OUT_ROOT=$ROOT/results/stage_n/phase1b/b_validate_shards_gpu
OFFSET=$((SLURM_ARRAY_TASK_ID * 16))

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/results/stage_n/phase1b/logs" "$OUT_ROOT/shard_$SLURM_ARRAY_TASK_ID"
cd "$ROOT"

"$PY" results/stage_n/phase1b/phase1b.py upstream-validate \
  --pairs results/stage_n/behavior_full_cpu/pairs.jsonl \
  --ranking results/stage_n/phase1b/upstream_ranking.json \
  --out-dir "$OUT_ROOT/shard_$SLURM_ARRAY_TASK_ID" \
  --dtype float32 --batch-size 16 --offset "$OFFSET" --limit 16 --n-random 64
