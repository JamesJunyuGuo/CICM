#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=o_a_data
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/data_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/data_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$ROOT/results/stage_o/logs" "$ROOT/results/stage_o/data"
cd "$ROOT"

SECONDS=0
"$PY" src/pythia_crossscale.py generate-data \
  --model EleutherAI/pythia-410m \
  --out results/stage_o/data/tasks_n15.jsonl \
  --summary results/stage_o/data/tasks_n15.summary.json \
  --seeds 11,29,47 --k-values 0,2,4,6,8,10,12,14 \
  --templates arrow,current,latest \
  --n-per-seed-cell 72 --n-lines 15
echo "elapsed_seconds=$SECONDS"
