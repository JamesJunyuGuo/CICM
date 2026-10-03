#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=o_d_report
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_report_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_report_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" MPLCONFIGDIR=/tmp/mpl-stage-o-d
cd "$ROOT"
"$PY" src/qwen_small_circuit.py summarize-part-d \
  --root results/stage_o --out-dir results/stage_o/partD_summary
