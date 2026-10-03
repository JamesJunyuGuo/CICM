#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=o_c15_report
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_report_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_report_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
"$PY" src/qwen_small_circuit.py report-part-c \
  --root results/stage_o/partC_qwen15b
