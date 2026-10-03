#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --job-name=o_c15_analyze
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_analyze_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_analyze_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
cd "$ROOT"
"$PY" src/qwen_small_circuit.py analyze-part-c \
  --model Qwen/Qwen2.5-1.5B \
  --harvest-dir results/stage_o/partC_qwen15b/mechanism \
  --summary results/stage_o/partC_qwen15b/mechanism/mechanism.summary.json \
  --probe-rows results/stage_o/partC_qwen15b/mechanism/probe_rows.jsonl \
  --n-boot 2000 --n-shuffle 2000
