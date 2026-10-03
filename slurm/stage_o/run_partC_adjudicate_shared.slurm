#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=o_c_gate
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_gate_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_gate_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
"$PY" src/qwen_small_circuit.py adjudicate-recon \
  --rows-05 results/stage_o/partC_qwen05b/recon_rows.jsonl \
  --rows-15 results/stage_o/partC_qwen15b/recon_rows.jsonl \
  --noop-summary results/stage_o/partC_qwen05b/hooked_noop_cpu.summary.json \
  --summary results/stage_o/partC_step0.summary.json \
  --report results/stage_o/partC_REPORT.md
