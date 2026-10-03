#!/bin/bash
#SBATCH --account=cis250190
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:12:00
#SBATCH --job-name=p_trace_analyze
#SBATCH --output=results/stage_p/logs/p_trace_analyze_%j.out
#SBATCH --error=results/stage_p/logs/p_trace_analyze_%j.err

set -euo pipefail
: "${CAPTURE_DIR:?set CAPTURE_DIR}"
: "${ANALYSIS_DIR:?set ANALYSIS_DIR}"
: "${MODEL_LABEL:?set MODEL_LABEL}"
module purge
module load conda
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$PROJ"
"$PY" -u src/genattn_analyze.py analyze \
  --capture-dir "$CAPTURE_DIR" \
  --out-dir "$ANALYSIS_DIR" \
  --model-label "$MODEL_LABEL" \
  --n-shuffle 1000 --predictor-shuffles 1000 --n-boot 2000
