#!/usr/bin/env bash
# CPU grouped probe and vectorized bootstrap/permutation analysis.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -J n_scale_analysis
#SBATCH -o results/stage_n/scale_transfer/logs/analysis_%x_%j.out
#SBATCH -e results/stage_n/scale_transfer/logs/analysis_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?submit with MODEL_KEY=160m,410m,1b,or2.8b}"
: "${MECH_DIR:?submit with the completed mechanism harvest directory}"
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL=EleutherAI/pythia-$MODEL_KEY
OUT=results/stage_n/scale_transfer/$MODEL_KEY/analysis
mkdir -p "$OUT"
SECONDS=0
"$PY" src/pythia_scale_transfer.py analyze-probe \
  --harvest-dir "$MECH_DIR" --model "$MODEL" \
  --summary "$OUT/probe.summary.json" --probe-rows "$OUT/probe_rows.jsonl" \
  --n-boot 2000 --n-shuffle 2000
"$PY" src/pythia_scale_transfer.py analyze-heads \
  --harvest-dir "$MECH_DIR" --model "$MODEL" \
  --summary "$OUT/heads.summary.json" --n-boot 2000 --n-shuffle 2000
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
