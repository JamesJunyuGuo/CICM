#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=o_d_race_report
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_race_report_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partD_race_report_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src" MPLCONFIGDIR=/tmp/mpl-stage-o-d-race
RACE_DIR="$ROOT/results/stage_o/portal_race"

for MODEL_KEY in qwen05b qwen15b qwen3b mistral7b; do
  MARKER="$RACE_DIR/partD_${MODEL_KEY}.winner"
  [[ -s "$MARKER" ]] || { echo "missing $MARKER" >&2; exit 4; }
  WINNER=$(awk -F= '$1 == "winner" {print $2}' "$MARKER")
  WIN_JOB=$(awk -F= '$1 == "winner_job" {print $2}' "$MARKER")
  STATE=$(sacct -X -n -P -j "$WIN_JOB" --format=JobIDRaw,State,ExitCode \
    | awk -F'|' -v id="$WIN_JOB" '$1 == id {print $2 "|" $3; exit}')
  [[ "$STATE" == COMPLETED*'|0:0' ]] || {
    echo "winner $WIN_JOB for $MODEL_KEY did not complete cleanly: $STATE" >&2
    exit 5
  }

  if [[ "$WINNER" == ai ]]; then
    BASE="$ROOT/results/stage_o/partD_${MODEL_KEY}"
    MIRROR="$BASE/mirror_h100"
    for FILE in rows.jsonl behavior.summary.json taxonomy_rows.jsonl \
      taxonomy_behavior.summary.json; do
      [[ -f "$MIRROR/$FILE" ]] || { echo "missing $MIRROR/$FILE" >&2; exit 6; }
      cp -p "$MIRROR/$FILE" "$BASE/$FILE"
    done
    for FILE in unused_pool.jsonl taxonomy_unused_pool.jsonl; do
      if [[ -f "$MIRROR/$FILE" ]]; then
        cp -p "$MIRROR/$FILE" "$BASE/$FILE"
      fi
    done
  fi
done

cd "$ROOT"
"$PY" src/qwen_small_circuit.py summarize-part-d \
  --root results/stage_o --out-dir results/stage_o/partD_summary
