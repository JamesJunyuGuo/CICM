#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=o_promote
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/promote_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/promote_%x_%j.err

set -euo pipefail
: "${RACE_LABEL:?set RACE_LABEL}"
: "${AI_DIR:?set AI_DIR}"
: "${GPU_DIR:?set GPU_DIR}"
: "${DEST_DIR:?set DEST_DIR}"
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
MARKER="$ROOT/results/stage_o/portal_race/${RACE_LABEL}.winner"
if [[ -s "$MARKER" ]]; then
  WINNER=$(awk -F= '$1 == "winner" {print $2}' "$MARKER")
  WIN_JOB=$(awk -F= '$1 == "winner_job" {print $2}' "$MARKER")
else
  : "${AI_JOB:?set AI_JOB when the race marker may be absent}"
  : "${GPU_JOB:?set GPU_JOB when the race marker may be absent}"
  AI_STATE=$(sacct -X -n -P -j "$AI_JOB" --format=JobIDRaw,State,ExitCode \
    | awk -F'|' -v id="$AI_JOB" '$1 == id {print $2 "|" $3; exit}')
  GPU_STATE=$(sacct -X -n -P -j "$GPU_JOB" --format=JobIDRaw,State,ExitCode \
    | awk -F'|' -v id="$GPU_JOB" '$1 == id {print $2 "|" $3; exit}')
  if [[ "$AI_STATE" == "COMPLETED|0:0" ]]; then
    WINNER=ai
    WIN_JOB=$AI_JOB
    LOSER_JOB=$GPU_JOB
  elif [[ "$GPU_STATE" == "COMPLETED|0:0" ]]; then
    WINNER=gpu
    WIN_JOB=$GPU_JOB
    LOSER_JOB=$AI_JOB
  else
    echo "neither mirror completed successfully: ai=$AI_STATE gpu=$GPU_STATE" >&2
    exit 4
  fi
  TMP="$MARKER.tmp.$$"
  {
    echo "label=$RACE_LABEL"
    echo "winner=$WINNER"
    echo "winner_job=$WIN_JOB"
    echo "loser_job=$LOSER_JOB"
    echo "ai_job=$AI_JOB"
    echo "gpu_job=$GPU_JOB"
    echo "selection_policy=promotion_fallback_after_completion"
    echo "decided_at=$(date --iso-8601=seconds)"
  } > "$TMP"
  mv "$TMP" "$MARKER"
fi
STATE=$(sacct -X -n -P -j "$WIN_JOB" --format=JobIDRaw,State,ExitCode \
  | awk -F'|' -v id="$WIN_JOB" '$1 == id {print $2 "|" $3; exit}')
[[ "$STATE" == "COMPLETED|0:0" ]] || { echo "winner did not complete: $STATE" >&2; exit 5; }
if [[ "$WINNER" == ai ]]; then SRC="$ROOT/$AI_DIR"; else SRC="$ROOT/$GPU_DIR"; fi
DEST="$ROOT/$DEST_DIR"
[[ -d "$SRC" ]] || { echo "missing winner output $SRC" >&2; exit 6; }
mkdir -p "$DEST"
cp -a "$SRC"/. "$DEST"/
printf '{"race_label":"%s","winner":"%s","winner_job":"%s","source":"%s"}\n' \
  "$RACE_LABEL" "$WINNER" "$WIN_JOB" "$SRC" > "$DEST/mirror_promotion.json"
