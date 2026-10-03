#!/bin/bash
set -euo pipefail

: "${RACE_LABEL:?set RACE_LABEL}"
: "${GPU_JOB:?set GPU_JOB}"
: "${AI_JOB:?set AI_JOB}"
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
RACE_DIR="${RACE_DIR:-$ROOT/results/stage_o/portal_race}"
mkdir -p "$RACE_DIR"

job_record() {
  sacct -X -n -P -j "$1" --format=JobIDRaw,State,Start,ExitCode \
    | awk -F'|' -v id="$1" '$1 == id {print $2 "|" $3 "|" $4; exit}'
}

is_success() {
  [[ "$1" == COMPLETED* && "$2" == 0:0 ]]
}

is_running_healthy() {
  local state=$1 start=$2 start_epoch now_epoch
  [[ "$state" == RUNNING* && -n "$start" && "$start" != Unknown ]] || return 1
  start_epoch=$(date -d "$start" +%s) || return 1
  now_epoch=$(date +%s)
  (( now_epoch - start_epoch >= HEALTHY_SECONDS ))
}

is_candidate() {
  is_success "$1" "$3" || is_running_healthy "$1" "$2"
}

cancel_job() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "dry_run_scancel=$1"
  else
    scancel "$1" 2>/dev/null || true
  fi
}

POLL_SECONDS=${POLL_SECONDS:-15}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1800}
HEALTHY_SECONDS=${HEALTHY_SECONDS:-180}
MAX_POLLS=$((MAX_WAIT_SECONDS / POLL_SECONDS))
if (( MAX_POLLS < 1 )); then
  MAX_POLLS=1
fi

GPU_REC= AI_REC= GPU_STATE= GPU_START= GPU_EXIT= AI_STATE= AI_START= AI_EXIT=
for _ in $(seq 1 "$MAX_POLLS"); do
  GPU_REC=$(job_record "$GPU_JOB")
  AI_REC=$(job_record "$AI_JOB")
  IFS='|' read -r GPU_STATE GPU_START GPU_EXIT <<< "$GPU_REC"
  IFS='|' read -r AI_STATE AI_START AI_EXIT <<< "$AI_REC"

  if is_candidate "$GPU_STATE" "$GPU_START" "$GPU_EXIT" || \
     is_candidate "$AI_STATE" "$AI_START" "$AI_EXIT"; then
    break
  fi
  sleep "$POLL_SECONDS"
done

if is_candidate "$GPU_STATE" "$GPU_START" "$GPU_EXIT" && \
   is_candidate "$AI_STATE" "$AI_START" "$AI_EXIT"; then
  if [[ "$AI_START" < "$GPU_START" ]]; then
    WINNER=ai; WIN_JOB=$AI_JOB; LOSER_JOB=$GPU_JOB
  else
    WINNER=gpu; WIN_JOB=$GPU_JOB; LOSER_JOB=$AI_JOB
  fi
elif is_candidate "$AI_STATE" "$AI_START" "$AI_EXIT"; then
  WINNER=ai; WIN_JOB=$AI_JOB; LOSER_JOB=$GPU_JOB
elif is_candidate "$GPU_STATE" "$GPU_START" "$GPU_EXIT"; then
  WINNER=gpu; WIN_JOB=$GPU_JOB; LOSER_JOB=$AI_JOB
else
  echo "Neither job completed successfully or stayed RUNNING for ${HEALTHY_SECONDS}s within ${MAX_WAIT_SECONDS}s" >&2
  echo "gpu=$GPU_REC ai=$AI_REC" >&2
  if [[ "${REQUEUE_ON_TIMEOUT:-1}" == 1 && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "requeueing watcher job ${SLURM_JOB_ID}" >&2
    scontrol requeue "$SLURM_JOB_ID"
    while true; do sleep 60; done
  fi
  exit 3
fi

cancel_job "$LOSER_JOB"
if [[ -n "${GPU_CHILD:-}" && -n "${AI_CHILD:-}" ]]; then
  if [[ "$WINNER" == ai ]]; then
    cancel_job "$GPU_CHILD"
  else
    cancel_job "$AI_CHILD"
  fi
fi

TMP="$RACE_DIR/${RACE_LABEL}.winner.tmp.$$"
{
  echo "label=$RACE_LABEL"
  echo "winner=$WINNER"
  echo "winner_job=$WIN_JOB"
  echo "loser_job=$LOSER_JOB"
  echo "gpu_job=$GPU_JOB"
  echo "gpu_state=$GPU_STATE"
  echo "gpu_start=$GPU_START"
  echo "ai_job=$AI_JOB"
  echo "ai_state=$AI_STATE"
  echo "ai_start=$AI_START"
  echo "selection_policy=completed_success_or_running_${HEALTHY_SECONDS}s"
  echo "decided_at=$(date --iso-8601=seconds)"
} > "$TMP"
mv "$TMP" "$RACE_DIR/${RACE_LABEL}.winner"
cat "$RACE_DIR/${RACE_LABEL}.winner"
