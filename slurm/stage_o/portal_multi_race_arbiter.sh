#!/bin/bash
set -euo pipefail

: "${RACES:?set semicolon-separated label:gpu_job:ai_job entries}"
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
RACE_DIR="${RACE_DIR:-$ROOT/results/stage_o/portal_race}"
POLL_SECONDS=${POLL_SECONDS:-15}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1500}
HEALTHY_SECONDS=${HEALTHY_SECONDS:-180}
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

write_winner() {
  local label=$1 winner=$2 winner_job=$3 loser_job=$4
  local gpu_job=$5 gpu_state=$6 gpu_start=$7 ai_job=$8 ai_state=$9 ai_start=${10}
  scancel "$loser_job" 2>/dev/null || true
  local tmp="$RACE_DIR/${label}.winner.tmp.$$"
  {
    echo "label=$label"
    echo "winner=$winner"
    echo "winner_job=$winner_job"
    echo "loser_job=$loser_job"
    echo "gpu_job=$gpu_job"
    echo "gpu_state=$gpu_state"
    echo "gpu_start=$gpu_start"
    echo "ai_job=$ai_job"
    echo "ai_state=$ai_state"
    echo "ai_start=$ai_start"
    echo "selection_policy=completed_success_or_running_${HEALTHY_SECONDS}s"
    echo "decided_at=$(date --iso-8601=seconds)"
  } > "$tmp"
  mv "$tmp" "$RACE_DIR/${label}.winner"
  cat "$RACE_DIR/${label}.winner"
}

IFS=';' read -r -a race_entries <<< "$RACES"
deadline=$((SECONDS + MAX_WAIT_SECONDS))
while (( SECONDS < deadline )); do
  unresolved=0
  for entry in "${race_entries[@]}"; do
    IFS=':' read -r label gpu_job ai_job <<< "$entry"
    [[ -s "$RACE_DIR/${label}.winner" ]] && continue
    unresolved=$((unresolved + 1))
    gpu_record=$(job_record "$gpu_job")
    ai_record=$(job_record "$ai_job")
    IFS='|' read -r gpu_state gpu_start gpu_exit <<< "$gpu_record"
    IFS='|' read -r ai_state ai_start ai_exit <<< "$ai_record"
    if is_candidate "$gpu_state" "$gpu_start" "$gpu_exit" && \
       is_candidate "$ai_state" "$ai_start" "$ai_exit"; then
      if [[ "$ai_start" < "$gpu_start" ]]; then
        write_winner "$label" ai "$ai_job" "$gpu_job" \
          "$gpu_job" "$gpu_state" "$gpu_start" "$ai_job" "$ai_state" "$ai_start"
      else
        write_winner "$label" gpu "$gpu_job" "$ai_job" \
          "$gpu_job" "$gpu_state" "$gpu_start" "$ai_job" "$ai_state" "$ai_start"
      fi
    elif is_candidate "$ai_state" "$ai_start" "$ai_exit"; then
      write_winner "$label" ai "$ai_job" "$gpu_job" \
        "$gpu_job" "$gpu_state" "$gpu_start" "$ai_job" "$ai_state" "$ai_start"
    elif is_candidate "$gpu_state" "$gpu_start" "$gpu_exit"; then
      write_winner "$label" gpu "$gpu_job" "$ai_job" \
        "$gpu_job" "$gpu_state" "$gpu_start" "$ai_job" "$ai_state" "$ai_start"
    fi
  done
  if (( unresolved == 0 )); then
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

echo "No new portal winner before ${MAX_WAIT_SECONDS}s; requeueing watcher ${SLURM_JOB_ID}" >&2
scontrol requeue "$SLURM_JOB_ID"
while true; do sleep 60; done
