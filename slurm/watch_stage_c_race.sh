#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <ai_jobid> <gpu_jobid>" >&2
  exit 2
fi

AI_JOB="$1"
GPU_JOB="$2"
POLL_SECONDS="${POLL_SECONDS:-20}"
MAX_SECONDS="${MAX_SECONDS:-21600}"

state_of() {
  squeue -h -j "$1" -o "%T" 2>/dev/null | head -n 1
}

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

log "watching ai=$AI_JOB gpu=$GPU_JOB poll=${POLL_SECONDS}s max=${MAX_SECONDS}s"
start=$(date +%s)

while true; do
  now=$(date +%s)
  if (( now - start > MAX_SECONDS )); then
    log "timeout; leaving both jobs untouched"
    exit 0
  fi

  ai_state="$(state_of "$AI_JOB" || true)"
  gpu_state="$(state_of "$GPU_JOB" || true)"
  log "ai=${ai_state:-NOT_IN_QUEUE} gpu=${gpu_state:-NOT_IN_QUEUE}"

  if [[ "$ai_state" == "RUNNING" && "$gpu_state" == "RUNNING" ]]; then
    log "both running; keeping ai=$AI_JOB and canceling gpu=$GPU_JOB"
    scancel "$GPU_JOB" || true
    exit 0
  fi

  if [[ "$ai_state" == "RUNNING" && "$gpu_state" =~ ^(PENDING|CONFIGURING)$ ]]; then
    log "ai started first; canceling gpu=$GPU_JOB"
    scancel "$GPU_JOB" || true
    exit 0
  fi

  if [[ "$gpu_state" == "RUNNING" && "$ai_state" =~ ^(PENDING|CONFIGURING)$ ]]; then
    log "gpu started first; canceling ai=$AI_JOB"
    scancel "$AI_JOB" || true
    exit 0
  fi

  if [[ -z "$ai_state" && -z "$gpu_state" ]]; then
    log "both jobs left queue before one was observed running"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
