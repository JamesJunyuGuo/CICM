#!/usr/bin/env bash
# Finalize Stage K K2/K3 after one full harvest directory has completed.
# Run on the login node with OPENROUTER_API_KEY set; no GPU required.
set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"

export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

if [[ -f results/stage_k/mech/full_h100/dynamic_preference.npz && -f results/stage_k/mech/full_h100/instructional_forgetting.npz ]]; then
  MECH_DIR=results/stage_k/mech/full_h100
elif [[ -f results/stage_k/mech/full_a100/dynamic_preference.npz && -f results/stage_k/mech/full_a100/instructional_forgetting.npz ]]; then
  MECH_DIR=results/stage_k/mech/full_a100
else
  echo "No complete full_h100/full_a100 harvest found yet." >&2
  exit 2
fi

echo "Using MECH_DIR=$MECH_DIR"

"$PY" src/icf_mech_label.py \
  --responses "$MECH_DIR/dynamic_preference_local_responses.json" \
  --rows-out "$MECH_DIR/dynamic_preference_local_labeled.jsonl" \
  --summary-out "$MECH_DIR/dynamic_preference_local_label_summary.json" \
  --judge-raw "$MECH_DIR/dynamic_preference_local_judge_raw.jsonl" \
  --judge-model openai/gpt-4o \
  --concurrency 6

"$PY" src/icf_mech_analyze.py \
  --mech-dir "$MECH_DIR" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out results/stage_k/mech/k2k3_summary.json \
  --report-out results/stage_k/mech/REPORT.md

echo "DONE"
