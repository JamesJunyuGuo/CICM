#!/bin/bash
#SBATCH --account=cis250190-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=p_trace_capture_a
#SBATCH --output=results/stage_p/logs/p_trace_capture_a_%j.out
#SBATCH --error=results/stage_p/logs/p_trace_capture_a_%j.err

set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${MODEL_LABEL:?set MODEL_LABEL}"
: "${OUT_DIR:?set OUT_DIR}"
: "${MODEL_DTYPE:=bfloat16}"
: "${LIMIT_ITEMS:=0}"
: "${DATA_PATH:=results/stage_p/data/event_items_broad.jsonl}"
module purge
module load modtree/gpu
module load conda
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
cd "$PROJ"
"$PY" -u src/genattn_capture.py capture \
  --data "$DATA_PATH" \
  --model "$MODEL_PATH" --model-label "$MODEL_LABEL" --dtype "$MODEL_DTYPE" \
  --limit-items "$LIMIT_ITEMS" --out-dir "$OUT_DIR"
