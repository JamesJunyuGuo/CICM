#!/bin/bash
#SBATCH --account=cis250190
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=p_model_smoke_c
#SBATCH --output=results/stage_p/logs/p_model_smoke_c_%j.out
#SBATCH --error=results/stage_p/logs/p_model_smoke_c_%j.err

set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${MODEL_LABEL:?set MODEL_LABEL}"
: "${OUT_DIR:?set OUT_DIR}"
: "${MODEL_DTYPE:=float32}"
module purge
module load conda
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
cd "$PROJ"
"$PY" -u src/genattn_capture.py capture \
  --data results/stage_p/data/event_items_broad.jsonl \
  --model "$MODEL_PATH" --model-label "$MODEL_LABEL" --dtype "$MODEL_DTYPE" \
  --limit-items 1 --out-dir "$OUT_DIR"
