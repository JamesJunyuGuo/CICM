#!/bin/bash
#SBATCH --account=cis250190-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:06:00
#SBATCH --job-name=p_trace_patch
#SBATCH --output=results/stage_p/logs/p_trace_patch_%j.out
#SBATCH --error=results/stage_p/logs/p_trace_patch_%j.err

set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${MODEL_LABEL:?set MODEL_LABEL}"
: "${ANALYSIS_DIR:?set ANALYSIS_DIR}"
: "${PATCH_DIR:?set PATCH_DIR}"
: "${MODEL_DTYPE:=bfloat16}"
module purge
module load modtree/gpu
module load conda
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$PROJ/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
cd "$PROJ"
"$PY" -u src/genattn_patch.py run \
  --model "$MODEL_PATH" --model-label "$MODEL_LABEL" --dtype "$MODEL_DTYPE" \
  --analysis-summary "$ANALYSIS_DIR/analysis.summary.json" \
  --pairs "$ANALYSIS_DIR/patch_pairs.jsonl" \
  --out-dir "$PATCH_DIR" --n-boot 2000
