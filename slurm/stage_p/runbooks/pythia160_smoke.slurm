#!/bin/bash
#SBATCH --account=cis250190-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --job-name=p_trace_smoke4_h
#SBATCH --output=results/stage_p/logs/p_trace_smoke4_h_%j.out
#SBATCH --error=results/stage_p/logs/p_trace_smoke4_h_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=/anvil/scratch/x-jguo7/hf_cache/hub/models--EleutherAI--pythia-160m/snapshots/b56d9bee36300031aeea723b73c4d62ac7fa71a2
export PYTHONPATH="$PROJ/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export TOKENIZERS_PARALLELISM=false

cd "$PROJ"
"$PY" -u src/genattn_capture.py capture \
  --data results/stage_p/data/smoke4_item.jsonl \
  --model "$MODEL" \
  --model-label pythia-160m \
  --dtype float32 \
  --out-dir results/stage_p/smoke4_h100/pythia160
