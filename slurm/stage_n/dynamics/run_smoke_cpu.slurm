#!/usr/bin/env bash
# Stage N Phase 2 revision-download and behavioral pipeline smoke.
# 216 rows were ~106-148s in Phase 1; 8m leaves room for one 358MB revision download.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:08:00
#SBATCH -J n2_smoke_cpu
#SBATCH -o results/stage_n/dynamics/logs/smoke_cpu_%j.out
#SBATCH -e results/stage_n/dynamics/logs/smoke_cpu_%j.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
cd "$ROOT"
module purge
module load conda

export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
mkdir -p results/stage_n/dynamics/logs results/stage_n/dynamics/smoke_behavior

SECONDS=0
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"$PY" src/pythia_dynamics.py behavior \
  --checkpoint step0 \
  --out-root results/stage_n/dynamics/smoke_behavior \
  --batch-size 128 \
  --limit 216 \
  --online
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
