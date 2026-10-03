#!/usr/bin/env bash
# Stage N Phase 2 behavior over the preregistered 25 checkpoints.
# Evidence: step0 download+216 rows took 96s; Phase 1's 7776-row CPU pass took
# ~9m, while four concurrent online revisions measured 7.5-8m for 3888 rows.
# Request 10m for any relaunch (25% buffer over the observed concurrent path).
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --array=0-24%4
#SBATCH -J n2_behavior
#SBATCH -o results/stage_n/dynamics/logs/behavior_%A_%a.out
#SBATCH -e results/stage_n/dynamics/logs/behavior_%A_%a.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
CHECKPOINTS=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 6000 8000 12000 16000 24000 32000 48000 64000 96000 143000)
STEP="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
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
mkdir -p results/stage_n/dynamics/logs

SECONDS=0
echo "host=$(hostname) checkpoint=step${STEP} start=$(date --iso-8601=seconds)"
"$PY" src/pythia_dynamics.py behavior \
  --checkpoint "step${STEP}" \
  --out-root results/stage_n/dynamics \
  --batch-size 128 \
  --online
echo "elapsed_seconds=$SECONDS checkpoint=step${STEP} end=$(date --iso-8601=seconds)"
echo DONE
