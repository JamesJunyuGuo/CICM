#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=o_a_cache14
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/cache14_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/cache14_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
mkdir -p "$ROOT/results/stage_o/logs" "$ROOT/results/stage_o/dependencies"
cd "$ROOT"

SECONDS=0
"$PY" src/pythia_crossscale.py cache-model \
  --model EleutherAI/pythia-1.4b \
  --cache-dir /anvil/projects/x-cis250190/my_hf_cache \
  --summary results/stage_o/dependencies/pythia_1.4b_cache.json
echo "elapsed_seconds=$SECONDS"
