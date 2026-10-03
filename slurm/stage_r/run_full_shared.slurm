#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=r_iti_fcpu
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/full_shared_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/full_shared_%j.err

# Runtime basis: latest 24-row/1-random CPU smoke took 2m56s. Scaling the
# dominant row-forward count to 241 held-out rows and 64 random controls gives
# about 4.5--5h; six hours is a 20--25% buffer.
set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$ROOT"
mkdir -p results/stage_r/logs
"$PY" src/stale_binding_correction.py run \
  --out-dir results/stage_r/full_shared \
  --dtype bfloat16 \
  --n-random 64 --n-boot 2000 --n-shuffle 2000 \
  --batch-size 16 --mediation-batch-size 4
