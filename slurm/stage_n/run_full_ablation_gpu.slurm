#!/usr/bin/env bash
# Stage N 3g full causal ablation after 3a-3f are complete.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:06:00
#SBATCH -J stage_n_ablate
#SBATCH -o results/stage_n/logs/full_ablation_%j.out
#SBATCH -e results/stage_n/logs/full_ablation_%j.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load modtree/gpu
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
TAG="${PORTAL_TAG:-ai32}"
CORE=results/stage_n/mechanism_full_core_ai32
OUT="results/stage_n/ablation_full_${TAG}"
mkdir -p "$OUT"
SECONDS=0
"$PY" src/pythia_circuit.py ablate \
  --pool results/stage_n/behavior_full_cpu/matched_pool.jsonl \
  --cumulative-summary "$CORE/cumulative_patch.summary.json" \
  --mechanism-summary results/stage_n/mechanism_full_cpu/mechanism.summary.json \
  --summary "$OUT/ablation.summary.json" \
  --max-per-class 96 \
  --n-random 64 \
  --n-boot 2000
echo "elapsed_seconds=$SECONDS"
echo DONE
