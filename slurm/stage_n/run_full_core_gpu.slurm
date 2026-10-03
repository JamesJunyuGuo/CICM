#!/usr/bin/env bash
# Stage N exhaustive causal core (3d-3f); statistics/harvest run separately.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:15:00
#SBATCH -J stage_n_core
#SBATCH -o results/stage_n/logs/full_core_%j.out
#SBATCH -e results/stage_n/logs/full_core_%j.err

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
TAG="${PORTAL_TAG:-ai}"
BASE=results/stage_n/behavior_full_cpu
OUT="results/stage_n/mechanism_full_core_${TAG}"
mkdir -p "$OUT"
SECONDS=0
echo "host=$(hostname) portal=$TAG start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_circuit.py path-patch \
  --pairs "$BASE/pairs.jsonl" \
  --out-dir "$OUT" \
  --batch-size 16 \
  --min-split 16 \
  --n-boot 2000
"$PY" src/pythia_circuit.py qkov \
  --pool "$BASE/matched_pool.jsonl" \
  --cumulative-summary "$OUT/cumulative_patch.summary.json" \
  --out-dir "$OUT" \
  --limit 96
"$PY" src/pythia_circuit.py induction \
  --cumulative-summary "$OUT/cumulative_patch.summary.json" \
  --summary "$OUT/induction.summary.json" \
  --n-sequences 64 \
  --n-boot 2000
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE
