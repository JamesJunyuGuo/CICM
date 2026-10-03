#!/usr/bin/env bash
# Validate clean/corrupted pairs from the already evaluated fixed-k pool.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH -J stage_n_pairs
#SBATCH -o results/stage_n/logs/pairs_%j.out
#SBATCH -e results/stage_n/logs/pairs_%j.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=results/stage_n/behavior_full_cpu
SECONDS=0
"$PY" src/pythia_eval.py build-pairs \
  --rows "$OUT/matched_pool.jsonl" \
  --out "$OUT/pairs.jsonl" \
  --summary "$OUT/pairs.summary.json" \
  --n-pairs 96 \
  --min-pairs 48 \
  --max-candidates 142 \
  --batch-size 64
"$PY" -c "import json; s=json.load(open('$OUT/pairs.summary.json')); assert s['pair_gate'], s"
echo "elapsed_seconds=$SECONDS"
echo DONE
