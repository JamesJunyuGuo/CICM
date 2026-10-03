#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --job-name=o_a_harvest
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/harvest_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/harvest_%x_%j.err

set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY=410m,1.4b,or2.8b}"
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "$MODEL_KEY" in
  410m) MODEL=EleutherAI/pythia-410m; export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface; export TRANSFORMERS_CACHE="$HF_HOME/transformers"; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" ;;
  1.4b) MODEL=EleutherAI/pythia-1.4b; export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache; unset TRANSFORMERS_CACHE || true; export HUGGINGFACE_HUB_CACHE="$HF_HOME" ;;
  2.8b) MODEL=EleutherAI/pythia-2.8b; export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache; unset TRANSFORMERS_CACHE || true; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" ;;
  *) echo "unsupported MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
BASE="$ROOT/results/stage_o/$MODEL_KEY"
test "$(jq -r .status "$BASE/behavior/adjudication.summary.json")" = mechanism_ready
test "$(jq -r .pair_gate "$BASE/behavior/pairs.summary.json")" = true
mkdir -p "$BASE/mechanism" "$ROOT/results/stage_o/logs"
cd "$ROOT"
SECONDS=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_circuit.py harvest \
  --pool "$BASE/behavior/matched_pool.jsonl" \
  --out-dir "$BASE/mechanism" --model "$MODEL" --dtype float32
echo "elapsed_seconds=$SECONDS"
