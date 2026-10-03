#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=o_bcd_deps
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/bcd_deps_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/bcd_deps_%j.err

set -euo pipefail
module purge
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
mkdir -p "$ROOT/results/stage_o/logs" "$ROOT/results/stage_o/dependencies"
cd "$ROOT"

"$PY" -m py_compile src/pythia_correction.py src/qwen_small_circuit.py
"$PY" -m unittest \
  results.stage_o.test_pythia_correction \
  results.stage_o.test_qwen_small_circuit
"$PY" src/qwen_small_circuit.py self-test

export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
for MODEL in \
  Qwen/Qwen2.5-0.5B \
  Qwen/Qwen2.5-1.5B \
  Qwen/Qwen2.5-3B \
  mistralai/Mistral-7B-v0.3
do
  KEY=$(printf '%s' "$MODEL" | tr '/.' '__')
  "$PY" src/qwen_small_circuit.py cache-model \
    --model "$MODEL" --cache-dir "$HF_HOME/hub" \
    --summary "results/stage_o/dependencies/${KEY}.json"
done

export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$ROOT/results/stage_o/partB_correction/dependencies/datasets_cache"
"$PY" src/pythia_correction.py prepare-lm \
  --out results/stage_o/partB_correction/dependencies/wikitext_segments.npy \
  --summary results/stage_o/partB_correction/dependencies/wikitext_segments.summary.json
