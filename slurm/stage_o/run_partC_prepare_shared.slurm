#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:12:00
#SBATCH --job-name=o_c_prepare
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_prepare_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC_prepare_%j.err

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
cd "$ROOT"
for KEY in qwen05b qwen15b; do
  if [[ "$KEY" == qwen05b ]]; then MODEL=Qwen/Qwen2.5-0.5B; else MODEL=Qwen/Qwen2.5-1.5B; fi
  OUT="results/stage_o/partC_${KEY}"
  mkdir -p "$OUT"
  "$PY" src/qwen_small_circuit.py tokenizer-audit --stage O-Part-C-tokenizer \
    --model "$MODEL" --summary "$OUT/tokenizer_gate.summary.json"
  "$PY" src/qwen_small_circuit.py generate-data --stage O-Part-C-recon-data \
    --model "$MODEL" --audit "$OUT/tokenizer_gate.summary.json" \
    --seeds 11 --n-per-seed-cell 24 \
    --out "$OUT/recon_tasks.jsonl" --summary "$OUT/recon_tasks.summary.json"
done
