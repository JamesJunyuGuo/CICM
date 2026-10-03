#!/bin/bash
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --job-name=o_c15_qkov
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_qkov_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_qkov_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/hf_cache
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
QKOV_OUT_DIR=${QKOV_OUT_DIR:-results/stage_o/partC_qwen15b/qkov}
INDUCTION_SUMMARY=${INDUCTION_SUMMARY:-results/stage_o/partC_qwen15b/induction.summary.json}
mkdir -p "$(dirname "$INDUCTION_SUMMARY")"
if [[ ! -s "$QKOV_OUT_DIR/qkov.summary.json" || \
      ! -s "$QKOV_OUT_DIR/qkov_rows.jsonl" ]]; then
  "$PY" src/qwen_small_circuit.py qkov-part-c \
    --model Qwen/Qwen2.5-1.5B --dtype bfloat16 --limit 96 --max-key-heads 12 \
    --pool results/stage_o/partC_qwen15b/matched_pool.jsonl \
    --cumulative-summary results/stage_o/partC_qwen15b/circuit/cumulative_patch.summary.json \
    --out-dir "$QKOV_OUT_DIR"
fi
"$PY" src/qwen_small_circuit.py induction-part-c \
  --model Qwen/Qwen2.5-1.5B --dtype bfloat16 --n-sequences 64 --sequence-length 24 \
  --max-key-heads 12 \
  --cumulative-summary results/stage_o/partC_qwen15b/circuit/cumulative_patch.summary.json \
  --summary "$INDUCTION_SUMMARY"
