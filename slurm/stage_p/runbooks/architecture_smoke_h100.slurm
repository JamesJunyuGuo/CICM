#!/bin/bash
#SBATCH --account=cis250190-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --job-name=p_arch_smoke
#SBATCH --output=results/stage_p/logs/p_arch_smoke_%j.out
#SBATCH --error=results/stage_p/logs/p_arch_smoke_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
DATA=results/stage_p/data/event_items_broad.jsonl
: "${OUT_ROOT:=results/stage_p/smoke_arch_h100}"
export PYTHONPATH="$PROJ/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export TOKENIZERS_PARALLELISM=false

cd "$PROJ"

"$PY" -u src/genattn_capture.py capture \
  --data "$DATA" \
  --model /anvil/scratch/x-jguo7/hf_cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323 \
  --model-label qwen2.5-1.5b \
  --dtype bfloat16 \
  --limit-items 1 \
  --out-dir "$OUT_ROOT/qwen15"

"$PY" -u src/genattn_capture.py capture \
  --data "$DATA" \
  --model /anvil/scratch/x-jguo7/hf_cache/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf \
  --model-label gemma-2-2b \
  --dtype bfloat16 \
  --limit-items 1 \
  --out-dir "$OUT_ROOT/gemma2"

"$PY" -u src/genattn_capture.py capture \
  --data "$DATA" \
  --model /anvil/scratch/x-jguo7/cache/huggingface/transformers/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --model-label llama-3.1-8b-instruct \
  --dtype bfloat16 \
  --limit-items 1 \
  --out-dir "$OUT_ROOT/llama8"
