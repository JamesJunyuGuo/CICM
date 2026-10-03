#!/bin/bash
#SBATCH --account=cis250190
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --job-name=p_q15_smoke_c
#SBATCH --output=results/stage_p/logs/p_q15_smoke_c_%j.out
#SBATCH --error=results/stage_p/logs/p_q15_smoke_c_%j.err

set -euo pipefail
module purge
module load conda
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=/anvil/scratch/x-jguo7/hf_cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
export PYTHONPATH="$PROJ/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "$PROJ"
"$PY" -u src/genattn_capture.py capture \
  --data results/stage_p/data/event_items_broad.jsonl \
  --model "$MODEL" --model-label qwen2.5-1.5b --dtype float32 \
  --limit-items 1 --out-dir results/stage_p/smoke_qwen15/cpu
