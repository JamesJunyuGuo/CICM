#!/bin/bash
# Duplicate-safe A100 calibration mirror. A 30-minute limit covers the
# size-scaled estimate plus the observed A100/H100 throughput difference.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --job-name=r13_l3_cal_a
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_causal_calibration_a100_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_causal_calibration_a100_%j.err

set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=/anvil/projects/x-cis250190/my_hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/projects/x-cis250190/my_hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/stage_r_llama_causal_heads.py \
  --model "$MODEL" --model-label meta-llama/Llama-3.2-3B-Instruct \
  --model-family llama --stage-label R13-llama3b \
  --out-dir results/stage_r/crossmodel/llama32_3b/r13_causal_calibration_a100 \
  --minimum-discovery-pool 20 \
  --candidate-count 32 --head-sizes 1,2,4,8,16,32 \
  --margins 0.5,1,2,4 --discovery-margin 1 --max-beta 8 \
  --batch-size 8 --capture-batch-size 1 --gradient-batch-size 1 \
  --n-boot 500 --max-new-tokens 8
