#!/usr/bin/env bash
# Stage K K1 full A100 duplicate: isolated output paths for Qwen2.5-7B.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=08:00:00
#SBATCH -J k_k1_qwen7b_a100
#SBATCH -o logs/k_k1_qwen7b_a100_%j.out
#SBATCH -e logs/k_k1_qwen7b_a100_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_k/qwen7b_a100

module purge
module load modtree/gpu
module load conda

export VERL_ENV="/anvil/projects/x-cis250190/software/envs/verl"
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

export HF_HOME="/anvil/scratch/x-jguo7/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" src/icf_generate.py \
  --scenario dynamic_preference \
  --input external_data/icf_bench/dynamic_preference/answer_gpt4o1120-Eval.json \
  --out results/stage_k/qwen7b_a100/dynamic_preference_qwen7b.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 8 \
  --max-new-tokens 128 \
  --dtype bfloat16

"$PY" src/icf_generate.py \
  --scenario instructional_forgetting \
  --input external_data/icf_bench/instructional_forgetting/instructional_forgetting.json \
  --out results/stage_k/qwen7b_a100/instructional_forgetting_qwen7b.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 8 \
  --max-new-tokens 192 \
  --dtype bfloat16

"$PY" src/icf_k1.py \
  --dynamic-preference results/stage_k/qwen7b_a100/dynamic_preference_qwen7b.json \
  --instructional-forgetting results/stage_k/qwen7b_a100/instructional_forgetting_qwen7b.json \
  --model Qwen/Qwen2.5-7B-Instruct-a100 \
  --source qwen_self_generated_full_a100 \
  --out-rows results/stage_k/qwen7b_a100/k1_rows_qwen7b.jsonl \
  --summary-out results/stage_k/qwen7b_a100/k1_summary_qwen7b.json \
  --report-out results/stage_k/qwen7b_a100/REPORT.md

echo "DONE"
