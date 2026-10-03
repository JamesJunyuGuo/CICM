#!/usr/bin/env bash
# Stage K K1b smoke: fixed left-padding decode, source-joined DP input, matcher validation.
#SBATCH -A cis250190-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -J k_k1b_smoke
#SBATCH -o logs/k_k1b_smoke_%j.out
#SBATCH -e logs/k_k1b_smoke_%j.err

set -euo pipefail

PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
mkdir -p logs results/stage_k/k1b_smoke

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

"$PY" src/icf_prepare.py \
  --old external_data/icf_bench/dynamic_preference/dynamic_preference_old.json \
  --new external_data/icf_bench/dynamic_preference/dynamic_preference_new.json \
  --out results/stage_k/k1b_dynamic_preference_input.json \
  --summary-out results/stage_k/k1b_dynamic_preference_prepare_summary.json

"$PY" src/icf_validate.py \
  --human-dir external_data/icf_bench/human_llm_align \
  --out-rows results/stage_k/k1b_matcher_validation_rows.jsonl \
  --summary-out results/stage_k/k1b_matcher_validation_summary.json

"$PY" src/icf_generate.py \
  --scenario dynamic_preference \
  --input results/stage_k/k1b_dynamic_preference_input.json \
  --out results/stage_k/k1b_smoke/dynamic_preference_qwen05b.json \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 20 \
  --batch-size 4 \
  --max-new-tokens 128 \
  --dtype bfloat16

"$PY" src/icf_generate.py \
  --scenario instructional_forgetting \
  --input external_data/icf_bench/instructional_forgetting/instructional_forgetting.json \
  --out results/stage_k/k1b_smoke/instructional_forgetting_qwen05b.json \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 20 \
  --batch-size 4 \
  --max-new-tokens 192 \
  --dtype bfloat16

"$PY" src/icf_k1.py \
  --dynamic-preference results/stage_k/k1b_smoke/dynamic_preference_qwen05b.json \
  --instructional-forgetting results/stage_k/k1b_smoke/instructional_forgetting_qwen05b.json \
  --model Qwen/Qwen2.5-0.5B-Instruct-smoke20 \
  --source qwen_self_generated_k1b_smoke \
  --out-rows results/stage_k/k1b_smoke/k1_rows_qwen05b.jsonl \
  --summary-out results/stage_k/k1b_smoke/k1_summary_qwen05b.json \
  --report-out results/stage_k/k1b_smoke/REPORT.md

echo "DONE"
