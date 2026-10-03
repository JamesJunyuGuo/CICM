#!/bin/bash
set -euo pipefail

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$ROOT"

ABLATE_AI=20036341
ABLATE_GPU=20036342
ABLATE_PROMOTE=20036345

INDUCTION_AI=$(sbatch --time=00:08:00 \
  --export=ALL,QKOV_OUT_DIR=results/stage_o/partC_qwen15b/qkov,INDUCTION_SUMMARY=results/stage_o/partC_qwen15b/induction_repair/mirror_h100_r2/induction.summary.json \
  --job-name=o_cind_h2 results/stage_o/run_partC_qkov_induction_h100.slurm | awk '{print $4}')
INDUCTION_GPU=$(sbatch -A cis250190-gpu -p gpu --time=00:10:00 \
  --export=ALL,QKOV_OUT_DIR=results/stage_o/partC_qwen15b/qkov,INDUCTION_SUMMARY=results/stage_o/partC_qwen15b/induction_repair/mirror_a100_r2/induction.summary.json \
  --job-name=o_cind_a2 results/stage_o/run_partC_qkov_induction_h100.slurm | awk '{print $4}')

INDUCTION_PROMOTE=$(sbatch --dependency="afterany:$INDUCTION_AI:$INDUCTION_GPU" \
  --export=ALL,RACE_LABEL=partC_induction_repair,AI_JOB=$INDUCTION_AI,GPU_JOB=$INDUCTION_GPU,AI_DIR=results/stage_o/partC_qwen15b/induction_repair/mirror_h100_r2,GPU_DIR=results/stage_o/partC_qwen15b/induction_repair/mirror_a100_r2,DEST_DIR=results/stage_o/partC_qwen15b \
  --job-name=o_ci_prom2 results/stage_o/run_mirror_promote_shared.slurm | awk '{print $4}')

REPORT_JOB=$(sbatch --dependency="afterok:$ABLATE_PROMOTE:$INDUCTION_PROMOTE" \
  results/stage_o/run_partC_report_shared.slurm | awk '{print $4}')

RACES="partC_ablate_repair:$ABLATE_GPU:$ABLATE_AI;partC_induction_repair:$INDUCTION_GPU:$INDUCTION_AI"
RACE_JOB=$(sbatch \
  --export="ALL,RACES=$RACES,HEALTHY_SECONDS=180,MAX_WAIT_SECONDS=1500,POLL_SECONDS=15" \
  --job-name=o_c_repair_r2 results/stage_o/run_portal_multi_race_arbiter_shared.slurm \
  | awk '{print $4}')

jq -n \
  --arg ablate_ai "$ABLATE_AI" --arg ablate_gpu "$ABLATE_GPU" \
  --arg ablate_promote "$ABLATE_PROMOTE" \
  --arg induction_ai "$INDUCTION_AI" --arg induction_gpu "$INDUCTION_GPU" \
  --arg induction_promote "$INDUCTION_PROMOTE" \
  --arg report "$REPORT_JOB" --arg race "$RACE_JOB" \
  '{stage:"O-Part-C-induction-repair-r2", ablate_ai:$ablate_ai,
    ablate_gpu:$ablate_gpu, ablate_promote:$ablate_promote,
    induction_ai:$induction_ai, induction_gpu:$induction_gpu,
    induction_promote:$induction_promote, report:$report, race_watcher:$race}' \
  > results/stage_o/partC_qwen15b/induction_repair_r2_manifest.json
cat results/stage_o/partC_qwen15b/induction_repair_r2_manifest.json
