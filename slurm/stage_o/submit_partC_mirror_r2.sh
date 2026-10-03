#!/bin/bash
set -euo pipefail

ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$ROOT"

PAIR_AI_JOB=$(sbatch \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/pairs/mirror_h100_r2 \
  --job-name=o_cpair_h2 results/stage_o/run_partC_pairs_h100.slurm \
  | awk '{print $4}')
PAIR_GPU_JOB=$(sbatch -A cis250190-gpu -p gpu --time=00:10:00 \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/pairs/mirror_a100_r2 \
  --job-name=o_cpair_a2 results/stage_o/run_partC_pairs_h100.slurm \
  | awk '{print $4}')
PAIR_PROMOTE_JOB=$(sbatch --dependency="afterany:$PAIR_AI_JOB:$PAIR_GPU_JOB" \
  --export=ALL,RACE_LABEL=partC_pairs,AI_JOB=$PAIR_AI_JOB,GPU_JOB=$PAIR_GPU_JOB,AI_DIR=results/stage_o/partC_qwen15b/pairs/mirror_h100_r2,GPU_DIR=results/stage_o/partC_qwen15b/pairs/mirror_a100_r2,DEST_DIR=results/stage_o/partC_qwen15b/pairs \
  --job-name=o_cp_prom2 results/stage_o/run_mirror_promote_shared.slurm \
  | awk '{print $4}')

SMOKE_AI_JOB=$(sbatch --dependency="afterok:$PAIR_PROMOTE_JOB" \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/patch_smoke/mirror_h100_r2 \
  --job-name=o_csmk_h2 results/stage_o/run_partC_patch_smoke_h100.slurm \
  | awk '{print $4}')
SMOKE_GPU_JOB=$(sbatch -A cis250190-gpu -p gpu --time=00:10:00 \
  --dependency="afterok:$PAIR_PROMOTE_JOB" \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/patch_smoke/mirror_a100_r2 \
  --job-name=o_csmk_a2 results/stage_o/run_partC_patch_smoke_h100.slurm \
  | awk '{print $4}')
HARVEST_AI_JOB=$(sbatch --dependency="afterok:$PAIR_PROMOTE_JOB" \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/mechanism/mirror_h100_r2 \
  --job-name=o_charv_h2 results/stage_o/run_partC_harvest_h100.slurm \
  | awk '{print $4}')
HARVEST_GPU_JOB=$(sbatch -A cis250190-gpu -p gpu --time=00:10:00 \
  --dependency="afterok:$PAIR_PROMOTE_JOB" \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/mechanism/mirror_a100_r2 \
  --job-name=o_charv_a2 results/stage_o/run_partC_harvest_h100.slurm \
  | awk '{print $4}')

SMOKE_PROMOTE_JOB=$(sbatch --dependency="afterany:$SMOKE_AI_JOB:$SMOKE_GPU_JOB" \
  --export=ALL,RACE_LABEL=partC_patch_smoke,AI_JOB=$SMOKE_AI_JOB,GPU_JOB=$SMOKE_GPU_JOB,AI_DIR=results/stage_o/partC_qwen15b/patch_smoke/mirror_h100_r2,GPU_DIR=results/stage_o/partC_qwen15b/patch_smoke/mirror_a100_r2,DEST_DIR=results/stage_o/partC_qwen15b/patch_smoke \
  --job-name=o_cs_prom2 results/stage_o/run_mirror_promote_shared.slurm \
  | awk '{print $4}')
HARVEST_PROMOTE_JOB=$(sbatch --dependency="afterany:$HARVEST_AI_JOB:$HARVEST_GPU_JOB" \
  --export=ALL,RACE_LABEL=partC_harvest,AI_JOB=$HARVEST_AI_JOB,GPU_JOB=$HARVEST_GPU_JOB,AI_DIR=results/stage_o/partC_qwen15b/mechanism/mirror_h100_r2,GPU_DIR=results/stage_o/partC_qwen15b/mechanism/mirror_a100_r2,DEST_DIR=results/stage_o/partC_qwen15b/mechanism \
  --job-name=o_ch_prom2 results/stage_o/run_mirror_promote_shared.slurm \
  | awk '{print $4}')

ANALYZE_JOB=$(sbatch --dependency="afterok:$HARVEST_PROMOTE_JOB" \
  results/stage_o/run_partC_analyze_shared.slurm | awk '{print $4}')
GATE_JOB=$(sbatch --dependency="afterok:$SMOKE_PROMOTE_JOB:$ANALYZE_JOB" \
  results/stage_o/run_partC_gate_submit_shared.slurm | awk '{print $4}')

RACES="partC_pairs:$PAIR_GPU_JOB:$PAIR_AI_JOB;partC_patch_smoke:$SMOKE_GPU_JOB:$SMOKE_AI_JOB;partC_harvest:$HARVEST_GPU_JOB:$HARVEST_AI_JOB"
RACE_JOB=$(sbatch \
  --export="ALL,RACES=$RACES,HEALTHY_SECONDS=180,MAX_WAIT_SECONDS=1500,POLL_SECONDS=15" \
  --job-name=o_c_races2 results/stage_o/run_portal_multi_race_arbiter_shared.slurm \
  | awk '{print $4}')

jq -n \
  --arg pair_ai "$PAIR_AI_JOB" --arg pair_gpu "$PAIR_GPU_JOB" \
  --arg pair_promote "$PAIR_PROMOTE_JOB" \
  --arg smoke_ai "$SMOKE_AI_JOB" --arg smoke_gpu "$SMOKE_GPU_JOB" \
  --arg harvest_ai "$HARVEST_AI_JOB" --arg harvest_gpu "$HARVEST_GPU_JOB" \
  --arg smoke_promote "$SMOKE_PROMOTE_JOB" \
  --arg harvest_promote "$HARVEST_PROMOTE_JOB" \
  --arg analyze "$ANALYZE_JOB" --arg gate "$GATE_JOB" --arg race "$RACE_JOB" \
  '{stage:"O-Part-C-mirror-r2", pair_ai:$pair_ai, pair_gpu:$pair_gpu,
    pair_promote:$pair_promote, smoke_ai:$smoke_ai, smoke_gpu:$smoke_gpu,
    harvest_ai:$harvest_ai, harvest_gpu:$harvest_gpu,
    smoke_promote:$smoke_promote, harvest_promote:$harvest_promote,
    analyze:$analyze, gate:$gate, race_watcher:$race,
    race_health_seconds:180}' \
  > results/stage_o/partC_qwen15b/submission_manifest_r2.json
cat results/stage_o/partC_qwen15b/submission_manifest_r2.json
