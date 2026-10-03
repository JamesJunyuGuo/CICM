#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=o_c15_gate
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_gate_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partC15_gate_%j.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
OUT="$ROOT/results/stage_o/partC_qwen15b"
cd "$ROOT"

IDENTITY=$(jq -r '.identity_gate.exact_zero' "$OUT/patch_smoke/path_patch_first_pass.summary.json")
PAIR_GATE=$(jq -r '.pair_gate' "$OUT/pairs/exact_pairs.summary.json")
PROBE_GATE=$(jq -r '.probe.above_shuffle95' "$OUT/mechanism/mechanism.summary.json")
[[ "$IDENTITY" == true ]] || { echo "identity patch gate failed" >&2; exit 2; }
[[ "$PAIR_GATE" == true ]] || { echo "exact-pair gate failed" >&2; exit 2; }

SMOKE_MARKER="$ROOT/results/stage_o/portal_race/partC_patch_smoke.winner"
[[ -s "$SMOKE_MARKER" ]] || { echo "missing smoke race marker" >&2; exit 3; }
SMOKE_JOB=$(awk -F= '$1 == "winner_job" {print $2}' "$SMOKE_MARKER")
SMOKE_SECONDS=$(sacct -X -n -P -j "$SMOKE_JOB" --format=JobIDRaw,ElapsedRaw \
  | awk -F'|' -v id="$SMOKE_JOB" '$1 == id {print $2; exit}')
[[ "$SMOKE_SECONDS" =~ ^[0-9]+$ ]] || { echo "missing smoke runtime" >&2; exit 3; }

# Full has about 9.2x the batched forward passes of the four-pair smoke.
# Use 10x plus a 25% buffer, bounded to 30-90 minutes.
FULL_SECONDS=$((SMOKE_SECONDS * 10 * 5 / 4))
(( FULL_SECONDS < 1800 )) && FULL_SECONDS=1800
(( FULL_SECONDS > 5400 )) && FULL_SECONDS=5400
printf -v FULL_TIME '%02d:%02d:%02d' \
  $((FULL_SECONDS / 3600)) $(((FULL_SECONDS % 3600) / 60)) $((FULL_SECONDS % 60))

PATCH_AI_JOB=$(sbatch --time="$FULL_TIME" \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/circuit/mirror_h100 \
  --job-name=o_cpat_h results/stage_o/run_partC_patch_full_h100.slurm \
  | awk '{print $4}')
PATCH_GPU_JOB=$(sbatch -A cis250190-gpu -p gpu --time="$FULL_TIME" \
  --export=ALL,OUT_DIR=results/stage_o/partC_qwen15b/circuit/mirror_a100 \
  --job-name=o_cpat_a results/stage_o/run_partC_patch_full_h100.slurm \
  | awk '{print $4}')
PATCH_RACE_JOB=$(sbatch \
  "--export=ALL,RACE_LABEL=partC_patch_full,GPU_JOB=$PATCH_GPU_JOB,AI_JOB=$PATCH_AI_JOB,MAX_WAIT_SECONDS=1500,POLL_SECONDS=15,REQUEUE_ON_TIMEOUT=1" \
  --job-name=o_rc_full results/stage_o/run_portal_race_arbiter_shared.slurm \
  | awk '{print $4}')
PATCH_PROMOTE_JOB=$(sbatch --dependency="afterany:$PATCH_AI_JOB:$PATCH_GPU_JOB" \
  "--export=ALL,RACE_LABEL=partC_patch_full,AI_JOB=$PATCH_AI_JOB,GPU_JOB=$PATCH_GPU_JOB,AI_DIR=results/stage_o/partC_qwen15b/circuit/mirror_h100,GPU_DIR=results/stage_o/partC_qwen15b/circuit/mirror_a100,DEST_DIR=results/stage_o/partC_qwen15b/circuit" \
  --job-name=o_cf_prom results/stage_o/run_mirror_promote_shared.slurm \
  | awk '{print $4}')
ABLATE_JOB=$(sbatch --dependency="afterok:$PATCH_PROMOTE_JOB" results/stage_o/run_partC_ablate_h100.slurm \
  | awk '{print $4}')
QKOV_JOB=$(sbatch --dependency="afterok:$PATCH_PROMOTE_JOB" results/stage_o/run_partC_qkov_induction_h100.slurm \
  | awk '{print $4}')
REPORT_JOB=$(sbatch --dependency="afterok:$ABLATE_JOB:$QKOV_JOB" results/stage_o/run_partC_report_shared.slurm \
  | awk '{print $4}')

TMP="$OUT/submission_manifest.json.tmp.$$"
jq -n \
  --arg smoke_job "$SMOKE_JOB" --argjson smoke_seconds "$SMOKE_SECONDS" \
  --arg full_time "$FULL_TIME" --arg patch_ai_job "$PATCH_AI_JOB" \
  --arg patch_gpu_job "$PATCH_GPU_JOB" --arg patch_race_job "$PATCH_RACE_JOB" \
  --arg patch_promote_job "$PATCH_PROMOTE_JOB" \
  --arg ablate_job "$ABLATE_JOB" --arg qkov_job "$QKOV_JOB" \
  --arg report_job "$REPORT_JOB" --argjson probe_gate "$PROBE_GATE" \
  '{stage:"O-Part-C-full-submission", smoke_job:$smoke_job,
    smoke_elapsed_seconds:$smoke_seconds,
    runtime_rule:"10x smoke elapsed plus 25% buffer; bounded to 30-90 minutes",
    full_walltime:$full_time, probe_gate:$probe_gate,
    patch_ai_job:$patch_ai_job, patch_gpu_job:$patch_gpu_job,
    patch_race_job:$patch_race_job, patch_promote_job:$patch_promote_job,
    ablate_job:$ablate_job,
    qkov_induction_job:$qkov_job, report_job:$report_job}' > "$TMP"
mv "$TMP" "$OUT/submission_manifest.json"
cat "$OUT/submission_manifest.json"
