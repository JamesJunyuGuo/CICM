# Raw-row archives (added 2026-10-03)

The small per-example raw records behind the headline results are kept as tarballs. Aggregates (`REPORT.md`, `*.summary.json`, figures,
runbooks) were always tracked; the tarballs add the per-row arms.

| Archive | Contents | Extract |
|---|---|---|
| `results/stage_r/raw_rows_and_logs_2026-09-18.tar.gz` (1.5 MB) | every gitignored Stage R file as of 2026-09-18: baseline/routed/opposite/recap/targeted rows, random-position and random-head control records, calibration validation rows, SLURM logs (411 files, 45 MB) | `tar -xzf results/stage_r/raw_rows_and_logs_2026-09-18.tar.gz` from the repo root |
| `results/stage_q/raw_items_and_responses_2026-09-14.tar.gz` (5.5 MB) | Stage Q frontier-API items, model responses, and caches (9,098 files, 111 MB) | same |

Paths inside the archives are repo-relative, so extraction restores the
original layout. The tarballs are snapshots; new runs should be archived as
new dated tarballs rather than overwriting these.

Larger raw artifacts (per-example attention harvests `*.npz`, Stage P attention dumps, Stage L `l2_rows.jsonl` steering rows) are not included in this repository; every script that produces them is, and the aggregate summaries derived from them are tracked under `results/`.
