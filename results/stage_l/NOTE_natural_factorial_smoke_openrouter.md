# Stage L Natural Factorial Smoke - OpenRouter

Date: 2026-07-21

## Purpose

Before scaling Stage L natural data, this smoke removes the confound in the
previous anchored version where same-slot stale competitors were always far and
recent-other-slot competitors were always near.

The new generator crosses two factors:

- same-slot stale distance: `near` vs `far`
- recent-other-slot distance: `near` vs `far`

It also balances target-current distance (`near` vs `far`) inside every cell.
Final queries remain explicitly anchored to the target slot, distractor slots are
retained, and scoring remains program-verifiable.

## Data

- Data: `data/stage_l/cicm_natural_factorial_smoke.jsonl`
- Datasheet: `data/stage_l/cicm_natural_factorial_smoke_DATASHEET.md`
- Templates reused from: `data/stage_l/cicm_natural_smoke_templates.json`
- Rows: 40 = 5 doses x 8 rows

Structural audit:

- Cell counts: each of the four cells has 10 rows.
- Each dose has 2 rows per cell.
- The previously missing cell `(same_slot_stale=near, recent_other_slot=far)` is present.
- Target-current distance is balanced: 20 near, 20 far.
- Within every cell, target-current near/far is 5/5.

Distance spans:

- same-slot near: 2-4 messages before query; same-slot far: 16 messages before query.
- recent-other near: 2 messages before query; recent-other far: 10-14 messages before query.
- target-current near/far are recorded per row; in this smoke they span 2-6 for near and
  12-14 for far.

Machine-readable audit:

- `results/stage_l/natural_factorial_smoke/factorial_smoke_audit.json`

## OpenRouter Smoke

- Model: OpenRouter `qwen/qwen-2.5-7b-instruct`
- Rows: `results/stage_l/natural_factorial_smoke/openrouter_qwen25_7b_rows.jsonl`
- Summary: `results/stage_l/natural_factorial_smoke/openrouter_qwen25_7b_summary.json`

All 40 API calls completed without `call_error`.

Overall:

- 38 `correct_current`
- 1 `within_stale`
- 1 `other`
- 0 `cross_slot`
- accuracy 0.950

By factorial cell:

- `(same=near, other=near)`: 10 correct
- `(same=near, other=far)`: 10 correct
- `(same=far, other=near)`: 9 correct, 1 other
- `(same=far, other=far)`: 9 correct, 1 within-stale

By dose:

- k=1: 7 correct, 1 within-stale
- k=2: 8 correct
- k=3: 7 correct, 1 other
- k=4: 8 correct
- k=6: 8 correct

## Interpretation

The factorial design is structurally correct and ready for inspection. It fixes
the distance confound without suppressing distractor competition.

Behaviorally, this 40-row smoke is too easy to characterize failure-mode effects:
only 2/40 rows fail, so the cell-level failure comparison is underpowered. This
should not be scaled automatically until the design is reviewed. The likely reason
is that the anchored query plus explicit stale-reminder wording makes the current
target slot easy to recover.

## Stop Point

Stop here for human review. Do not launch full L0 or L1/L2 from this artifact
until the four-cell design and the failure-rate target are approved.
