# Stage L Natural Factorial Hard Smoke - OpenRouter

Date: 2026-07-21

## Purpose

This smoke adds difficulty on top of the factorial identity-vs-recency design:

- main rows keep target-current in the `far` regime;
- a small near-current control is retained;
- four cells remain balanced:
  same-slot-stale `near/far` x recent-other-slot `near/far`;
- extra interference is symmetric: longer context and more distractor slot updates.

## Data

- Data: `data/cicm/cicm_natural_factorial_hard_smoke.jsonl`
- Datasheet: `data/cicm/cicm_natural_factorial_hard_smoke_DATASHEET.md`
- Rows: 40 = 5 doses x 8 rows
- Structural/audit JSON:
  `results/cicm/natural_factorial_hard_smoke/hard_smoke_audit.json`

Structural audit:

- Four cells are balanced: 10 rows each.
- Each cell has target-current 8 far / 2 near.
- Overall target-current distance: 32 far / 8 near.
- Query is target-slot anchored; distractor slot updates are retained.

## OpenRouter Smoke

- Model: OpenRouter `qwen/qwen-2.5-7b-instruct`
- Rows: `results/cicm/natural_factorial_hard_smoke/openrouter_qwen25_7b_rows.jsonl`
- Summary: `results/cicm/natural_factorial_hard_smoke/openrouter_qwen25_7b_summary.json`

All 40 API calls completed.

Overall:

- 20 `correct_current`
- 18 `within_stale`
- 2 `other`
- 0 `cross_slot`
- accuracy 0.500
- failure rate 0.500

By cell:

- `(same=near, other=near)`: 7 within-stale, 3 correct
- `(same=near, other=far)`: 7 within-stale, 3 correct
- `(same=far, other=near)`: 1 within-stale, 1 other, 8 correct
- `(same=far, other=far)`: 3 within-stale, 1 other, 6 correct

By target-current distance:

- near-current control rows: all correct
- far-current main rows: contain all observed failures

## Interpretation

The difficulty target is met: failure rate is 50%, and all four factorial cells
have at least one failure.

The two-mode target is NOT met: the failure pool is almost entirely same-slot
stale, with zero cross-slot errors. This is an informative smoke result, not a
reason to tune answers post hoc. It suggests that the current hard profile
strongly creates stale capture once target-current is far, but does not create
recency capture despite recent-other-slot distractors.

## Hard2 Candidate

I generated, but did not API-run, a stronger symmetric-interference candidate:

- Data: `data/cicm/cicm_natural_factorial_hard2_smoke.jsonl`
- Datasheet: `data/cicm/cicm_natural_factorial_hard2_smoke_DATASHEET.md`
- Changes: 6 distractor updates, 6 extra interference turns, same four-cell
  factorial structure, same 32 far / 8 near target-current split.

Its structure is audited in
`results/cicm/natural_factorial_hard_smoke/hard_smoke_audit.json`, but API
inference was not run because the protocol is to stop after the 40-row smoke and
wait for human review.

## Stop Point

Stop here. Do not launch full L0 or L1/L2 until the PI decides whether to:

- accept the hard profile as a stale-dominant factorial substrate;
- run hard2 to test whether stronger symmetric interference recovers cross-slot
  recency capture;
- alter the design to explicitly vary recency salience as a separate factor.
