# Stage L L0b Hardness Calibration

Date: 2026-07-21

## Why L0b

Original L-0 on `data/cicm/cicm.jsonl` was too easy for Qwen2.5-7B-Instruct:
all 1050/1050 rows were `correct_current`, so L-1/L-2 had no within-stale pool to
diagnose. L0b keeps the same controlled repeated-binding substrate, but makes the
conversation more realistic after the final binding:

- remove the explicit tracking system prompt;
- ask with a referential final query instead of direct "current value" wording;
- insert unrelated filler turns after the final target update;
- insert updates to other slots after the final target update.

Scoring is still program-verifiable against the controlled vocabulary.

## Candidate Sweep

All candidates used OpenRouter `qwen/qwen-2.5-7b-instruct`, temperature 0,
70 rows each (14 per dose), and saved raw API rows under `results/cicm/`.
The user explicitly approved OpenRouter use for this Stage L behavioral
calibration, and `AGENTS.md` now records the approval rule.

| candidate | system | query | post-final filler | post-final distractor updates | correct | within-stale | accuracy | stale share among failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `l0b_medium` | minimal | referential | 8 | 4 | 44 | 26 | 0.629 | 1.000 |
| `l0b_hard` | minimal | referential | 12 | 6 | 45 | 25 | 0.643 | 1.000 |

`l0b_mild` was interrupted after raw streaming had completed but before clean
summary finalization; its raw JSONL has an interrupted line and is not used for
the decision.

After the API runner was repaired to rewrite ordered JSONL at completion and to
support bounded sequential calibration, a 30-row mild sanity sample completed
cleanly: 23 correct, 5 within-stale, 2 same-slot-other
(`results/cicm/l0b_mild/api_rows_30.jsonl`,
`results/cicm/l0b_mild/api_summary_30.json`). This confirms the mild direction
but does not change the formal `l0b_medium` decision above.

## Dose Readout

`l0b_medium`:

- k=1: 13 correct, 1 within-stale; accuracy 0.929
- k=2: 13 correct, 1 within-stale; accuracy 0.929
- k=3: 6 correct, 8 within-stale; accuracy 0.429
- k=4: 6 correct, 8 within-stale; accuracy 0.429
- k=6: 6 correct, 8 within-stale; accuracy 0.429

`l0b_hard`:

- k=1: 14 correct, 0 within-stale; accuracy 1.000
- k=2: 12 correct, 2 within-stale; accuracy 0.857
- k=3: 5 correct, 9 within-stale; accuracy 0.357
- k=4: 9 correct, 5 within-stale; accuracy 0.643
- k=6: 5 correct, 9 within-stale; accuracy 0.357

## Decision

Use `l0b_medium` as the formal harder Stage L substrate. It is hard enough to
produce a within-stale pool, but it preserves clean low-dose anchors and has a
more regular dose profile than `l0b_hard`. The errors are format-clean:
no `other`, no `same_slot_other`, and no `cross_slot` labels in the completed
medium/hard API runs.

Formal generated dataset:

- `data/cicm/cicm_l0b_medium.jsonl`
- `data/cicm/cicm_l0b_medium_DATASHEET.md`

Hardening parameters:

- `post_final_filler_turns=8`
- `post_final_distractor_updates=4`
- `system_style=minimal`
- `query_style=referential`
- `n_per_dose=210`, doses `1,2,3,4,6`, total n=1050

## Local Model Follow-up

Local Qwen2.5-7B sweep completed on the A100 duplicate:

- A100: `19432779`, completed 2026-07-21 13:30:08, exit code 0,
  output `results/cicm/l0b_sweep_a100/`
- H100 duplicate: `19432630`, canceled while still pending after A100 success

Local A100 readout:

- `mild`: 52 correct, 17 within-stale, 1 same-slot-other; accuracy 0.743,
  stale share among failures 0.944
- `medium`: 39 correct, 29 within-stale, 2 same-slot-other; accuracy 0.557,
  stale share among failures 0.935
- `hard`: 40 correct, 30 within-stale; accuracy 0.571,
  stale share among failures 1.000

This confirms that the L0b hardening transfers to the local Qwen2.5-7B path.
The formal dataset decision remains `l0b_medium`: it gives a large within-stale
pool without making the low-dose anchors collapse as aggressively as `hard`.

## Mechanism Caveat Before L1b

L0b introduces post-final distractor slot updates using the same shared value
vocabulary. Before running L1 on the formal L0b dataset, update the mechanism
span extraction to use the target-slot `value_mentions` metadata rather than
global string matching. Otherwise distractor occurrences of the same value could
pollute current/stale span attribution.
