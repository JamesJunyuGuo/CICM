# Stage L Natural Anchored Smoke - OpenRouter

Date: 2026-07-21

## Purpose

This is the legal cleanup after the first natural smoke. The first natural
version had real cross-slot failures, but some final queries were ambiguous. This
version removes that confound without suppressing distractor competition:

- final query is programmatically anchored to the target slot;
- post-final distractor slot updates are retained;
- each row records `competition` metadata for target current, same-slot stale,
  and recent other-slot values.

This frames Stage L as identity-vs-recency retrieval competition, not just
stale-binding replication.

## Inputs

- Data: `data/cicm/cicm_natural_anchor_smoke.jsonl`
- Datasheet: `data/cicm/cicm_natural_anchor_smoke_DATASHEET.md`
- Template cache reused from: `data/cicm/cicm_natural_smoke_templates.json`
- Model: OpenRouter `qwen/qwen-2.5-7b-instruct`
- Decoding: temperature 0 via chat completions, max tokens 16

## Outputs

- Rows: `results/cicm/natural_anchor_smoke/openrouter_qwen25_7b_rows.jsonl`
- Summary: `results/cicm/natural_anchor_smoke/openrouter_qwen25_7b_summary.json`
- Failure-mode competition summary:
  `results/cicm/natural_anchor_smoke/failure_mode_competition_summary.json`

## Readout

All 20 API calls completed without `call_error`.

- Overall counts: 14 `correct_current`, 3 `within_stale`, 3 `cross_slot`
- Accuracy: 0.700
- Within-stale share among failures: 0.500

By dose:

- k=1: 3 correct, 1 within-stale
- k=2: 4 correct
- k=3: 2 correct, 1 within-stale, 1 cross-slot
- k=4: 2 correct, 1 within-stale, 1 cross-slot
- k=6: 3 correct, 1 cross-slot

## Interpretation

The query ambiguity confound is reduced: compared with the first natural smoke,
there are no `other` errors and cross-slot failures fall from 5/20 to 3/20. But
cross-slot remains, which is the desired result under the updated framing:
recent other-slot values are genuine competitors, not artifacts to tune away.

In this smoke profile the distances are fixed by design:

- recent other-slot competitor: 2 messages before query;
- target current value: 14 messages before query;
- nearest same-slot stale value: 18 messages before query.

So this smoke does not estimate a distance curve yet. It establishes that, after
legal query cleanup, both same-slot stale and recent-other-slot failures survive.

## Next

Proceed with Stage L as an identity-vs-recency competition study:

- full natural anchored L0 can characterize failure-mode rates
  (`correct_current`, `within_stale`, `cross_slot`) under retained distractors;
- a follow-up generator sweep should systematically vary competitor type and
  distance/salience rather than suppressing distractors;
- L1 should condition on failure mode. For both `within_stale` and `cross_slot`,
  ask whether the target-slot current value is decodable from hidden state but
  not selected. If yes for cross-slot, the mechanism is recency-over-identity
  selection failure.
