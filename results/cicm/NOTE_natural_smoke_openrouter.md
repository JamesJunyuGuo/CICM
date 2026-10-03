# Stage L Natural Smoke - OpenRouter

Date: 2026-07-21

## Inputs

- Data: `data/cicm/cicm_natural_smoke.jsonl`
- Datasheet: `data/cicm/cicm_natural_smoke_DATASHEET.md`
- Template cache: `data/cicm/cicm_natural_smoke_templates.json`
- Model: OpenRouter `qwen/qwen-2.5-7b-instruct`
- Decoding: temperature 0 via chat completions, max tokens 16

## API Artifacts

- Initial rows: `results/cicm/natural_smoke/openrouter_qwen25_7b_rows.jsonl`
- Initial summary: `results/cicm/natural_smoke/openrouter_qwen25_7b_summary.json`
- Retry input for 5 timed-out rows:
  `results/cicm/natural_smoke/openrouter_qwen25_7b_retry_call_error_input.jsonl`
- Retry rows:
  `results/cicm/natural_smoke/openrouter_qwen25_7b_retry_call_error_rows.jsonl`
- Final merged rows:
  `results/cicm/natural_smoke/openrouter_qwen25_7b_rows_retrymerged.jsonl`
- Final merged summary:
  `results/cicm/natural_smoke/openrouter_qwen25_7b_summary_retrymerged.json`

## Final Smoke Readout

After retrying API timeouts, all 20 rows have model responses.

- Overall counts: 12 `correct_current`, 5 `cross_slot`, 1 `within_stale`, 2 `other`
- Overall accuracy: 0.600
- Within-stale share among failures: 0.125

By dose:

- k=1: 4 correct
- k=2: 2 correct, 2 cross-slot
- k=3: 2 correct, 1 within-stale, 1 cross-slot
- k=4: 1 correct, 1 cross-slot, 2 other
- k=6: 3 correct, 1 cross-slot

## Interpretation

The natural dialogue upgrade succeeded structurally: slot values are natural and
slot-specific, OpenAI generated reusable natural templates, and the program injects
exact values with checked target-slot character spans.

Behaviorally, this smoke should NOT be scaled as-is because query ambiguity is a
confound. However, the `cross_slot` failures are not evidence to suppress
distractors; they are a real identity-vs-recency competition signal once the query
is cleaned up.

## Next Fix Before Full Natural L0

Keep natural slot values, OpenAI-generated coherent filler, and distractor slot
competition, but revise the natural generator before full-scale inference:

- make the final query more target-slot anchored while still natural;
- keep post-final distractor updates as recent-other-slot competitors;
- record competitor type and distance/salience metadata;
- rerun only API smoke before scaling.

Do not run L1/L2 from this natural smoke artifact.

Update: this cleanup was implemented in
`data/cicm/cicm_natural_anchor_smoke.jsonl`; see
`results/cicm/NOTE_natural_anchor_smoke_openrouter.md`.
