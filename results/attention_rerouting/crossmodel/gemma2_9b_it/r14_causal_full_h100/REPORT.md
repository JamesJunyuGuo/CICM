# Gemma-2-9B-IT retrieval confirmation: completed primary task

Recorded 2026-09-18. Model: `google/gemma-2-9b-it`.

## Completion and stop decision

Slurm job `20801835` timed out after 03:40:24 against a 03:40:00 allocation.
The retrieval task completed and saved its four raw response arms, all 16
random-position controls, all 64 layer-matched random-head controls, and
`retrieval/task_summary.json`. Derived decision reached random-head 30/64 in
the log before timeout. Only its identity diagnostic was persisted; its main
responses, control results, and task summary are unavailable.

The PI authorized stopping on 2026-09-18 without further jobs. Retrieval can be
reported from the completed artifacts. Derived decision is incomplete and
unassessed, not a negative or zero effect. No combined full-run summary was
emitted by the timed-out process. Other models' derived-task results cannot
substitute for this missing result.

## Frozen protocol

- Data: `data/cicm/cicm_natural_factorial_otherdist_l0.jsonl`.
- Calibration: 240 rows, split into 120 discovery and 120 validation.
- Held-out retrieval: 960 rows, 160 in each of six factorial cells.
- Configuration: top 32 Gemma-specific causal-gradient-ranked query heads,
  adaptive margin 4.0; frozen from
  `../r14_causal_calibration_h100/summary.json` before confirmation.
- Validation accuracy: 0.5000 to 0.9083; preservation 1.0000; stale correction
  0.90196. These are calibration results, not confirmation estimates.
- The selected margin sets the desired current-versus-stale attention-logit
  mass advantage. The actual balanced logit shift is adaptive, not a constant
  shift of 4.0. Head identities and discovery details remain in the frozen
  calibration summary.
- H100, batch size 6, greedy decoding, maximum 8 generated tokens.
- 2,000 semantic-ID clustered paired-bootstrap replicates, 210 clusters.
- Route audit passed. Identity remained measured but was excluded from the
  decision gate by explicit PI instruction before this run.

## Retrieval results

| Metric | Result |
|---|---:|
| Baseline correct | 452/960 = 47.0833% |
| Targeted correct | 882/960 = 91.8750% |
| Raw accuracy gain | +44.7917 percentage points |
| Equal-semantic-cluster mean gain | +45.0635 percentage points |
| Clustered gain bootstrap 95% CI | [+39.6347, +50.9927] percentage points |
| Within-stale corrected | 407/455 = 89.4505% |
| Initially correct preserved | 452/452 = 100% |
| Cross-slot corrected | 17/42 = 40.4762% |
| Other corrected | 6/11 = 54.5455% |

The stored confidence interval belongs to the cluster-based estimate, not a
separately computed row-weighted interval. The sensitivity entry for all 960
rows has the same point estimates and a slightly different bootstrap interval;
use the primary `routed` entry consistently for reporting.

| Response type | Baseline | Targeted | Opposite | Recap |
|---|---:|---:|---:|---:|
| Correct current | 452 | 882 | 186 | 795 |
| Within stale | 455 | 48 | 738 | 164 |
| Cross-slot | 42 | 25 | 30 | 1 |
| Other | 11 | 5 | 6 | 0 |

Every targeted transition is either an unchanged response category or a
correction: correct->correct 452; stale->correct 407; stale->stale 48;
cross-slot->correct 17; cross-slot->cross-slot 25; other->correct 6;
other->other 5. These counts concern categories, not exact response strings.

## Controls

| Arm | Mean raw gain (points) | Empirical 95% range across random sets |
|---|---:|---:|
| Targeted | +44.7917 | not a random-set distribution |
| Random positions, 16 sets | +0.1888 | [-0.0651, +0.3776] |
| Layer-matched random heads, 64 sets | -3.0371 | [-7.0677, +1.3229] |
| Opposite direction | -27.7083 | not a random-set distribution |
| Explicit current-state recap | +35.7292 | not a random-set distribution |

The best random-head gain was +1.7708 points; the best random-position gain
was +0.4167 points. Targeted routing reached 91.875% accuracy versus 82.8125%
for this explicit-recap implementation. This is a point-estimate comparison;
no new targeted-versus-recap paired significance test was computed here.
Opposite routing preserved only 186/452 initially correct responses and
corrected no baseline stale answers.

## Per-cell results

Rates and gains below are percentages and percentage points, respectively.

| Same-slot old | Other-slot | n | Baseline | Targeted | Gain |
|---|---|---:|---:|---:|---:|
| far | far | 160 | 73.125 | 87.500 | +14.375 |
| far | mid | 160 | 73.125 | 96.875 | +23.750 |
| far | near2 | 160 | 85.000 | 96.250 | +11.250 |
| near | far | 160 | 7.500 | 99.375 | +91.875 |
| near | mid | 160 | 38.125 | 89.375 | +51.250 |
| near | near2 | 160 | 5.625 | 81.875 | +76.250 |

All six cells improve. The balanced aggregate describes this factorial, not
the prevalence or average improvement in unrestricted user dialogue.

## Identity diagnostic and CPU sensitivity

The full retrieval run's identity diagnostic reports seven changed responses
and a maximum first-token score difference of 1.8125, with no non-finite
entries. The separate diagnostic job `20801083` found exactly the same changed
IDs and differences for first-baseline versus identity and first-baseline
versus repeated baseline. Repeated baseline versus identity was exactly equal
in scores and responses. This suggests an execution-order/state difference;
its underlying cause has not been established, and exact identity relative to
the first baseline must not be claimed.

A post-hoc sensitivity check used only saved retrieval rows, excluding the
seven changed-response IDs listed in `retrieval/identity_diagnostic.json`:

| Metric | Remaining rows |
|---|---:|
| n | 953 |
| Baseline correct | 451 |
| Targeted correct | 875 |
| Raw gain | 424/953 = +44.4911 points |
| Stale corrected | 402/450 = 89.3333% |
| Correct preserved | 451/451 = 100% |

This sensitivity check does not replace the primary 960-row analysis or
establish numerical equivalence of all execution paths. It shows that the
observed correction is not concentrated in the seven response discrepancies.
If only those seven baseline classifications change and the stored targeted
outputs are fixed, the full-set raw gain is bounded by +44.0625 to +45.5208
points. Exact repeated-baseline preservation cannot be reconstructed because
that diagnostic did not save its generated response rows.

## Audit and reporting scope

CPU verification checked 960 unique IDs in identical order across all four
retrieval arms; each arm's label counts match the task summary. Control files
contain all 16 and 64 records. The stored retrieval gate has `all_pass=true`
under the PI-approved identity exclusion, with `identity_exact=false` and
`identity_required=false`; preserve all fields as recorded.

Supported: a large Gemma-specific attention-routing correction for direct
binding retrieval on held-out controlled natural dialogue, strongly separated
from random controls, with no observed damage among 452 baseline-correct
responses. Model-specific calibration, a parseable binding schema, and access
to attention internals are required. The result does not establish unrestricted
context-management repair, universal heads, or a derived-decision benefit.

Artifact sources: `retrieval/task_summary.json`, the six retrieval JSONL files,
`retrieval/identity_diagnostic.json`,
`../r14_identity_diag_h100/retrieval/identity_repeat_diagnostic.json`, and
`../r14_causal_calibration_h100/summary.json`. Runtime and timeout evidence:
`results/attention_rerouting/logs/r14_gemma2_9b_it_causal_full_h100_20801835.{out,err}`.
