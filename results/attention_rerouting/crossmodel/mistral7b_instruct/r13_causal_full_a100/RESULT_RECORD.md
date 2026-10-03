# Mistral-7B Stage R13 frozen result record

Recorded: 2026-09-17

This note freezes the complete scientific reading of the Mistral confirmation
run. It supplements, but does not replace or rewrite, the machine-generated
`summary.json`, `REPORT.md`, and per-example JSONL files.

## Adjudication

The direct-retrieval intervention produced a large, model-specific held-out
effect. Accuracy increased from 52/960 (0.0542) to 781/960 (0.8135), a raw gain
of 0.7594. The semantic-ID clustered paired-bootstrap estimate was 0.7501 with
95% CI [0.7033, 0.7912]. The targeted route corrected 564/674 (0.8368) baseline
within-slot stale responses and preserved 51/52 (0.9808) baseline-correct
responses.

The machine verdict remains `confirmation_gate_failed` because the frozen gate
required at least 100 baseline-correct confirmation examples and only 52 were
available. Every retrieval effect, specificity, identity, preservation, and
per-cell check passed. This is therefore evidence for a large held-out
retrieval correction with limited precision for preservation, not a pass of
all preregistered gates.

The same frozen intervention did not transfer to the derived-decision task.
That arm changed accuracy from 69/960 (0.0719) to 71/960 (0.0740), raw gain
0.0021, with clustered 95% CI [-0.0022, 0.0071]. This is a clean task-boundary
result and must not be merged with the direct-retrieval effect.

## Run and substrate

- Model: `mistralai/Mistral-7B-Instruct-v0.3`.
- Data: `data/cicm/cicm_natural_factorial_otherdist_l0.jsonl`.
- Data design: six identity-by-recency factorial cells, 200 rows per cell.
- Split: 240 calibration rows and 960 untouched confirmation rows; confirmation
  contains 160 rows per cell.
- Primary confirmation exclusions: none.
- Confirmation semantic clusters: 210.
- Decoding: greedy, maximum 8 new tokens.
- Bootstrap: 2,000 semantic-ID clustered paired replicates.
- Controls: 16 random-position routes, 64 layer-matched random-head sets,
  opposite-direction route, explicit-current-state recap, and beta=0 identity.
- Job: Slurm `20767071`, A100 40 GB, account/partition
  `cis250190-gpu/gpu`.
- Slurm state: `COMPLETED`, exit `0:0`.
- Runtime: 05:32:54 of a 06:15:00 allocation.
- Peak batch MaxRSS reported by Slurm: 17,166,860 KB.

## Intervention and freeze

The intervention is training-free, model-internal attention-logit routing at
the final answer query. For selected query heads, it subtracts an adaptive
shift beta from automatically detected stale same-slot positions and adds
beta to the automatically detected current-write positions. Specifically,
delta = logsumexp(current scores) - logsumexp(stale scores), and
beta = min(8, max(0, (margin - delta)/2)). The frozen margin is 2.0,
not a fixed per-token shift of 2.0. This method-description correction was
recorded on 2026-09-18; no experimental numbers or artifacts changed. Confirmation
routing may read the raw dialogue, declared slot schema, tokenizer offsets,
and the calibration-frozen parameters; it does not use confirmation outcomes
or gold value spans.

The A100 calibration used 120 discovery and 120 validation rows. Discovery had
9 baseline-correct and 78 within-stale first-token-identifiable rows after six
current/stale first-token collision exclusions. Candidate-token agreement was
85/87 (0.9770). The asymmetric discovery-pool gate was frozen at at least five
correct and at least 20 within-stale rows.

Validation selected the top 32 causally ranked heads with margin 2.0. On the
120 validation rows, retrieval accuracy changed from 0.1000 to 0.7833 (gain
0.6833), correct preservation was 1.0000, within-stale correction was 0.8095,
and the opposite-direction gain was -0.0833. These parameters were frozen
before the 960 confirmation rows were opened.

Frozen heads, in rank order:

`L19H9, L19H16, L16H29, L18H3, L31H19, L21H7, L19H8, L16H1,`
`L31H17, L18H12, L25H29, L31H18, L31H4, L24H15, L18H30, L24H5,`
`L30H1, L26H17, L21H11, L24H21, L31H6, L20H14, L28H25, L29H22,`
`L27H29, L29H9, L30H2, L20H6, L17H0, L31H31, L22H1, L26H6`.

The route audit matched all 1,200 dataset rows exactly. The hooked identity arm
changed zero responses and had maximum absolute first-token score change 0.0.
The targeted route's maximum absolute first-token score change was 12.0625.

## Direct-retrieval confirmation

### Overall response composition

| Arm | Correct | Within stale | Cross-slot | Other | Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 52 | 674 | 9 | 225 | 0.0542 |
| Targeted route | 781 | 73 | 10 | 96 | 0.8135 |
| Opposite route | 17 | 737 | 6 | 200 | 0.0177 |
| Explicit recap | 945 | 4 | 0 | 11 | 0.9844 |

### Targeted transitions

| Baseline response | Targeted response | Count |
|---|---|---:|
| correct current | correct current | 51 |
| correct current | other | 1 |
| within stale | correct current | 564 |
| within stale | within stale | 66 |
| within stale | cross-slot | 6 |
| within stale | other | 38 |
| cross-slot | correct current | 4 |
| cross-slot | cross-slot | 4 |
| cross-slot | other | 1 |
| other | correct current | 162 |
| other | within stale | 7 |
| other | other | 56 |

Conditional correction/preservation rates were 51/52 (0.9808) for baseline
correct responses, 564/674 (0.8368) for within-stale responses, 4/9 (0.4444)
for cross-slot responses, and 162/225 (0.7200) for other responses.

### Factorial cells

| Same-slot stale distance | Other-slot distance | n | Baseline | Targeted | Gain |
|---|---|---:|---:|---:|---:|
| far | far | 160 | 0.1500 | 0.7125 | 0.5625 |
| far | mid | 160 | 0.0688 | 0.8000 | 0.7313 |
| far | near-2 | 160 | 0.0688 | 0.9438 | 0.8750 |
| near | far | 160 | 0.0000 | 0.8188 | 0.8188 |
| near | mid | 160 | 0.0063 | 0.8125 | 0.8063 |
| near | near-2 | 160 | 0.0313 | 0.7938 | 0.7625 |

The gain is positive in all six cells. The result is not driven by one
identity-by-recency condition.

### Specificity and direction controls

- Targeted raw accuracy gain: 0.7594.
- Targeted clustered paired-bootstrap estimate: 0.7501, 95% CI
  [0.7033, 0.7912].
- Random-position gain, 16 routes: mean 0.00026; empirical 95% interval
  [-0.00104, 0.00208]; maximum 0.00208.
- Layer-matched random-head gain, 64 sets: mean -0.00485; empirical 95%
  interval [-0.01503, 0.00445]; maximum 0.00833.
- Opposite-direction raw gain: -0.03646; clustered paired-bootstrap estimate
  -0.04111, 95% CI [-0.06175, -0.02452].
- Explicit-recap raw gain: 0.93021; clustered paired-bootstrap estimate
  0.92524, 95% CI [0.90047, 0.94865].

The targeted effect is far above both random-control distributions, reverses
sign under the opposite route, and coexists with an exact identity/no-op arm.

### Frozen retrieval gate audit

| Check | Result | Reading |
|---|---|---|
| Route audit exact | pass | 1,200/1,200 |
| Identity exact | pass | zero score or response change |
| Correct pool at least 100 | **fail** | 52/100 |
| Within-stale pool at least 100 | pass | 674/100 |
| Gain CI above zero | pass | lower endpoint 0.7033 |
| Correct preservation at least 0.95 | pass | 51/52 = 0.9808 |
| Above random-position 95% bound | pass | 0.7594 > 0.00208 |
| Above random-head 95% bound | pass | 0.7594 > 0.00445 |
| Positive in at least four cells | pass | 6/6 |

Because the correct-pool count is a frozen gate, the official aggregate verdict
must remain `confirmation_gate_failed`. The result should not be relabeled as a
preregistered pass after observing the data.

## Derived-decision boundary

| Arm | Correct | Within stale | Other | Accuracy |
|---|---:|---:|---:|---:|
| Baseline | 69 | 552 | 339 | 0.0719 |
| Targeted route | 71 | 542 | 347 | 0.0740 |
| Opposite route | 72 | 545 | 343 | 0.0750 |
| Explicit recap | 863 | 69 | 28 | 0.8990 |

For the targeted route, correct preservation was 67/69 (0.9710) and only
1/552 (0.0018) within-stale derived responses became correct. The targeted
gain did not exceed either random control, the CI included zero, and gains were
not consistently positive across cells. The explicit recap remained effective,
showing that the task itself was answerable when the current state was made
explicit; the calibrated retrieval route did not transfer to this computation.

## Claim boundary for later writing

Supported:

- On controlled, parseable natural dialogue, a Mistral-specific set of
  calibration-discovered attention heads provides a large training-free
  correction for direct current-binding retrieval on untouched held-out rows.
- The effect is direction-specific, position-specific, head-set-specific, and
  positive in every preregistered factorial cell.
- The result extends the attention-routing correction to Mistral: it corrected
  564/674 within-stale errors while preserving 51/52 initially correct
  responses. In this held-out set, the correction therefore caused almost no
  observed damage to answers that were already correct.

Required paper-facing qualification:

- Report preservation as 51/52 rather than presenting 0.981 without its
  denominator. The exact count communicates the available support without
  introducing internal gate terminology into the paper.
- Report the derived-decision null. The method corrects direct retrieval; it
  does not generally improve downstream computation using the retrieved state.
- Parameters are model-specific: 32 heads and margin 2.0 were calibrated for
  this Mistral checkpoint. No universal head set or universal margin is shown.
- The intervention requires local access to model attention logits and an
  automatically parseable binding schema. It is not demonstrated as a
  text-only API method or on unrestricted open-ended dialogue.
- This run does not establish population prevalence of stale binding.

Internal audit note, not required as paper narrative:

- The preregistered automation marked the run `confirmation_gate_failed`
  because its conservative baseline-correct pool target was 100 and the
  observed pool was 52. Preserve that fact in experiment records, but do not
  frame the paper around internal gate mechanics. The paper should report the
  phenomenon, exact denominators, uncertainty, and controls.

Not supported:

- `The correction is universal or parameter-free.`
- `The correction fixes all forms of contextual forgetting or reasoning.`
- `The same intervention improves derived decisions.`

## Artifact index

- Machine summary and official gate: `summary.json`.
- Concise generated report: `REPORT.md`.
- Retrieval aggregate: `retrieval/task_summary.json`.
- Retrieval raw arms: `retrieval/baseline_rows.jsonl`,
  `retrieval/routed_rows.jsonl`, `retrieval/opposite_rows.jsonl`, and
  `retrieval/recap_rows.jsonl`.
- Retrieval controls: `retrieval/random_position_controls.jsonl` and
  `retrieval/random_head_controls.jsonl`.
- Derived-decision aggregate and raw arms: `derived_decision/task_summary.json`
  and the JSONL files in `derived_decision/`.
- Frozen calibration: `../r13_causal_calibration_a100/summary.json`,
  `../r13_causal_calibration_a100/REPORT.md`, and
  `../r13_causal_calibration_a100/discovery/head_summary.json`.
- Slurm logs:
  `results/attention_rerouting/logs/r13_mistral7b_causal_full_a100_20767071.out` and
  `results/attention_rerouting/logs/r13_mistral7b_causal_full_a100_20767071.err`.
