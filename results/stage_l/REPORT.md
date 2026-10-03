# Stage L Report

## Positioning and Expected Value

Stage L fills the gap left by ICF-Bench and Stage G: realistic preference-dialogue surface, controlled repeated bindings, program-verifiable labels, identifiable mechanism probes, and in-distribution causal tests.

Honest expected value: the mechanism step is likely to land (repeated bindings make the probe identifiable). The causal step is high-value but uncertain -- prior causal transfer results were weak, so both causal branches are pre-registered as publishable.

## L-0 Behavior

- Data: `data/stage_l/cicm.jsonl`
- Rows: `results/stage_l/l0_rows.jsonl`
- n: 1050
- Overall counts: `{'correct_current': 1050}`
- Overall accuracy: 1.0000
- Overall within-stale share among failures: n/a

Per dose:

- k=1: n=210, counts=`{'correct_current': 210}`, accuracy=1.0000, within_stale_failure_share=n/a
- k=2: n=210, counts=`{'correct_current': 210}`, accuracy=1.0000, within_stale_failure_share=n/a
- k=3: n=210, counts=`{'correct_current': 210}`, accuracy=1.0000, within_stale_failure_share=n/a
- k=4: n=210, counts=`{'correct_current': 210}`, accuracy=1.0000, within_stale_failure_share=n/a
- k=6: n=210, counts=`{'correct_current': 210}`, accuracy=1.0000, within_stale_failure_share=n/a

L-1 pre-registered readings:

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

L-2 pre-registered readings (kill-gate):

- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).
- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.

## L-1 Mechanism

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value decodable on cross-slot trials → **recency-over-identity selection failure**: the target-slot value is represented, but a recent other-slot value is selected.
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

Analysis discipline applied here: per-cell rows are the result; aggregate rows are included only as equal-cell-size diagnostics and are not prevalence claims. L-0 crossover is treated as a conditioned tie/recency-gradient, not a clean global reversal.

- Stage-A probe reuse: `protocol_reused_weights_not_loadable`. No serialized Stage-A linear probe weights exist in results/stage_a, and the Stage-A synthetic label space is incompatible with Stage-L natural values. L-1 therefore freezes the Stage-A probe protocol: decision-position residual stream, linear multinomial probe, standardization, GroupKFold, and shuffle null.
- Label counts: `{'within_stale': 589, 'correct_current': 471, 'other': 64, 'cross_slot': 76}`
- Factorial cell counts: `{'same_near__other_far': 200, 'same_near__other_mid': 200, 'same_near__other_near2': 200, 'same_far__other_far': 200, 'same_far__other_mid': 200, 'same_far__other_near2': 200}`
- Probe layer: 28; splitter: `GroupKFold(n_splits=5)`; n_classes=21; CV accuracy=0.9125.
- Length control: `probe_true_current_score ~ log(prompt_tokens)` fit once on `all_l1_rows` with n=1200.
- Within-stale true-current score: 0.8485; value-label shuffle95 [0.03186157155764323, 0.06385486765497339]; above null: `True`.
- Probe true-current score within_stale - correct_current: raw -0.0790 CI [-0.1046, -0.0512], length-controlled -0.0746 CI [-0.1024, -0.0466], shuffle95 [-0.02735263952707336, 0.028704098252936183].
- Attention stale/(stale+current) within_stale - correct_current: raw 0.2246 CI [0.2081, 0.2403], length-controlled 0.1939 CI [0.1791, 0.2095], shuffle95 [-0.017976849339346047, 0.017889424058326095].
- L-1 status: `selection`.
- L-1 gate pass: `True`.
- Pre-registered reading selected: Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result). Nuance: true-current probe score is lower on within-stale than correct trials after length control, and stale-token attention is strongly elevated, so Stage L shows selection failure with a small retention/retrieval degradation rather than a pure no-loss story.

### Within-Stale Failure Pool

- Overall diagnostic: n=589, true-current score=0.8485, above value-label shuffle95=yes.

| same-slot stale | other-slot distance | labels in cell | fail n | true-current score | above shuffle95 | raw delta vs correct | length-controlled delta vs correct | length-null95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far | far | `{'correct_current': 149, 'cross_slot': 17, 'within_stale': 34}` | 34 | 0.8311 | yes | -0.0305 | -0.0274 | [-0.1071, 0.0945] |
| far | mid | `{'correct_current': 122, 'cross_slot': 23, 'other': 39, 'within_stale': 16}` | 16 | 0.9479 | yes | -0.0142 | -0.0138 | [-0.0644, 0.0374] |
| far | near2 | `{'correct_current': 125, 'other': 10, 'within_stale': 32, 'cross_slot': 33}` | 32 | 0.8405 | yes | -0.1260 | -0.1251 | [-0.0817, 0.0621] |
| near | far | `{'within_stale': 189, 'correct_current': 11}` | 189 | 0.8500 | yes | -0.1310 | -0.1235 | [-0.1248, 0.1686] |
| near | mid | `{'correct_current': 32, 'within_stale': 167, 'other': 1}` | 167 | 0.8138 | yes | -0.0670 | -0.0593 | [-0.1015, 0.1201] |
| near | near2 | `{'within_stale': 151, 'other': 14, 'correct_current': 32, 'cross_slot': 3}` | 151 | 0.8799 | yes | -0.0980 | -0.0964 | [-0.0755, 0.0850] |

### Cross-Slot Failure Pool

- Overall diagnostic: n=76, true-current score=0.8651, above value-label shuffle95=yes.

| same-slot stale | other-slot distance | labels in cell | fail n | true-current score | above shuffle95 | raw delta vs correct | length-controlled delta vs correct | length-null95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far | far | `{'correct_current': 149, 'cross_slot': 17, 'within_stale': 34}` | 17 | 0.7942 | yes | -0.0674 | -0.0678 | [-0.1561, 0.1276] |
| far | mid | `{'correct_current': 122, 'cross_slot': 23, 'other': 39, 'within_stale': 16}` | 23 | 0.7620 | yes | -0.2000 | -0.1959 | [-0.0890, 0.0660] |
| far | near2 | `{'correct_current': 125, 'other': 10, 'within_stale': 32, 'cross_slot': 33}` | 33 | 0.9691 | yes | 0.0027 | 0.0081 | [-0.0581, 0.0352] |
| near | far | `{'within_stale': 189, 'correct_current': 11}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | mid | `{'correct_current': 32, 'within_stale': 167, 'other': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | near2 | `{'within_stale': 151, 'other': 14, 'correct_current': 32, 'cross_slot': 3}` | 3 | 0.9137 | yes | -0.0642 | -0.0565 | [-0.0759, 0.0316] |
