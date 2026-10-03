# Stage L Report

## Positioning and Expected Value

Stage L fills the gap left by ICF-Bench and Stage G: realistic preference-dialogue surface, controlled repeated bindings, program-verifiable labels, identifiable mechanism probes, and in-distribution causal tests.

Honest expected value: the mechanism step is likely to land (repeated bindings make the probe identifiable). The causal step is high-value but uncertain -- prior causal transfer results were weak, so both causal branches are pre-registered as publishable.

## L-0 Behavior

- Data: `data/cicm/cicm.jsonl`
- Rows: `results/cicm/smoke/l0_rows.jsonl`
- n: 20
- Overall counts: `{'correct_current': 20}`
- Overall accuracy: 1.0000
- Overall within-stale share among failures: n/a

Per dose:

- k=1: n=20, counts=`{'correct_current': 20}`, accuracy=1.0000, within_stale_failure_share=n/a

L-1 pre-registered readings:

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

L-2 pre-registered readings (kill-gate):

- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).
- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.
## L-1 Mechanism

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

- Label counts: `{'correct_current': 20}`
- Probe layer: 24; splitter: `GroupKFold(n_splits=5)`; n_classes=7; CV accuracy=0.9000.
- Within-stale true-current score: nan; value-label shuffle95 [nan, nan]; above null: `False`.
- Probe true-current score within_stale - correct_current: raw nan CI [nan, nan], length-controlled nan CI [nan, nan], shuffle95 [nan, nan].
- Attention stale/(stale+current) within_stale - correct_current: raw nan CI [nan, nan], length-controlled nan CI [nan, nan], shuffle95 [nan, nan].
- L-1 gate pass: `False`.
- Pre-registered reading selected: Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.
## L-2 Causality

- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).
- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.

- n paired: 0
- arm counts: `{'identity': {'counts': {}, 'within_stale_rate': nan, 'correct_rate': nan}, 'random_matched_norm': {'counts': {}, 'within_stale_rate': nan, 'correct_rate': nan}, 'targeted': {'counts': {}, 'within_stale_rate': nan, 'correct_rate': nan}}`
- identity gate mismatches: 0
- targeted-minus-random stale-error reduction: `{'n': 0, 'mean': nan, 'ci': [nan, nan]}`
- L-2 gate pass: `False`.
- Pre-registered reading selected: Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.
