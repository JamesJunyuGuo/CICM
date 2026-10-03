# Stage L Report

## Positioning and Expected Value

Stage L fills the gap left by ICF-Bench and Stage G: realistic preference-dialogue surface, controlled repeated bindings, program-verifiable labels, identifiable mechanism probes, and in-distribution causal tests.

Honest expected value: the mechanism step is likely to land (repeated bindings make the probe identifiable). The causal step is high-value but uncertain -- prior causal transfer results were weak, so both causal branches are pre-registered as publishable.

## L-0 Behavior

- Data: `data/stage_l/l0b_hard.jsonl`
- Rows: `results/stage_l/l0b_sweep_a100/hard_rows.jsonl`
- n: 70
- Overall counts: `{'correct_current': 40, 'within_stale': 30}`
- Overall accuracy: 0.5714
- Overall within-stale share among failures: 1.0000

Per dose:

- k=1: n=14, counts=`{'correct_current': 12, 'within_stale': 2}`, accuracy=0.8571, within_stale_failure_share=1.0000
- k=2: n=14, counts=`{'correct_current': 11, 'within_stale': 3}`, accuracy=0.7857, within_stale_failure_share=1.0000
- k=3: n=14, counts=`{'within_stale': 8, 'correct_current': 6}`, accuracy=0.4286, within_stale_failure_share=1.0000
- k=4: n=14, counts=`{'within_stale': 9, 'correct_current': 5}`, accuracy=0.3571, within_stale_failure_share=1.0000
- k=6: n=14, counts=`{'correct_current': 6, 'within_stale': 8}`, accuracy=0.4286, within_stale_failure_share=1.0000

L-1 pre-registered readings:

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

L-2 pre-registered readings (kill-gate):

- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).
- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.
