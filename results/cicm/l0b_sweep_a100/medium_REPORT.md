# Stage L Report

## Positioning and Expected Value

Stage L fills the gap left by ICF-Bench and Stage G: realistic preference-dialogue surface, controlled repeated bindings, program-verifiable labels, identifiable mechanism probes, and in-distribution causal tests.

Honest expected value: the mechanism step is likely to land (repeated bindings make the probe identifiable). The causal step is high-value but uncertain -- prior causal transfer results were weak, so both causal branches are pre-registered as publishable.

## L-0 Behavior

- Data: `data/cicm/l0b_medium.jsonl`
- Rows: `results/cicm/l0b_sweep_a100/medium_rows.jsonl`
- n: 70
- Overall counts: `{'correct_current': 39, 'within_stale': 29, 'same_slot_other': 2}`
- Overall accuracy: 0.5571
- Overall within-stale share among failures: 0.9355

Per dose:

- k=1: n=14, counts=`{'correct_current': 11, 'within_stale': 3}`, accuracy=0.7857, within_stale_failure_share=1.0000
- k=2: n=14, counts=`{'correct_current': 10, 'within_stale': 2, 'same_slot_other': 2}`, accuracy=0.7143, within_stale_failure_share=0.5000
- k=3: n=14, counts=`{'within_stale': 7, 'correct_current': 7}`, accuracy=0.5000, within_stale_failure_share=1.0000
- k=4: n=14, counts=`{'within_stale': 7, 'correct_current': 7}`, accuracy=0.5000, within_stale_failure_share=1.0000
- k=6: n=14, counts=`{'correct_current': 4, 'within_stale': 10}`, accuracy=0.2857, within_stale_failure_share=1.0000

L-1 pre-registered readings:

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

L-2 pre-registered readings (kill-gate):

- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).
- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.
