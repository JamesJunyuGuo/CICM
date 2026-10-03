# Stage L Report

## Positioning and Expected Value

Stage L fills the gap left by ICF-Bench and Stage G: realistic preference-dialogue surface, controlled repeated bindings, program-verifiable labels, identifiable mechanism probes, and in-distribution causal tests.

Honest expected value: the mechanism step is likely to land (repeated bindings make the probe identifiable). The causal step is high-value but uncertain -- prior causal transfer results were weak, so both causal branches are pre-registered as publishable.

## L-0 Behavior

- Data: `data/cicm/cicm_natural_smoke.jsonl`
- Rows: `results/cicm/natural_smoke/qwen05b_rows.jsonl`
- n: 20
- Overall counts: `{'correct_current': 2, 'within_stale': 8, 'other': 6, 'same_slot_other': 3, 'cross_slot': 1}`
- Overall accuracy: 0.1000
- Overall within-stale share among failures: 0.4444

Per dose:

- k=1: n=4, counts=`{'correct_current': 1, 'within_stale': 2, 'other': 1}`, accuracy=0.2500, within_stale_failure_share=0.6667
- k=2: n=4, counts=`{'same_slot_other': 1, 'cross_slot': 1, 'within_stale': 1, 'other': 1}`, accuracy=0.0000, within_stale_failure_share=0.2500
- k=3: n=4, counts=`{'within_stale': 2, 'same_slot_other': 2}`, accuracy=0.0000, within_stale_failure_share=0.5000
- k=4: n=4, counts=`{'within_stale': 2, 'other': 2}`, accuracy=0.0000, within_stale_failure_share=0.5000
- k=6: n=4, counts=`{'correct_current': 1, 'within_stale': 1, 'other': 2}`, accuracy=0.2500, within_stale_failure_share=0.3333

L-1 pre-registered readings:

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

L-2 pre-registered readings (kill-gate):

- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).
- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.
