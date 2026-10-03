# Stage R12 Llama held-out confirmation

**Verdict:** `llama_retrieval_correction_only`.

Frozen before confirmation: top-16 causal heads, adaptive margin=4.0.

All 960 rows were generated; the primary analysis uses 959 rows after excluding the single smoke-overlap row.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| retrieval | 0.417 | 0.886 | 0.469 | [0.4199047619047619, 0.5192876984126984] | 1.000 | 0.846 |
| derived_decision | 0.189 | 0.190 | 0.001 | [-0.02310119047619048, 0.022380952380952383] | 0.801 | 0.046 |

## Retrieval controls

- Opposite-direction gain: -0.282.
- Explicit-recap gain: 0.237.
- Random-position gain interval: [-0.006908237747653806, 0.0].
- Layer-matched random-head gain interval: [-0.10896767466110532, 0.030500521376433755].

## Preregistered retrieval gate

- route_audit_exact: True
- identity_exact: True
- adequate_correct_pool: True
- adequate_within_stale_pool: True
- net_gain_ci_above_zero: True
- correct_preservation: True
- exceeds_random_position95: True
- exceeds_random_head95: True
- positive_in_four_cells: True
- all_pass: True
