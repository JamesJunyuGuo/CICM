# R14-qwen3b-causal-head-routing-confirmation held-out confirmation: Qwen/Qwen2.5-3B-Instruct

**Verdict:** `retrieval_and_derived_correction_supported`.

Frozen before confirmation: top-32 causal heads, adaptive margin=4.0.

All 960 rows are untouched held-out confirmation rows.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| retrieval | 0.165 | 0.952 | 0.787 | [0.745079365079365, 0.8220714285714286] | 0.987 | 0.954 |
| derived_decision | 0.266 | 0.285 | 0.020 | [0.010317460317460319, 0.03150992063492063] | 0.992 | 0.030 |

## Retrieval controls

- Opposite-direction gain: -0.164.
- Explicit-recap gain: 0.828.
- Random-position gain interval: [-0.0027343750000000003, 0.0037760416666666667].
- Layer-matched random-head gain interval: [-0.14419270833333334, -0.0128385416666667].

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
