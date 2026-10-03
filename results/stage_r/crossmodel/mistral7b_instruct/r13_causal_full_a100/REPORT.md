# R13-mistral7b-causal-head-routing-confirmation held-out confirmation: mistralai/Mistral-7B-Instruct-v0.3

**Verdict:** `confirmation_gate_failed`.

Frozen before confirmation: top-32 causal heads, adaptive margin=2.0.

All 960 rows are untouched held-out confirmation rows.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| retrieval | 0.054 | 0.814 | 0.759 | [0.7033253968253969, 0.7911964285714287] | 0.981 | 0.837 |
| derived_decision | 0.072 | 0.074 | 0.002 | [-0.0022222222222222227, 0.007142857142857143] | 0.971 | 0.002 |

## Retrieval controls

- Opposite-direction gain: -0.036.
- Explicit-recap gain: 0.930.
- Random-position gain interval: [-0.0010416666666666667, 0.0020833333333333333].
- Layer-matched random-head gain interval: [-0.015026041666666667, 0.004453124999999992].

## Preregistered retrieval gate

- route_audit_exact: True
- identity_exact: True
- adequate_correct_pool: False
- adequate_within_stale_pool: True
- net_gain_ci_above_zero: True
- correct_preservation: True
- exceeds_random_position95: True
- exceeds_random_head95: True
- positive_in_four_cells: True
- all_pass: False
