# R13-llama3b-causal-head-routing-confirmation held-out confirmation: meta-llama/Llama-3.2-3B-Instruct

**Verdict:** `retrieval_correction_only`.

Frozen before confirmation: top-32 causal heads, adaptive margin=4.0.

All 960 rows are untouched held-out confirmation rows.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| retrieval | 0.277 | 0.931 | 0.654 | [0.5940436507936508, 0.6962738095238095] | 1.000 | 0.950 |
| derived_decision | 0.231 | 0.234 | 0.003 | [-0.012466269841269839, 0.015638888888888865] | 0.914 | 0.027 |

## Retrieval controls

- Opposite-direction gain: -0.271.
- Explicit-recap gain: 0.411.
- Random-position gain interval: [-0.0048177083333333336, 0.0006510416666666666].
- Layer-matched random-head gain interval: [-0.121875, 0.008776041666666663].

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
