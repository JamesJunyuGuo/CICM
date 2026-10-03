# Stage R11: meta-llama/Llama-3.1-8B-Instruct balanced attention routing

**Verdict:** `model_correction_gate_failed`.

Calibration selected beta=0.5 and eight model-specific heads.
Head discovery and beta selection used calibration only.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| derived_decision | 0.417 | 0.417 | 0.000 | [0.0, 0.0] | 1.000 | 0.000 |
| retrieval | 0.438 | 0.417 | -0.021 | [-0.08333333333333333, 0.0] | 0.952 | 0.000 |

## Selected heads

- L1H19: score=5.6313
- L19H3: score=5.1814
- L16H26: score=5.1069
- L16H19: score=4.6967
- L6H16: score=4.6248
- L3H26: score=4.4904
- L3H7: score=4.4000
- L20H13: score=4.3851

## Gates

### derived_decision
- adequate_correct_pool: **PASS**
- adequate_within_stale_pool: **PASS**
- all_pass: **FAIL**
- correct_preservation: **PASS**
- exceeds_random_position95: **FAIL**
- identity_exact: **PASS**
- net_gain_ci_above_zero: **FAIL**
- positive_in_four_cells: **FAIL**
- route_audit_exact: **PASS**

### retrieval
- adequate_correct_pool: **PASS**
- adequate_within_stale_pool: **PASS**
- all_pass: **FAIL**
- correct_preservation: **PASS**
- exceeds_random_position95: **FAIL**
- identity_exact: **PASS**
- net_gain_ci_above_zero: **FAIL**
- positive_in_four_cells: **FAIL**
- route_audit_exact: **PASS**

## Scope

Model-specific balanced attention routing on meta-llama/Llama-3.1-8B-Instruct over the controlled natural CICM grammar; not unrestricted-dialogue, closed-API, or unique-circuit evidence.
