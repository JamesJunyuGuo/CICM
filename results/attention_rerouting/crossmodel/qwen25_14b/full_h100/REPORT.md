# Stage R11: Qwen/Qwen2.5-14B-Instruct balanced attention routing

**Verdict:** `model_retrieval_correction_supported`.

Calibration selected beta=2.0 and eight model-specific heads.
Head discovery and beta selection used calibration only.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| derived_decision | 0.280 | 0.279 | -0.001 | [-0.002380952380952381, 0.0] | 0.996 | 0.000 |
| retrieval | 0.405 | 0.417 | 0.011 | [0.0014265873015873011, 0.0223829365079365] | 0.985 | 0.026 |

## Selected heads

- L38H35: score=10.0489
- L38H38: score=7.3881
- L38H36: score=7.1261
- L35H11: score=6.7905
- L36H23: score=6.6438
- L43H34: score=6.1977
- L37H16: score=6.1124
- L43H32: score=5.8101

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
- all_pass: **PASS**
- correct_preservation: **PASS**
- exceeds_random_position95: **PASS**
- identity_exact: **PASS**
- net_gain_ci_above_zero: **PASS**
- positive_in_four_cells: **PASS**
- route_audit_exact: **PASS**

## Scope

Model-specific balanced attention routing on Qwen/Qwen2.5-14B-Instruct over the controlled natural CICM grammar; not unrestricted-dialogue, closed-API, or unique-circuit evidence.
