# Stage R11: Qwen/Qwen2.5-14B-Instruct balanced attention routing

**Verdict:** `model_correction_gate_failed`.

Calibration selected beta=8.0 and eight model-specific heads.
Head discovery and beta selection used calibration only.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| derived_decision | 0.333 | 0.354 | 0.021 | [0.0, 0.05555555555555555] | 1.000 | 0.033 |
| retrieval | 0.479 | 0.521 | 0.042 | [0.0, 0.1111111111111111] | 1.000 | 0.087 |

## Selected heads

- L38H35: score=9.3369
- L35H11: score=7.9665
- L38H38: score=7.6942
- L36H23: score=7.1503
- L37H16: score=7.0328
- L42H0: score=5.8616
- L36H21: score=5.7931
- L40H11: score=5.7306

## Gates

### derived_decision
- adequate_correct_pool: **PASS**
- adequate_within_stale_pool: **PASS**
- all_pass: **FAIL**
- correct_preservation: **PASS**
- exceeds_random_position95: **PASS**
- identity_exact: **PASS**
- net_gain_ci_above_zero: **FAIL**
- positive_in_four_cells: **FAIL**
- route_audit_exact: **PASS**

### retrieval
- adequate_correct_pool: **PASS**
- adequate_within_stale_pool: **PASS**
- all_pass: **FAIL**
- correct_preservation: **PASS**
- exceeds_random_position95: **PASS**
- identity_exact: **PASS**
- net_gain_ci_above_zero: **FAIL**
- positive_in_four_cells: **FAIL**
- route_audit_exact: **PASS**

## Scope

Model-specific balanced attention routing on Qwen/Qwen2.5-14B-Instruct over the controlled natural CICM grammar; not unrestricted-dialogue, closed-API, or unique-circuit evidence.
