# Stage R11: Qwen/Qwen2.5-14B-Instruct balanced attention routing

**Verdict:** `model_retrieval_correction_supported`.

Calibration selected beta=1.0 and eight model-specific heads.
Head discovery and beta selection used calibration only.

| Task | baseline | routed | gain | 95% CI | preservation | stale correction |
|---|---:|---:|---:|---:|---:|---:|
| derived_decision | 0.279 | 0.279 | 0.000 | [0.0, 0.0] | 1.000 | 0.000 |
| retrieval | 0.405 | 0.416 | 0.010 | [0.0035714285714285713, 0.0169861111111111] | 0.997 | 0.020 |

## Selected heads

- L38H35: score=9.9271
- L38H38: score=7.3159
- L38H36: score=7.0280
- L35H11: score=6.7318
- L36H23: score=6.5424
- L43H34: score=6.0721
- L37H16: score=6.0141
- L43H32: score=5.6861

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
