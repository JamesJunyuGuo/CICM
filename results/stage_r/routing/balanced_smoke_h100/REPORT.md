# Stage R10-balanced-natural-dialogue-routing: Natural-dialogue routing

**Verdict:** `controlled_natural_dialogue_retrieval_correction_only`.

The R9-B correction-efficacy authorization does not reverse its failed head-specificity gate.

| Task | baseline acc. | routed acc. | net gain | 95% CI | preservation | stale correction | recap acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| retrieval | 0.438 | 0.708 | 0.271 | [0.16666666666666666, 0.4305555555555556] | 1.000 | 0.435 | 0.854 |
| derived_decision | 0.458 | 0.417 | -0.042 | [-0.1388888888888889, 0.0] | 0.909 | 0.000 | 0.854 |

## Gates

### retrieval

- route_audit_exact: **PASS**
- identity_exact: **PASS**
- adequate_correct_pool: **PASS**
- adequate_within_stale_pool: **PASS**
- net_gain_ci_above_zero: **PASS**
- correct_preservation: **PASS**
- exceeds_random_position95: **PASS**
- positive_in_four_cells: **PASS**
- all_pass: **PASS**

### derived_decision

- route_audit_exact: **PASS**
- identity_exact: **PASS**
- adequate_correct_pool: **PASS**
- adequate_within_stale_pool: **PASS**
- net_gain_ci_above_zero: **FAIL**
- correct_preservation: **FAIL**
- exceeds_random_position95: **FAIL**
- positive_in_four_cells: **FAIL**
- all_pass: **FAIL**

## Scope

Qwen2.5-7B-Instruct routing transfer on controlled natural CICM grammar; not unique-head evidence, unrestricted dialogue generalization, cross-model transfer, or closed-API deployment
