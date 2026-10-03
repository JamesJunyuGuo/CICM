# Stage R10-balanced-natural-dialogue-routing: Natural-dialogue routing

**Verdict:** `controlled_natural_dialogue_retrieval_correction_only`.

The R9-B correction-efficacy authorization does not reverse its failed head-specificity gate.

| Task | baseline acc. | routed acc. | net gain | 95% CI | preservation | stale correction | recap acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| retrieval | 0.369 | 0.689 | 0.320 | [0.2864246031746031, 0.3660416666666666] | 1.000 | 0.414 | 0.882 |
| derived_decision | 0.310 | 0.300 | -0.010 | [-0.02365277777777778, -0.0020615079365079478] | 0.953 | 0.006 | 0.863 |

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
- correct_preservation: **PASS**
- exceeds_random_position95: **FAIL**
- positive_in_four_cells: **FAIL**
- all_pass: **FAIL**

## Scope

Qwen2.5-7B-Instruct routing transfer on controlled natural CICM grammar; not unique-head evidence, unrestricted dialogue generalization, cross-model transfer, or closed-API deployment
