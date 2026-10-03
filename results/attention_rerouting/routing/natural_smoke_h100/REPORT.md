# Stage R9-C: Natural-dialogue stale-key routing

**Verdict:** `natural_dialogue_transfer_gate_failed`.

The R9-B correction-efficacy authorization does not reverse its failed head-specificity gate.

| Task | baseline acc. | routed acc. | net gain | 95% CI | preservation | stale correction | recap acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| retrieval | 0.389 | 0.389 | 0.000 | [0.0, 0.0] | 1.000 | 0.000 | 0.889 |
| derived_decision | 0.500 | 0.500 | 0.000 | [0.0, 0.0] | 1.000 | 0.000 | 0.833 |

## Gates

### retrieval

- route_audit_exact: **PASS**
- identity_exact: **PASS**
- adequate_correct_pool: **PASS**
- adequate_within_stale_pool: **PASS**
- net_gain_ci_above_zero: **FAIL**
- correct_preservation: **PASS**
- exceeds_random_position95: **FAIL**
- positive_in_four_cells: **FAIL**
- all_pass: **FAIL**

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
