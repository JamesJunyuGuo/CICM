# Stage R9: Update-aware stale-key suppression

**Verdict:** `limited_deployable_structured_log_correction_supported`.

| Route | n | beta | correction | preservation | net gain | paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 12 | 4.00 | 0.750 | 1.000 | 0.500 | [0.25, 0.8333333333333333] |
| automatic | 12 | 4.00 | 0.750 | 1.000 | 0.500 | [0.16666666666666666, 0.8333333333333333] |

Random-head net-gain 95%: [0.0875, 0.24583333333333332].
Random-position net-gain 95%: [-0.08124999999999999, -0.002083333333333335].
Opposite-sign net gain: -0.333.
k=0 maximum logit change: 0.0.

## Oracle gates

- eligible_calibration_beta: **PASS**
- identity_exact_zero: **PASS**
- net_gain_ci_above_zero: **PASS**
- net_gain_exceeds_random_head95: **PASS**
- net_gain_exceeds_random_position95: **PASS**
- correct_preservation: **PASS**
- multi_template_support: **PASS**
- k0_exact_noop: **PASS**
- beats_opposite: **PASS**
- all_pass: **PASS**

## Automatic structured-log gates

- route_audit_exact: **PASS**
- net_gain_ci_above_zero: **PASS**
- net_gain_exceeds_random_head95: **PASS**
- net_gain_exceeds_random_position95: **PASS**
- correct_preservation: **PASS**
- multi_template_support: **PASS**
- retains_half_oracle_gain: **PASS**
- k0_exact_noop: **PASS**
- all_pass: **PASS**

## Scope

open-weight Qwen2.5-1.5B with explicitly structured update logs; not a natural-dialogue, cross-model, or closed-API repair
