# Stage R9: Update-aware stale-key suppression

**Verdict:** `stale_key_suppression_gate_failed`.

| Route | n | beta | correction | preservation | net gain | paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 648 | 2.00 | 0.628 | 0.972 | 0.301 | [0.25462962962962965, 0.35185185185185186] |
| automatic | 648 | 2.00 | 0.628 | 0.972 | 0.301 | [0.25154320987654316, 0.3503086419753086] |

Random-head net-gain 95%: [0.02472993827160494, 0.422608024691358].
Random-position net-gain 95%: [-0.01273148148148148, 0.0].
Opposite-sign net gain: -0.323.
k=0 maximum logit change: 0.0.

## Oracle gates

- eligible_calibration_beta: **PASS**
- identity_exact_zero: **PASS**
- net_gain_ci_above_zero: **PASS**
- net_gain_exceeds_random_head95: **FAIL**
- net_gain_exceeds_random_position95: **PASS**
- correct_preservation: **PASS**
- multi_template_support: **PASS**
- k0_exact_noop: **PASS**
- beats_opposite: **PASS**
- all_pass: **FAIL**

## Automatic structured-log gates

- route_audit_exact: **PASS**
- net_gain_ci_above_zero: **PASS**
- net_gain_exceeds_random_head95: **FAIL**
- net_gain_exceeds_random_position95: **PASS**
- correct_preservation: **PASS**
- multi_template_support: **PASS**
- retains_half_oracle_gain: **PASS**
- k0_exact_noop: **PASS**
- all_pass: **FAIL**

## Scope

open-weight Qwen2.5-1.5B with explicitly structured update logs; not a natural-dialogue, cross-model, or closed-API repair
