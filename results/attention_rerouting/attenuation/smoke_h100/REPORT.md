# Stage R2: Fresh-seed causal-head attenuation

**Verdict:** `attenuation_correction_gate_failed`.

The intervention attenuates a discovery-frozen 12-head set and receives no answer value, token id, or write span at inference time.

## Fresh confirmation

| k=6 n | gamma | stale correction | correct preservation | net gain | paired 95% CI | random net-gain 95% |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.90 | 0.125 | 1.000 | 0.083 | [0.0, 0.25] | [0.0, 0.0] |

Fresh k=0 net change: 0.000. Gamma=1 max logit change: 0.0.

## Per-template net gain

- arrow: 0.250
- current: 0.000
- latest: 0.000

## Output gates

- eligible_calibration_gamma: **PASS**
- identity_exact_zero: **PASS**
- targeted_gain_ci_above_zero: **FAIL**
- net_gain_exceeds_random95: **PASS**
- correct_preservation: **PASS**
- no_overwrite_drop_le_002: **PASS**
- beats_opposite: **PASS**
- multi_template_support: **FAIL**
- all_pass: **FAIL**

## Pre-final qualifier

Using 5 heads at layers <= 25, net gain was 0.000 (95% CI [0.0, 0.0]).

- eligible_calibration_gamma: **PASS**
- gain_ci_above_zero: **FAIL**
- net_gain_exceeds_random95: **FAIL**
- correct_preservation: **PASS**
- no_overwrite_drop_le_002: **PASS**
- multi_template_support: **FAIL**
- all_pass: **FAIL**

## Scope

fresh-seed controlled Qwen2.5-1.5B test; not adaptive or deployment-ready
