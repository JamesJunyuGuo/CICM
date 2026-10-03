# Stage R: Mechanism-guided inference-time correction

**Verdict:** `correction_gate_failed`.

The intervention uses one global correct-vs-stale direction per frozen causal head. It receives no answer value, token id, or write span at inference time.

## Held-out result

| alpha | correction | preserve correct | net accuracy gain | random correction 95% | random net-gain 95% |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.033 | 0.975 | 0.004 | [0.000, 0.029] | [-0.021, 0.004] |

Paired net-gain 95% CI: [-0.016736401673640166, 0.02510460251046025]. No-overwrite net change: -0.002.

## Mechanism mediation

Against 16 matched-norm random directions, targeted-minus-random downstream attention-margin change was 0.003 (paired 95% CI [0.001513496478398655, 0.0038062130610148157]); targeted exceeded the empirical random 95th percentile: True.

## Pre-final validity arm

Using 5 frozen heads at layers <=25, held-out net gain was -0.008 (95% CI [-0.02510460251046025, 0.008368200836820083]); mediation-versus-random CI was [0.0006327411731084257, 0.002226339975992852].

## Gates

- identity_exact_zero: **PASS**
- targeted_gain_ci_above_zero: **FAIL**
- correction_exceeds_random95: **PASS**
- net_gain_exceeds_random95: **FAIL**
- correct_preservation: **PASS**
- no_overwrite_drop_le_002: **PASS**
- direction_specificity: **PASS**
- attention_mediation_vs_random: **PASS**
- upstream_nonfinal_effect: **FAIL**
- all_pass: **FAIL**

## Scope

controlled Qwen2.5-1.5B existence test; not a general or deployable repair
