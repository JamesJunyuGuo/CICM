# Stage R: Mechanism-guided inference-time correction

**Verdict:** `correction_gate_failed`.

The intervention uses one global correct-vs-stale direction per frozen causal head. It receives no answer value, token id, or write span at inference time.

## Held-out result

| alpha | correction | preserve correct | net accuracy gain | random correction 95% | random net-gain 95% |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.000 | 1.000 | 0.000 | [0.000, 0.000] | [0.000, 0.000] |

Paired net-gain 95% CI: [0.0, 0.0]. No-overwrite net change: 0.000.

## Mechanism mediation

Against 4 matched-norm random directions, targeted-minus-random downstream attention-margin change was -0.001 (paired 95% CI [-0.009258015950521035, 0.005271250406901001]); targeted exceeded the empirical random 95th percentile: False.

## Pre-final validity arm

Using 5 frozen heads at layers <=25, held-out net gain was 0.000 (95% CI [0.0, 0.0]); mediation-versus-random CI was [-0.0065976460774738585, 0.007287343343099009].

## Gates

- identity_exact_zero: **PASS**
- targeted_gain_ci_above_zero: **FAIL**
- correction_exceeds_random95: **FAIL**
- net_gain_exceeds_random95: **FAIL**
- correct_preservation: **PASS**
- no_overwrite_drop_le_002: **PASS**
- direction_specificity: **FAIL**
- attention_mediation_vs_random: **FAIL**
- upstream_nonfinal_effect: **FAIL**
- all_pass: **FAIL**

## Scope

controlled Qwen2.5-1.5B existence test; not a general or deployable repair
