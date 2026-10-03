# Stage R: Mechanism-guided inference-time correction

**Verdict:** `correction_gate_failed`.

The intervention uses one global correct-vs-stale direction per frozen causal head. It receives no answer value, token id, or write span at inference time.

## Held-out result

| alpha | correction | preserve correct | net accuracy gain | random correction 95% | random net-gain 95% |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.025 | 0.991 | 0.008 | [0.000, 0.034] | [-0.008, 0.013] |

Paired net-gain 95% CI: [-0.00847457627118644, 0.025423728813559324]. No-overwrite net change: 0.002.

## Mechanism mediation

Against 16 matched-norm random directions, targeted-minus-random downstream attention-margin change was 0.001 (paired 95% CI [0.0004902131216866561, 0.0023023542035527614]); targeted exceeded the empirical random 95th percentile: True.

## Pre-final validity arm

Using 5 frozen heads at layers <=25, held-out net gain was 0.004 (95% CI [0.0, 0.012711864406779662]); mediation-versus-random CI was [-0.00037183889821797557, 0.0014714432163398896].

## Gates

- identity_exact_zero: **PASS**
- targeted_gain_ci_above_zero: **FAIL**
- correction_exceeds_random95: **FAIL**
- net_gain_exceeds_random95: **FAIL**
- correct_preservation: **PASS**
- no_overwrite_drop_le_002: **PASS**
- direction_specificity: **PASS**
- attention_mediation_vs_random: **PASS**
- upstream_nonfinal_effect: **FAIL**
- all_pass: **FAIL**

## Scope

controlled Qwen2.5-1.5B existence test; not a general or deployable repair
