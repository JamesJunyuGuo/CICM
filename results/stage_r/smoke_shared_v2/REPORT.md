# Stage R: Mechanism-guided inference-time correction

**Verdict:** `correction_gate_failed`.

The intervention uses one global correct-vs-stale direction per frozen causal head. It receives no answer value, token id, or write span at inference time.

## Held-out result

| alpha | correction | preserve correct | net accuracy gain | random correction 95% |
|---:|---:|---:|---:|---:|
| 0.5 | 0.250 | 1.000 | 0.125 | [0.000, 0.000] |

Paired net-gain 95% CI: [0.0, 0.375]. No-overwrite net change: 0.000.

## Mechanism mediation

Targeted-minus-random downstream attention-margin change: -0.001 (95% CI [-0.014875156315975002, 0.020313237421214603]).

## Pre-final validity arm

Using 5 frozen heads at layers <=25, held-out net gain was 0.125 (95% CI [0.0, 0.375]); mediation-versus-random CI was [-0.01685850881040102, 0.014829669147729874].

## Gates

- identity_exact_zero: **PASS**
- targeted_gain_ci_above_zero: **FAIL**
- correction_exceeds_random95: **PASS**
- correct_preservation: **PASS**
- no_overwrite_drop_le_002: **PASS**
- direction_specificity: **FAIL**
- attention_mediation_vs_random: **FAIL**
- upstream_nonfinal_effect: **FAIL**
- all_pass: **FAIL**

## Scope

controlled Qwen2.5-1.5B existence test; not a general or deployable repair
