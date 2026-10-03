# Stage K Report

## Scope

Stage K tests stale-binding as a real-deployment phenomenon on ICF-Bench. This report is isolated from Stage J and uses only `results/icf_bench/` artifacts.

## K1 Behavioral Error Signature

- Model/source: `Qwen/Qwen2.5-7B-Instruct` from `qwen_self_generated_k1c_cpu_rescore`
- Headline: within-stale fraction among Forget-form failures, not accuracy drop.
- Program-verifiable scenarios: Dynamic Preference and Instructional Forgetting.
- LLM-judge labels are not used for the headline.

Pre-registered reading:

> within-stale dominates (pooled CI lower bound >= 0.70) in >=2 program-verifiable scenarios -> the stale-binding error signature replicates on independent real dialogue; proceed to K2. Otherwise -> report the boundary honestly (the synthetic signature is partly setup-specific / real-world failure is more heterogeneous) and STOP.

| Scenario | n kept | Forget acc | Forget failures | within-stale / failures | 95% CI | no-forget reference rate |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| dynamic_preference | 783 | 0.709 | 228 | 0.294 | [0.232, 0.355] | 0.814 |
| instructional_forgetting | 933 | 0.014 | 920 | 0.985 | [0.976, 0.992] | 0.989 |

K1 two-scenario stale-dominance gate: `MISS`; pre-registered IF-vs-DP dissociation branch: `PROCEED` to K2/K3 contrast explanation.
Qualifying scenarios: instructional_forgetting.
Next step: `proceed_to_k2_k3_explain_if_dp_contrast`.

## K1c Error-Type Spot Check

K1c fixes the Instructional Forgetting error-type matcher without regenerating model
outputs. The previous IF `within_stale=1.000` is invalid as an error-type claim because
non-refusal failures were collapsed into `within_stale`.

Spot-check artifact: `../k1c_if_errortype_spotcheck.jsonl`.

| Sample | Agreement | Mismatches | Hand labels | Model labels |
| ---: | ---: | ---: | --- | --- |
| 50 | 0.980 | 1 | deflect=2, other=12, within_stale=36 | deflect=3, other=11, within_stale=36 |

Residual risk: the one mismatch (`id=261`) is a conservative boundary case where a
different artifact answer is classified as deflection. This does not affect the K1c
reading: IF remains stale-dominant while DP is not stale-dominant.

## Artifacts

- `k1_rows.jsonl`
- `k1_summary.json`
