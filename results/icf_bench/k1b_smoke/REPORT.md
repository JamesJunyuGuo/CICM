# Stage K Report

## Scope

Stage K tests stale-binding as a real-deployment phenomenon on ICF-Bench. This report is isolated from Stage J and uses only `results/icf_bench/` artifacts.

## K1 Behavioral Error Signature

- Model/source: `Qwen/Qwen2.5-0.5B-Instruct-smoke20` from `qwen_self_generated_k1b_smoke`
- Headline: within-stale fraction among Forget-form failures, not accuracy drop.
- Program-verifiable scenarios: Dynamic Preference and Instructional Forgetting.
- LLM-judge labels are not used for the headline.

Pre-registered reading:

> within-stale dominates (pooled CI lower bound >= 0.70) in >=2 program-verifiable scenarios -> the stale-binding error signature replicates on independent real dialogue; proceed to K2. Otherwise -> report the boundary honestly (the synthetic signature is partly setup-specific / real-world failure is more heterogeneous) and STOP.

| Scenario | n kept | Forget acc | Forget failures | within-stale / failures | 95% CI | no-forget reference rate |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| dynamic_preference | 20 | 0.000 | 20 | 0.100 | [0.000, 0.250] | 0.100 |
| instructional_forgetting | 20 | 0.050 | 19 | 1.000 | [1.000, 1.000] | 0.900 |

K1 gate: `STOP`.
Qualifying scenarios: instructional_forgetting.

## Artifacts

- `k1_rows.jsonl`
- `k1_summary.json`
