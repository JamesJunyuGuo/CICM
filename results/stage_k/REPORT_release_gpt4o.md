# Stage K Report

## Scope

Stage K tests stale-binding as a real-deployment phenomenon on ICF-Bench. This report is isolated from Stage J and uses only `results/stage_k/` artifacts.

## K1 Behavioral Error Signature

- Model/source: `gpt4o1120-Eval-release-answer` from `external_data/icf_bench_release_answer_files`
- Headline: within-stale fraction among Forget-form failures, not accuracy drop.
- Program-verifiable scenarios: Dynamic Preference and Instructional Forgetting.
- LLM-judge labels are not used for the headline.

Pre-registered reading:

> within-stale dominates (pooled CI lower bound >= 0.70) in >=2 program-verifiable scenarios -> the stale-binding error signature replicates on independent real dialogue; proceed to K2. Otherwise -> report the boundary honestly (the synthetic signature is partly setup-specific / real-world failure is more heterogeneous) and STOP.

| Scenario | n kept | Forget acc | Forget failures | within-stale / failures | 95% CI | no-forget reference rate |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| dynamic_preference | 783 | 0.124 | 686 | 0.821 | [0.793, 0.848] | 0.870 |
| instructional_forgetting | 984 | 0.185 | 802 | 0.843 | [0.818, 0.869] | 0.766 |

K1 gate: `PASS`.
Qualifying scenarios: dynamic_preference, instructional_forgetting.

## Artifacts

- `k1_rows.jsonl`
- `k1_summary.json`
