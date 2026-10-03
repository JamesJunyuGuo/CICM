# Stage K Report

## Scope

K1d is API-only behavioral inference on ICF-Bench Dynamic Preference free-form prompts plus existing K1c Instructional Forgetting error-type repair. No hidden states, no GPU, no Stage-J artifacts.

## K1d Result

- DP primary model: `qwen/qwen-2.5-7b-instruct` via OpenRouter, temperature 0.
- Judge model: `openai/gpt-4o` via OpenRouter, temperature 0.
- Judge-vs-human calibration: 36/40 = 0.900.
- Headline allowed by validation threshold: `True`.

| Scenario / format | n | Forget acc | Failures | within-stale / failures | 95% CI | NoForget/reference |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Dynamic Preference free-form | 783 | 0.693 | 240 | 0.821 (197/240) | [0.771, 0.871] | 0.911 |
| Dynamic Preference MC contrast | 783 | 0.709 | 228 | 0.294 (67/228) | [0.232, 0.355] | 0.814 |
| Instructional Forgetting free-form (K1c fixed matcher) | 933 | 0.014 | 920 | 0.985 (906/920) | [0.976, 0.992] | 0.989 |

Pre-registered K1d reading:

> within-stale dominates among failures in BOTH IF and DP -> stale-binding is format-robust and pervasive across interference types.

Adjudication: `format_robust_cross_scenario_stale_binding`.
Next step: `stop_k1_api_phase_mechanism_k2_k3_deferred_to_local_gpu`.

## Format Contrast

DP free-form within-stale among failures is 0.821; DP multiple-choice contrast is 0.294. The MC format masks stale-binding by distributing wrong answers across distractors and by presenting the current option explicitly.

## Artifacts

- `k1d_dp_freeform_openrouter.jsonl`
- `k1d_dp_freeform_judge_raw.jsonl`
- `k1d_dp_freeform_rows.jsonl`
- `k1d_dp_freeform_summary.json`
- `k1d_freeform_dp_judge_validation_rows.jsonl`
- `k1d_freeform_dp_judge_validation_summary.json`
