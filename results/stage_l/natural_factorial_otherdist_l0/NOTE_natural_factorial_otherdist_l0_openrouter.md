# Stage L L0 Other-Distance Factorial OpenRouter Summary

## Scope

- Dataset: `data/stage_l/cicm_natural_factorial_otherdist_l0.jsonl`
- Model/API: OpenRouter `qwen/qwen-2.5-7b-instruct`, temperature 0, max_tokens 16
- Rows: 1200; target-current fixed far; same-slot stale distance in {far, near}; recent other-slot distance in {far, mid, near2}
- This is behavior only. No hard2 run, no GPU, no L-1 mechanism.

## Overall

- Counts: `{'correct_current': 471, 'within_stale': 589, 'cross_slot': 76, 'other': 64}`
- Accuracy: 39.2%; failure rate: 60.8%
- Failure mix: within-stale 80.8%, cross-slot 10.4%, other 8.8%

## Factorial Cell

| same-slot stale | other-slot distance | n | accuracy | within-stale | cross-slot | other |
|---|---:|---:|---:|---:|---:|---:|
| far | far | 200 | 74.5% | 17.0% | 8.5% | 0.0% |
| far | mid | 200 | 61.0% | 8.0% | 11.5% | 19.5% |
| far | near2 | 200 | 62.5% | 16.0% | 16.5% | 5.0% |
| near | far | 200 | 5.5% | 94.5% | 0.0% | 0.0% |
| near | mid | 200 | 16.0% | 83.5% | 0.0% | 0.5% |
| near | near2 | 200 | 16.0% | 75.5% | 1.5% | 7.0% |

## Collapsed Effects

| factor | level | n | accuracy | within-stale | cross-slot | failure rate |
|---|---:|---:|---:|---:|---:|---:|
| other-slot distance | far | 400 | 40.0% | 55.8% | 4.2% | 60.0% |
| other-slot distance | mid | 400 | 38.5% | 45.8% | 5.8% | 61.5% |
| other-slot distance | near2 | 400 | 39.2% | 45.8% | 9.0% | 60.8% |
| same-slot stale distance | far | 600 | 66.0% | 13.7% | 12.2% | 34.0% |
| same-slot stale distance | near | 600 | 12.5% | 84.5% | 0.5% | 87.5% |

## Dose

| k overwrites | n | accuracy | within-stale | cross-slot | failure rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 240 | 47.1% | 44.6% | 3.8% | 52.9% |
| 2 | 240 | 45.4% | 45.0% | 4.6% | 54.6% |
| 3 | 240 | 40.0% | 48.3% | 3.8% | 60.0% |
| 4 | 240 | 32.1% | 53.3% | 8.3% | 67.9% |
| 6 | 240 | 31.7% | 54.2% | 11.2% | 68.3% |

## Crossover Read

- Collapsed cross-slot rates far -> mid -> near2: far=4.2%, mid=5.8%, near2=9.0%.
- Collapsed within-stale rates far -> mid -> near2: far=55.8%, mid=45.8%, near2=45.8%.
- Collapsed over same-slot distance, cross-slot does not exceed within-stale at any other-distance level; the observed result is a recency-gradient, not a full dominance crossover.
- Conditioned on `same-slot stale=far`, identity-to-recency crossover starts at `other-slot distance=mid`: cross-slot=11.5% vs within-stale=8.0%, and remains essentially tied/slightly above at `near2`: cross-slot=16.5% vs within-stale=16.0%.
- Conditioned on `same-slot stale=near`, same-slot stale dominates in every other-distance level: within-stale=94.5%/83.5%/75.5% for far/mid/near2, while cross-slot is 0.0%/0.0%/1.5%.
- Interpretation to review: the run gives a large within-stale pool and a measurable identity-to-recency component, but same-slot stale retrieval remains dominant under this exact hard factorial setting.

## Artifacts

- Merged rows: `results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_rows_merged.jsonl`
- Merged summary JSON: `results/stage_l/natural_factorial_otherdist_l0/openrouter_qwen25_7b_summary_merged.json`
- Chunk raw outputs: `results/stage_l/natural_factorial_otherdist_l0/chunks/output_*.jsonl`
- Retry raw output: `results/stage_l/natural_factorial_otherdist_l0/retry_call_errors_001_output.jsonl`

## Stop Point

Stop here for PI review. Do not launch L-1/GPU until this behavior summary is approved.
