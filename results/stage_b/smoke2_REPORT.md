# Stage B Report

## Verdict

mixed or inconclusive; layer-diffuse. Best late-scope W flip-to-correct rate was 0.200 at alpha=4.0; best top-k flip rate was 0.100. Late dose-response monotone: False.

## Gates and Controls

- Alpha=1 identity gate: True (n=10, mismatches=0).
- Positive-control boosted-value match: 0.5555555555555556.
- Simple no-op accuracy: 0.8.
- Harm on C at working alpha: 0.3.

## Primary Metrics

| run | set | scope | alpha | target | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 1 | current | 10 | 0.2 | 0.0 | 0.8 | 0.2 |
| dose_response | C | late | 4 | current | 10 | 0.7 | 0.0 | 0.3 | 0.7 |
| dose_response | C | topk | 1 | current | 10 | 0.2 | 0.0 | 0.8 | 0.2 |
| dose_response | C | topk | 4 | current | 10 | 0.2 | 0.0 | 0.8 | 0.2 |
| dose_response | W | late | 1 | current | 10 | 0.1 | 0.1 | 0.0 | 0.1 |
| dose_response | W | late | 4 | current | 10 | 0.2 | 0.2 | 0.0 | 0.2 |
| dose_response | W | topk | 1 | current | 10 | 0.1 | 0.1 | 0.0 | 0.1 |
| dose_response | W | topk | 4 | current | 10 | 0.1 | 0.1 | 0.0 | 0.1 |
| positive_control_answered_stale | W | late | 4 | answered_stale | 9 | 0.0 | 0.0 | 0.0 | 0.5555555555555556 |
| sanity_alpha1_identity | sanity | all | 1 | current | 10 | 0.2 | 0.0 | 0.0 | 0.2 |
| scope_comparison | C | all | 4 | current | 10 | 0.7 | 0.0 | 0.3 | 0.7 |
| scope_comparison | W | all | 4 | current | 10 | 0.3 | 0.3 | 0.0 | 0.3 |
| simple_noop | S | late | 4 | current | 10 | 0.8 | 0.0 | 0.0 | 0.8 |

## Per-I Breakdown

| run | set | scope | alpha | I | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 1 | 4 | 10 | 0.2 | 0.0 | 0.8 | 0.2 |
| dose_response | C | late | 4 | 4 | 10 | 0.7 | 0.0 | 0.3 | 0.7 |
| dose_response | C | topk | 1 | 4 | 10 | 0.2 | 0.0 | 0.8 | 0.2 |
| dose_response | C | topk | 4 | 4 | 10 | 0.2 | 0.0 | 0.8 | 0.2 |
| dose_response | W | late | 1 | 4 | 10 | 0.1 | 0.1 | 0.0 | 0.1 |
| dose_response | W | late | 4 | 4 | 10 | 0.2 | 0.2 | 0.0 | 0.2 |
| dose_response | W | topk | 1 | 4 | 10 | 0.1 | 0.1 | 0.0 | 0.1 |
| dose_response | W | topk | 4 | 4 | 10 | 0.1 | 0.1 | 0.0 | 0.1 |
| positive_control_answered_stale | W | late | 4 | 4 | 9 | 0.0 | 0.0 | 0.0 | 0.5555555555555556 |
| sanity_alpha1_identity | sanity | all | 1 | 2 | 2 | 0.5 | 0.0 | 0.0 | 0.5 |
| sanity_alpha1_identity | sanity | all | 1 | 4 | 8 | 0.125 | 0.0 | 0.0 | 0.125 |
| scope_comparison | C | all | 4 | 4 | 10 | 0.7 | 0.0 | 0.3 | 0.7 |
| scope_comparison | W | all | 4 | 4 | 10 | 0.3 | 0.3 | 0.0 | 0.3 |
| simple_noop | S | late | 4 | 0 | 10 | 0.8 | 0.0 | 0.0 | 0.8 |
