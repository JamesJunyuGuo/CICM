# Stage B Report

## Verdict

transport-selectivity confirmed; head-localized. Best late-scope W flip-to-correct rate was 0.789 at alpha=8.0; best top-k flip rate was 0.866. Late dose-response monotone: True.

## Gates and Controls

- Alpha=1 identity gate: True (n=50, mismatches=0).
- Positive-control boosted-value match: 1.0.
- Simple no-op accuracy: 1.0.
- Harm on C at working alpha: 0.0.

## Primary Metrics

| run | set | scope | alpha | target | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 2 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 4 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 8 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 2 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 4 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 8 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | W | late | 2 | current | 194 | 0.4896907216494845 | 0.4896907216494845 | 0.0 | 0.4896907216494845 |
| dose_response | W | late | 4 | current | 194 | 0.6855670103092784 | 0.6855670103092784 | 0.0 | 0.6855670103092784 |
| dose_response | W | late | 8 | current | 194 | 0.788659793814433 | 0.788659793814433 | 0.0 | 0.788659793814433 |
| dose_response | W | topk | 2 | current | 194 | 0.32989690721649484 | 0.32989690721649484 | 0.0 | 0.32989690721649484 |
| dose_response | W | topk | 4 | current | 194 | 0.6649484536082474 | 0.6649484536082474 | 0.0 | 0.6649484536082474 |
| dose_response | W | topk | 8 | current | 194 | 0.865979381443299 | 0.865979381443299 | 0.0 | 0.865979381443299 |
| positive_control_answered_stale | W | late | 4 | answered_stale | 192 | 0.0 | 0.0 | 0.0 | 1.0 |
| sanity_alpha1_identity | sanity | all | 1 | current | 50 | 0.56 | 0.0 | 0.0 | 0.56 |
| scope_comparison | C | all | 4 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | W | all | 4 | current | 194 | 0.7938144329896907 | 0.7938144329896907 | 0.0 | 0.7938144329896907 |
| simple_noop | S | late | 4 | current | 145 | 1.0 | 0.0 | 0.0 | 1.0 |

## Per-I Breakdown

| run | set | scope | alpha | I | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 2 | 2 | 51 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 2 | 4 | 122 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 2 | 8 | 27 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 4 | 2 | 51 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 4 | 4 | 122 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 4 | 8 | 27 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 8 | 2 | 51 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 8 | 4 | 122 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 8 | 8 | 27 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 2 | 2 | 51 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 2 | 4 | 122 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 2 | 8 | 27 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 4 | 2 | 51 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 4 | 4 | 122 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 4 | 8 | 27 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 8 | 2 | 51 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 8 | 4 | 122 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | topk | 8 | 8 | 27 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | W | late | 2 | 2 | 12 | 0.6666666666666666 | 0.6666666666666666 | 0.0 | 0.6666666666666666 |
| dose_response | W | late | 2 | 4 | 122 | 0.5819672131147541 | 0.5819672131147541 | 0.0 | 0.5819672131147541 |
| dose_response | W | late | 2 | 8 | 60 | 0.26666666666666666 | 0.26666666666666666 | 0.0 | 0.26666666666666666 |
| dose_response | W | late | 4 | 2 | 12 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 0.8333333333333334 |
| dose_response | W | late | 4 | 4 | 122 | 0.7622950819672131 | 0.7622950819672131 | 0.0 | 0.7622950819672131 |
| dose_response | W | late | 4 | 8 | 60 | 0.5 | 0.5 | 0.0 | 0.5 |
| dose_response | W | late | 8 | 2 | 12 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 0.8333333333333334 |
| dose_response | W | late | 8 | 4 | 122 | 0.8688524590163934 | 0.8688524590163934 | 0.0 | 0.8688524590163934 |
| dose_response | W | late | 8 | 8 | 60 | 0.6166666666666667 | 0.6166666666666667 | 0.0 | 0.6166666666666667 |
| dose_response | W | topk | 2 | 2 | 12 | 0.5 | 0.5 | 0.0 | 0.5 |
| dose_response | W | topk | 2 | 4 | 122 | 0.36885245901639346 | 0.36885245901639346 | 0.0 | 0.36885245901639346 |
| dose_response | W | topk | 2 | 8 | 60 | 0.21666666666666667 | 0.21666666666666667 | 0.0 | 0.21666666666666667 |
| dose_response | W | topk | 4 | 2 | 12 | 0.75 | 0.75 | 0.0 | 0.75 |
| dose_response | W | topk | 4 | 4 | 122 | 0.7540983606557377 | 0.7540983606557377 | 0.0 | 0.7540983606557377 |
| dose_response | W | topk | 4 | 8 | 60 | 0.4666666666666667 | 0.4666666666666667 | 0.0 | 0.4666666666666667 |
| dose_response | W | topk | 8 | 2 | 12 | 0.9166666666666666 | 0.9166666666666666 | 0.0 | 0.9166666666666666 |
| dose_response | W | topk | 8 | 4 | 122 | 0.9098360655737705 | 0.9098360655737705 | 0.0 | 0.9098360655737705 |
| dose_response | W | topk | 8 | 8 | 60 | 0.7666666666666667 | 0.7666666666666667 | 0.0 | 0.7666666666666667 |
| positive_control_answered_stale | W | late | 4 | 2 | 12 | 0.0 | 0.0 | 0.0 | 1.0 |
| positive_control_answered_stale | W | late | 4 | 4 | 120 | 0.0 | 0.0 | 0.0 | 1.0 |
| positive_control_answered_stale | W | late | 4 | 8 | 60 | 0.0 | 0.0 | 0.0 | 1.0 |
| sanity_alpha1_identity | sanity | all | 1 | 2 | 8 | 0.75 | 0.0 | 0.0 | 0.75 |
| sanity_alpha1_identity | sanity | all | 1 | 4 | 30 | 0.6333333333333333 | 0.0 | 0.0 | 0.6333333333333333 |
| sanity_alpha1_identity | sanity | all | 1 | 8 | 12 | 0.25 | 0.0 | 0.0 | 0.25 |
| scope_comparison | C | all | 4 | 2 | 51 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | C | all | 4 | 4 | 122 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | C | all | 4 | 8 | 27 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | W | all | 4 | 2 | 12 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 0.8333333333333334 |
| scope_comparison | W | all | 4 | 4 | 122 | 0.8442622950819673 | 0.8442622950819673 | 0.0 | 0.8442622950819673 |
| scope_comparison | W | all | 4 | 8 | 60 | 0.6833333333333333 | 0.6833333333333333 | 0.0 | 0.6833333333333333 |
| simple_noop | S | late | 4 | 0 | 145 | 1.0 | 0.0 | 0.0 | 1.0 |
