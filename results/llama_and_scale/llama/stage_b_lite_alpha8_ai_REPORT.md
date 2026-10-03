# Stage B Report

## Verdict

mixed or inconclusive; layer-diffuse. Best late-scope W flip-to-correct rate was 0.608 at alpha=8.0; best top-k flip rate was 0.068. Late dose-response monotone: False.

## Gates and Controls

- Alpha=1 identity gate: True (n=50, mismatches=0).
- Positive-control boosted-value match: 0.9979253112033195.
- Simple no-op accuracy: 1.0.
- Harm on C at working alpha: 0.025.

## Primary Metrics

| run | set | scope | alpha | target | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 8 | current | 200 | 0.975 | 0.0 | 0.025 | 0.975 |
| dose_response | C | topk | 8 | current | 200 | 0.94 | 0.0 | 0.06 | 0.94 |
| dose_response | W | late | 8 | current | 482 | 0.6078838174273858 | 0.6078838174273858 | 0.0 | 0.6078838174273858 |
| dose_response | W | topk | 8 | current | 482 | 0.06846473029045644 | 0.06846473029045644 | 0.0 | 0.06846473029045644 |
| positive_control_answered_stale | W | late | 4 | answered_stale | 482 | 0.0 | 0.0 | 0.0 | 0.9979253112033195 |
| sanity_alpha1_identity | sanity | all | 1 | current | 50 | 0.28 | 0.0 | 0.0 | 0.28 |
| scope_comparison | C | all | 4 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | W | all | 4 | current | 482 | 0.8713692946058091 | 0.8713692946058091 | 0.0 | 0.8713692946058091 |
| simple_noop | S | late | 4 | current | 138 | 1.0 | 0.0 | 0.0 | 1.0 |

## Per-I Breakdown

| run | set | scope | alpha | I | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 8 | 2 | 56 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 8 | 4 | 104 | 0.9711538461538461 | 0.0 | 0.028846153846153848 | 0.9711538461538461 |
| dose_response | C | late | 8 | 8 | 40 | 0.95 | 0.0 | 0.05 | 0.95 |
| dose_response | C | topk | 8 | 2 | 56 | 0.9107142857142857 | 0.0 | 0.08928571428571429 | 0.9107142857142857 |
| dose_response | C | topk | 8 | 4 | 104 | 0.9423076923076923 | 0.0 | 0.057692307692307696 | 0.9423076923076923 |
| dose_response | C | topk | 8 | 8 | 40 | 0.975 | 0.0 | 0.025 | 0.975 |
| dose_response | W | late | 8 | 2 | 75 | 0.6533333333333333 | 0.6533333333333333 | 0.0 | 0.6533333333333333 |
| dose_response | W | late | 8 | 4 | 316 | 0.5949367088607594 | 0.5949367088607594 | 0.0 | 0.5949367088607594 |
| dose_response | W | late | 8 | 8 | 91 | 0.6153846153846154 | 0.6153846153846154 | 0.0 | 0.6153846153846154 |
| dose_response | W | topk | 8 | 2 | 75 | 0.04 | 0.04 | 0.0 | 0.04 |
| dose_response | W | topk | 8 | 4 | 316 | 0.04746835443037975 | 0.04746835443037975 | 0.0 | 0.04746835443037975 |
| dose_response | W | topk | 8 | 8 | 91 | 0.16483516483516483 | 0.16483516483516483 | 0.0 | 0.16483516483516483 |
| positive_control_answered_stale | W | late | 4 | 2 | 75 | 0.0 | 0.0 | 0.0 | 1.0 |
| positive_control_answered_stale | W | late | 4 | 4 | 316 | 0.0 | 0.0 | 0.0 | 0.9968354430379747 |
| positive_control_answered_stale | W | late | 4 | 8 | 91 | 0.0 | 0.0 | 0.0 | 1.0 |
| sanity_alpha1_identity | sanity | all | 1 | 2 | 10 | 0.5 | 0.0 | 0.0 | 0.5 |
| sanity_alpha1_identity | sanity | all | 1 | 4 | 32 | 0.1875 | 0.0 | 0.0 | 0.1875 |
| sanity_alpha1_identity | sanity | all | 1 | 8 | 8 | 0.375 | 0.0 | 0.0 | 0.375 |
| scope_comparison | C | all | 4 | 2 | 56 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | C | all | 4 | 4 | 104 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | C | all | 4 | 8 | 40 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | W | all | 4 | 2 | 75 | 0.9066666666666666 | 0.9066666666666666 | 0.0 | 0.9066666666666666 |
| scope_comparison | W | all | 4 | 4 | 316 | 0.8607594936708861 | 0.8607594936708861 | 0.0 | 0.8607594936708861 |
| scope_comparison | W | all | 4 | 8 | 91 | 0.8791208791208791 | 0.8791208791208791 | 0.0 | 0.8791208791208791 |
| simple_noop | S | late | 4 | 0 | 138 | 1.0 | 0.0 | 0.0 | 1.0 |
