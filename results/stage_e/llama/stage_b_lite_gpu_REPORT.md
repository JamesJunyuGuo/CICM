# Stage B Report

## Verdict

mixed or inconclusive; layer-diffuse. Best late-scope W flip-to-correct rate was 0.465 at alpha=4.0; best top-k flip rate was 0.035. Late dose-response monotone: False.

## Gates and Controls

- Alpha=1 identity gate: True (n=50, mismatches=0).
- Positive-control boosted-value match: 0.9979253112033195.
- Simple no-op accuracy: 1.0.
- Harm on C at working alpha: 0.005.

## Primary Metrics

| run | set | scope | alpha | target | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 4 | current | 200 | 0.995 | 0.0 | 0.005 | 0.995 |
| dose_response | C | topk | 4 | current | 200 | 0.965 | 0.0 | 0.035 | 0.965 |
| dose_response | W | late | 4 | current | 482 | 0.46473029045643155 | 0.46473029045643155 | 0.0 | 0.46473029045643155 |
| dose_response | W | topk | 4 | current | 482 | 0.035269709543568464 | 0.035269709543568464 | 0.0 | 0.035269709543568464 |
| positive_control_answered_stale | W | late | 4 | answered_stale | 482 | 0.0 | 0.0 | 0.0 | 0.9979253112033195 |
| sanity_alpha1_identity | sanity | all | 1 | current | 50 | 0.28 | 0.0 | 0.0 | 0.28 |
| scope_comparison | C | all | 4 | current | 200 | 1.0 | 0.0 | 0.0 | 1.0 |
| scope_comparison | W | all | 4 | current | 482 | 0.8713692946058091 | 0.8713692946058091 | 0.0 | 0.8713692946058091 |
| simple_noop | S | late | 4 | current | 138 | 1.0 | 0.0 | 0.0 | 1.0 |

## Per-I Breakdown

| run | set | scope | alpha | I | n_valid | acc | flip | harm | boosted_match |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dose_response | C | late | 4 | 2 | 56 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 4 | 4 | 104 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | C | late | 4 | 8 | 40 | 0.975 | 0.0 | 0.025 | 0.975 |
| dose_response | C | topk | 4 | 2 | 56 | 0.9642857142857143 | 0.0 | 0.03571428571428571 | 0.9642857142857143 |
| dose_response | C | topk | 4 | 4 | 104 | 0.9519230769230769 | 0.0 | 0.04807692307692308 | 0.9519230769230769 |
| dose_response | C | topk | 4 | 8 | 40 | 1.0 | 0.0 | 0.0 | 1.0 |
| dose_response | W | late | 4 | 2 | 75 | 0.5333333333333333 | 0.5333333333333333 | 0.0 | 0.5333333333333333 |
| dose_response | W | late | 4 | 4 | 316 | 0.4525316455696203 | 0.4525316455696203 | 0.0 | 0.4525316455696203 |
| dose_response | W | late | 4 | 8 | 91 | 0.45054945054945056 | 0.45054945054945056 | 0.0 | 0.45054945054945056 |
| dose_response | W | topk | 4 | 2 | 75 | 0.0 | 0.0 | 0.0 | 0.0 |
| dose_response | W | topk | 4 | 4 | 316 | 0.03481012658227848 | 0.03481012658227848 | 0.0 | 0.03481012658227848 |
| dose_response | W | topk | 4 | 8 | 91 | 0.06593406593406594 | 0.06593406593406594 | 0.0 | 0.06593406593406594 |
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
