# Stage N cross-scale descriptive mechanism transfer

## Preregistered scope

Descriptive transfer of selection, QK displacement, and stale-relative attention; not conservation of head identity or of the 160M minimal causal circuit.

All models use the byte-identical 7,776-example task and fixed `k=4`. Mechanism analysis is run only after the frozen matched-pool gate passes.

## Headline

The abstract retention-versus-selection description transfers, but the localized 160M QK-attention coupling does not. At every identifiable scale, the current value remains decodable on stale failures above a label-shuffle null, although its score is lower than on correct trials. The exact same-head conjunction weakens from four heads at 160M to zero at 410M and 2.8B.

## Results

### Behavior across overwrite count

| Scale | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---:|---:|---:|---:|---:|---:|
| 160m | 0.715 | 0.549 | 0.492 | 0.474 | 0.414 | 0.412 |
| 410m | 0.646 | 0.570 | 0.464 | 0.429 | 0.399 | 0.508 |
| 1b | 0.590 | 0.610 | 0.575 | 0.581 | 0.564 | 0.593 |
| 2.8b | 0.771 | 0.816 | 0.721 | 0.627 | 0.516 | 0.504 |

Cells are exact-match current-value accuracy on all rows at each overwrite count. The fixed-`k=4` gate below uses only the clean single-variable arm.

### Failure-conditioned mechanism

| Scale | k=4 correct | k=4 stale | Pool gate | Probe score | Probe null upper | QK heads | Attention heads | Same-head | Transfer |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 160m | 466 | 142 | yes | 0.400 | 0.083 | 4 | 8 | 4 | yes |
| 410m | 453 | 194 | yes | 0.273 | 0.073 | 1 | 5 | 0 | no |
| 1b | 593 | 55 | no | -- | -- | -- | -- | -- | -- |
| 2.8b | 438 | 210 | yes | 0.311 | 0.075 | 0 | 1 | 0 | no |

QK and attention head counts use pooled log-length residuals and 2,000 within-(template, seed) permutations with model-wide family-wise extrema.

### Selection probe

| Scale | Failure score | Shuffle 95% | Failure-correct raw | Failure-correct controlled [95% CI] |
|---|---:|---:|---:|---:|
| 160m | 0.400 | [0.012, 0.083] | -0.321 | -0.320 [-0.404, -0.237] |
| 410m | 0.273 | [0.012, 0.073] | -0.211 | -0.211 [-0.278, -0.138] |
| 1b | -- | -- | -- | -- |
| 2.8b | 0.311 | [0.013, 0.075] | -0.108 | -0.108 [-0.185, -0.032] |

The positive above-null score rejects complete representational loss. The negative failure-minus-correct difference shows concurrent retention degradation, so the result is present-but-weakened rather than a claim of perfectly intact retention.

### Canonical 160M sanity

The new pipeline reproduces the frozen Phase-1 failure score (0.400472 vs 0.400342; absolute difference 0.000130) and the length-controlled failure-minus-correct delta (-0.319927 vs -0.320016; absolute difference 0.000089).

## Interpretation

- **160m:** full preregistered transfer: probe, QK, attention, and same-head conjunction all pass.
- **410m:** partial transfer: the probe passes; 1 QK and 5 attention heads pass separately, but 0 pass their conjunction.
- **1b:** the fixed behavior gate failed, so failure-conditioned mechanism transfer is not identifiable under this task.
- **2.8b:** partial transfer: the probe passes; 0 QK and 1 attention heads pass separately, but 0 pass their conjunction.

A pass does not show that numerical head indices or a minimal causal circuit are preserved across scales. A failure does not show that no stale-binding mechanism exists outside this frozen task.

The 1B exception also rules out a monotonic size law on this task: its fixed-`k=4` stale pool is too small and template-imbalanced for the primary mechanism test, while 2.8B again produces a large stale-dominant pool. No post-hoc difficulty or gate tuning was used to remove that non-monotonicity.

For connecting the 160M circuit result to larger instruction-tuned models, the supported bridge is therefore the selection-level diagnosis, not conservation of a particular small-model head set or QK route.
