# Free-energy scaling validation

## Analysis protocol

Fit `logit P(current | current or stale) = alpha - log(k)` on the lower-dose cells and evaluate it on frozen higher-dose cells that are excluded from parameter fitting. A dataset passes exact scaling only when every held-out rate lies inside its 95% parametric-bootstrap predictive interval. Training pools with fewer than five successes or failures are marked ceiling/floor-limited. A free exponent `gamma` is diagnostic, not a replacement headline.

## Results

| Dataset | Train k | Held-out k | alpha | gamma (free) | Held-out MAE | Status |
|---|---:|---:|---:|---:|---:|---|
| controlled_qwen7b_20lines | 2,4 | 8 | 2.369 | 2.38 | 0.057 | heldout_pass |
| controlled_qwen7b_40lines | 2,4 | 8 | 2.141 | 1.67 | 0.065 | heldout_pass |
| controlled_qwen7b_80lines | 2,4 | 8 | 1.610 | 0.82 | 0.001 | heldout_pass |
| scale_llama-3_1-70b-instruct | 2,4,8 | 12,16 | 5.841 | 2.77 | 0.146 | heldout_fail |
| scale_llama-3_1-8b-instruct | 2,4,8 | 12,16 | 0.286 | 0.99 | 0.038 | heldout_pass |
| scale_gpt-4o | 2,4,8 | 12,16 | 7.647 | 17.10 | 0.007 | ceiling_or_floor_limited |
| scale_qwen-2_5-72b-instruct | 2,4,8 | 12,16 | 7.246 | -0.00 | 0.001 | ceiling_or_floor_limited |
| scale_qwen-2_5-7b-instruct | 2,4,8 | 12,16 | 2.531 | 1.51 | 0.096 | heldout_pass |
| cicm_natural_qwen7b | 1,2,3 | 4,6 | 0.551 | 0.20 | 0.109 | heldout_fail |

### Held-out cells

| Dataset | k | Observed | Predicted | 95% predictive interval | In interval |
|---|---:|---:|---:|---:|:---:|
| controlled_qwen7b_20lines | 8 | 0.515 | 0.572 | [0.444, 0.697] | yes |
| controlled_qwen7b_40lines | 8 | 0.450 | 0.515 | [0.390, 0.650] | yes |
| controlled_qwen7b_80lines | 8 | 0.384 | 0.385 | [0.273, 0.505] | yes |
| scale_llama-3_1-70b-instruct | 12 | 0.860 | 0.966 | [0.920, 1.000] | no |
| scale_llama-3_1-70b-instruct | 16 | 0.770 | 0.956 | [0.900, 1.000] | no |
| scale_llama-3_1-8b-instruct | 12 | 0.153 | 0.100 | [0.041, 0.173] | yes |
| scale_llama-3_1-8b-instruct | 16 | 0.100 | 0.077 | [0.030, 0.140] | yes |
| scale_gpt-4o | 12 | 1.000 | 0.994 | [0.970, 1.000] | yes |
| scale_gpt-4o | 16 | 1.000 | 0.992 | [0.960, 1.000] | yes |
| scale_qwen-2_5-72b-instruct | 12 | 0.990 | 0.992 | [0.960, 1.000] | yes |
| scale_qwen-2_5-72b-instruct | 16 | 0.990 | 0.989 | [0.950, 1.000] | yes |
| scale_qwen-2_5-7b-instruct | 12 | 0.400 | 0.512 | [0.400, 0.630] | yes |
| scale_qwen-2_5-7b-instruct | 16 | 0.360 | 0.440 | [0.330, 0.550] | yes |
| cicm_natural_qwen7b | 4 | 0.376 | 0.302 | [0.234, 0.376] | yes |
| cicm_natural_qwen7b | 6 | 0.369 | 0.224 | [0.165, 0.291] | no |

## Reading

Exact exchangeable scaling passes 5 testable datasets, fails 2, and leaves 2 ceiling/floor-limited. A pass supports a fixed-margin multiplicity account in that regime. A failure means competitor scores or the current margin change with dose/context; it does not negate the free-energy competition identity.

The validation is deliberately conditional on current-versus-within-stale outcomes. Cross-slot and other errors are excluded because they belong to different candidate partitions.
