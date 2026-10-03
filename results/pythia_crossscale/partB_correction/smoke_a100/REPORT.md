# Stage O Part B: graded circuit correction

A 160M inference-time existence test, not a deployable or cross-scale repair. LM loss applies the query-position rule at every next-token prediction site.

Identity gate: **PASS**. 
Gamma-zero consistency: not adjudicated on the smoke subset.

| gamma | correction | random 95% | preserve correct | k=0 acc | LM delta | random LM mean | hard controls |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1.0 | 0.000 | [0.000, 0.000] | 1.000 | 1.000 | 0.0000 | 0.0000 | fail |
| 0.0 | 1.000 | [0.000, 0.000] | 1.000 | 1.000 | 0.2894 | 0.0989 | pass |

The operating curve is the result; no gamma is selected post hoc. The LM-loss difference is reported numerically without inventing a materiality threshold.
