# Stage O Part B: graded circuit correction

A 160M inference-time existence test, not a deployable or cross-scale repair. LM loss applies the query-position rule at every next-token prediction site.

Identity gate: **PASS**. 
Gamma-zero consistency: **PASS** (0.384 vs Phase-1b 0.384).

| gamma | correction | random 95% | preserve correct | k=0 acc | LM delta | random LM mean | hard controls |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1.0 | 0.000 | [0.000, 0.000] | 1.000 | 0.989 | 0.0000 | 0.0000 | fail |
| 0.9 | 0.027 | [0.000, 0.041] | 1.000 | 0.994 | 0.0042 | 0.0048 | fail |
| 0.7 | 0.082 | [0.000, 0.129] | 0.988 | 1.000 | 0.0223 | 0.0232 | fail |
| 0.5 | 0.164 | [0.008, 0.162] | 0.951 | 0.994 | 0.0558 | 0.0656 | pass |
| 0.3 | 0.219 | [0.000, 0.196] | 0.926 | 0.994 | 0.1097 | 0.1400 | fail |
| 0.1 | 0.301 | [0.000, 0.258] | 0.840 | 0.994 | 0.1980 | 0.2237 | fail |
| 0.0 | 0.384 | [0.000, 0.258] | 0.815 | 0.992 | 0.2512 | 0.2718 | fail |

The operating curve is the result; no gamma is selected post hoc. The LM-loss difference is reported numerically without inventing a materiality threshold.
