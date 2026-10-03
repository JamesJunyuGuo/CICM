# R14-qwen3b-causal-head-routing causal-head discovery: Qwen/Qwen2.5-3B-Instruct

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=48, validation=48, reserved confirmation=960.

Candidate-token audit: 1.000 (42/42).
First-token collision exclusions (discovery only): 2.

## Causally ranked heads

- L26H5: failure gradient=2.03089, correct gradient=0.67535, target mass=0.21444
- L31H7: failure gradient=1.35863, correct gradient=0.23780, target mass=0.55100
- L32H7: failure gradient=1.35727, correct gradient=0.58879, target mass=0.53208
- L27H1: failure gradient=1.30842, correct gradient=0.21927, target mass=0.37842
- L33H9: failure gradient=1.21870, correct gradient=0.66612, target mass=0.29409
- L29H2: failure gradient=0.99658, correct gradient=0.18351, target mass=0.35945
- L33H15: failure gradient=0.98182, correct gradient=0.95906, target mass=0.29596
- L33H0: failure gradient=0.79857, correct gradient=-0.01266, target mass=0.26774
- L31H5: failure gradient=0.76896, correct gradient=0.19287, target mass=0.25380
- L33H14: failure gradient=0.71474, correct gradient=0.54013, target mass=0.30195
- L30H11: failure gradient=0.65318, correct gradient=0.23437, target mass=0.57203
- L29H1: failure gradient=0.62504, correct gradient=0.40272, target mass=0.32910
- L26H10: failure gradient=0.61939, correct gradient=0.16507, target mass=0.16698
- L33H3: failure gradient=0.56100, correct gradient=0.95230, target mass=0.29949
- L31H3: failure gradient=0.55436, correct gradient=0.27169, target mass=0.28367
- L32H1: failure gradient=0.50965, correct gradient=0.17513, target mass=0.30193
- L31H0: failure gradient=0.44391, correct gradient=0.02659, target mass=0.28768
- L29H3: failure gradient=0.44317, correct gradient=0.26352, target mass=0.32977
- L26H12: failure gradient=0.42887, correct gradient=0.21977, target mass=0.31160
- L27H4: failure gradient=0.27738, correct gradient=0.05067, target mass=0.06468
- L32H0: failure gradient=0.25728, correct gradient=-0.03973, target mass=0.25884
- L26H0: failure gradient=0.25440, correct gradient=0.00358, target mass=0.03623
- L32H6: failure gradient=0.24076, correct gradient=0.06351, target mass=0.25067
- L24H5: failure gradient=0.23532, correct gradient=0.02068, target mass=0.14775
- L31H15: failure gradient=0.23226, correct gradient=0.14928, target mass=0.79942
- L32H10: failure gradient=0.22961, correct gradient=-0.08358, target mass=0.27327
- L32H13: failure gradient=0.22458, correct gradient=0.06549, target mass=0.20398
- L32H3: failure gradient=0.21445, correct gradient=-0.03558, target mass=0.57153
- L33H4: failure gradient=0.20332, correct gradient=-0.05469, target mass=0.15272
- L31H12: failure gradient=0.20160, correct gradient=0.28879, target mass=0.64520
- L29H5: failure gradient=0.18991, correct gradient=0.03538, target mass=0.22373
- L27H5: failure gradient=0.17624, correct gradient=0.03718, target mass=0.12436

## Validation

Selected top-32 at margin=4.0.

Validation accuracy: 0.208 -> 0.938; gain=0.729; preservation=1.000; within-stale correction=0.935.

Opposite-direction gain: -0.208.
