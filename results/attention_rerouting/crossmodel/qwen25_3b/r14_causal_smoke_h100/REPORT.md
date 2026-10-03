# R14-qwen3b-causal-head-routing causal-head discovery: Qwen/Qwen2.5-3B-Instruct

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=48, validation=48, reserved confirmation=960.

Candidate-token audit: 0.977 (42/43).
First-token collision exclusions (discovery only): 2.

## Causally ranked heads

- L26H5: failure gradient=2.02147, correct gradient=0.63713, target mass=0.21151
- L31H7: failure gradient=1.34606, correct gradient=0.25626, target mass=0.54384
- L32H7: failure gradient=1.30577, correct gradient=0.60087, target mass=0.52502
- L27H1: failure gradient=1.29152, correct gradient=0.16521, target mass=0.36562
- L33H9: failure gradient=1.16595, correct gradient=0.75174, target mass=0.29496
- L33H15: failure gradient=1.01292, correct gradient=0.89949, target mass=0.29949
- L29H2: failure gradient=0.99086, correct gradient=0.18874, target mass=0.36055
- L33H0: failure gradient=0.79185, correct gradient=-0.01394, target mass=0.26209
- L33H14: failure gradient=0.77216, correct gradient=0.51212, target mass=0.29886
- L31H5: failure gradient=0.74187, correct gradient=0.19656, target mass=0.24709
- L30H11: failure gradient=0.65640, correct gradient=0.22194, target mass=0.56687
- L29H1: failure gradient=0.62363, correct gradient=0.36540, target mass=0.32758
- L26H10: failure gradient=0.60152, correct gradient=0.16195, target mass=0.16407
- L33H3: failure gradient=0.56364, correct gradient=0.89133, target mass=0.30055
- L31H3: failure gradient=0.54938, correct gradient=0.26950, target mass=0.28267
- L32H1: failure gradient=0.48250, correct gradient=0.18497, target mass=0.29550
- L29H3: failure gradient=0.46329, correct gradient=0.27606, target mass=0.33342
- L31H0: failure gradient=0.44350, correct gradient=0.02636, target mass=0.28417
- L26H12: failure gradient=0.42143, correct gradient=0.19811, target mass=0.30995
- L32H6: failure gradient=0.27761, correct gradient=0.07662, target mass=0.24933
- L26H0: failure gradient=0.25874, correct gradient=0.00358, target mass=0.03561
- L27H4: failure gradient=0.25658, correct gradient=0.03954, target mass=0.06160
- L32H0: failure gradient=0.24961, correct gradient=-0.04376, target mass=0.25187
- L32H13: failure gradient=0.23239, correct gradient=0.06782, target mass=0.20504
- L24H5: failure gradient=0.23179, correct gradient=0.02102, target mass=0.14246
- L31H15: failure gradient=0.22884, correct gradient=0.14951, target mass=0.80247
- L32H3: failure gradient=0.21949, correct gradient=-0.02772, target mass=0.56379
- L32H10: failure gradient=0.21166, correct gradient=-0.08184, target mass=0.26382
- L31H12: failure gradient=0.20886, correct gradient=0.26609, target mass=0.64315
- L33H4: failure gradient=0.20492, correct gradient=-0.05755, target mass=0.15059
- L30H14: failure gradient=0.18265, correct gradient=0.19163, target mass=0.45897
- L29H5: failure gradient=0.17797, correct gradient=0.03579, target mass=0.22113

## Validation

Selected top-32 at margin=4.0.

Validation accuracy: 0.208 -> 0.938; gain=0.729; preservation=1.000; within-stale correction=0.933.

Opposite-direction gain: -0.208.
