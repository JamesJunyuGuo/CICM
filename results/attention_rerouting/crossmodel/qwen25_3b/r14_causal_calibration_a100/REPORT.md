# R14-qwen3b-causal-head-routing causal-head discovery: Qwen/Qwen2.5-3B-Instruct

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=120, validation=120, reserved confirmation=960.

Candidate-token audit: 0.990 (102/103).
First-token collision exclusions (discovery only): 6.

## Causally ranked heads

- L26H5: failure gradient=2.19837, correct gradient=0.65511, target mass=0.23256
- L33H9: failure gradient=1.44449, correct gradient=2.22232, target mass=0.30015
- L32H7: failure gradient=1.41671, correct gradient=0.76859, target mass=0.54980
- L31H7: failure gradient=1.33645, correct gradient=0.36493, target mass=0.56660
- L27H1: failure gradient=1.28053, correct gradient=0.28958, target mass=0.40163
- L33H15: failure gradient=1.06163, correct gradient=0.85172, target mass=0.30470
- L29H2: failure gradient=0.98264, correct gradient=0.22995, target mass=0.39809
- L33H0: failure gradient=0.94165, correct gradient=0.70502, target mass=0.26954
- L31H5: failure gradient=0.80891, correct gradient=0.26048, target mass=0.26816
- L33H14: failure gradient=0.70372, correct gradient=0.54315, target mass=0.31553
- L26H10: failure gradient=0.69980, correct gradient=0.12389, target mass=0.17613
- L30H11: failure gradient=0.67057, correct gradient=0.33453, target mass=0.57864
- L29H1: failure gradient=0.66033, correct gradient=0.52715, target mass=0.37026
- L31H3: failure gradient=0.54638, correct gradient=0.26696, target mass=0.28061
- L29H3: failure gradient=0.53150, correct gradient=0.36945, target mass=0.35858
- L32H1: failure gradient=0.53140, correct gradient=0.34228, target mass=0.28260
- L31H0: failure gradient=0.49524, correct gradient=0.16712, target mass=0.27255
- L33H3: failure gradient=0.47205, correct gradient=0.56091, target mass=0.30161
- L26H12: failure gradient=0.42829, correct gradient=0.25648, target mass=0.33489
- L27H4: failure gradient=0.34069, correct gradient=0.03079, target mass=0.07543
- L31H12: failure gradient=0.30354, correct gradient=0.10629, target mass=0.63747
- L32H3: failure gradient=0.27139, correct gradient=0.00008, target mass=0.58210
- L26H0: failure gradient=0.26411, correct gradient=0.00880, target mass=0.04211
- L31H15: failure gradient=0.24784, correct gradient=0.12153, target mass=0.81155
- L33H10: failure gradient=0.22991, correct gradient=-0.01804, target mass=0.14757
- L24H5: failure gradient=0.22922, correct gradient=0.01863, target mass=0.16426
- L32H0: failure gradient=0.22547, correct gradient=0.01150, target mass=0.28152
- L32H6: failure gradient=0.21585, correct gradient=0.00567, target mass=0.26482
- L33H12: failure gradient=0.21580, correct gradient=0.35607, target mass=0.16595
- L32H13: failure gradient=0.20931, correct gradient=0.09058, target mass=0.20585
- L27H5: failure gradient=0.20102, correct gradient=0.06174, target mass=0.13917
- L33H11: failure gradient=0.19022, correct gradient=0.00615, target mass=0.19570

## Validation

Selected top-32 at margin=4.0.

Validation accuracy: 0.167 -> 0.967; gain=0.800; preservation=1.000; within-stale correction=0.976.

Opposite-direction gain: -0.167.
