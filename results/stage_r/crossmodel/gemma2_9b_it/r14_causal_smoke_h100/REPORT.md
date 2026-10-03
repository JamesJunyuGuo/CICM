# R14-gemma2-9b-it-causal-head-routing causal-head discovery: google/gemma-2-9b-it

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=12, validation=12, reserved confirmation=960.

Candidate-token audit: 1.000 (12/12).
First-token collision exclusions (discovery only): 0.

## Causally ranked heads

- L28H8: failure gradient=0.38697, correct gradient=0.05720, target mass=0.28458
- L35H1: failure gradient=0.34886, correct gradient=0.04128, target mass=0.26841
- L28H2: failure gradient=0.31008, correct gradient=0.05980, target mass=0.16933
- L40H14: failure gradient=0.22486, correct gradient=0.11177, target mass=0.08601
- L32H7: failure gradient=0.22391, correct gradient=0.02584, target mass=0.17505
- L34H14: failure gradient=0.21539, correct gradient=0.16396, target mass=0.31709
- L28H12: failure gradient=0.21479, correct gradient=0.01813, target mass=0.19214
- L33H1: failure gradient=0.19321, correct gradient=0.23562, target mass=0.16746
- L39H8: failure gradient=0.19236, correct gradient=0.24357, target mass=0.17036
- L31H15: failure gradient=0.17129, correct gradient=0.01361, target mass=0.48820
- L37H12: failure gradient=0.14529, correct gradient=0.03207, target mass=0.11333
- L36H12: failure gradient=0.13623, correct gradient=0.02545, target mass=0.33089
- L35H0: failure gradient=0.12766, correct gradient=0.08772, target mass=0.24285
- L32H1: failure gradient=0.11898, correct gradient=0.10830, target mass=0.16928
- L30H1: failure gradient=0.10359, correct gradient=0.00952, target mass=0.23056
- L31H13: failure gradient=0.10136, correct gradient=0.01184, target mass=0.19861
- L28H4: failure gradient=0.09195, correct gradient=0.00588, target mass=0.09656
- L33H7: failure gradient=0.08871, correct gradient=0.01082, target mass=0.13909
- L35H14: failure gradient=0.08736, correct gradient=0.01991, target mass=0.22596
- L30H9: failure gradient=0.08686, correct gradient=0.03646, target mass=0.48549
- L30H15: failure gradient=0.07714, correct gradient=0.00382, target mass=0.16952
- L37H6: failure gradient=0.07674, correct gradient=0.06687, target mass=0.05977
- L32H13: failure gradient=0.07513, correct gradient=0.05820, target mass=0.15176
- L26H0: failure gradient=0.05180, correct gradient=0.00020, target mass=0.02854
- L25H13: failure gradient=0.04653, correct gradient=0.03281, target mass=0.07628
- L36H6: failure gradient=0.04316, correct gradient=-0.01098, target mass=0.19296
- L34H6: failure gradient=0.04073, correct gradient=-0.00140, target mass=0.37309
- L36H5: failure gradient=0.03934, correct gradient=0.00664, target mass=0.07777
- L35H12: failure gradient=0.03462, correct gradient=0.00075, target mass=0.06799
- L36H2: failure gradient=0.03430, correct gradient=0.00878, target mass=0.12216
- L25H7: failure gradient=0.03294, correct gradient=0.00058, target mass=0.00733
- L26H4: failure gradient=0.03232, correct gradient=0.00064, target mass=0.01734

## Validation

Selected top-32 at margin=4.0.

Validation accuracy: 0.417 -> 0.833; gain=0.417; preservation=1.000; within-stale correction=0.833.

Opposite-direction gain: -0.250.
