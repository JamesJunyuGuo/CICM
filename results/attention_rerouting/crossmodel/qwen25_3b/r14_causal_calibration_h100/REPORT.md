# R14-qwen3b-causal-head-routing causal-head discovery: Qwen/Qwen2.5-3B-Instruct

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=120, validation=120, reserved confirmation=960.

Candidate-token audit: 0.990 (102/103).
First-token collision exclusions (discovery only): 6.

## Causally ranked heads

- L26H5: failure gradient=2.23072, correct gradient=0.65355, target mass=0.23261
- L33H9: failure gradient=1.42850, correct gradient=2.24876, target mass=0.30033
- L32H7: failure gradient=1.42145, correct gradient=0.77401, target mass=0.55002
- L31H7: failure gradient=1.34477, correct gradient=0.34955, target mass=0.56761
- L27H1: failure gradient=1.27933, correct gradient=0.28764, target mass=0.39966
- L33H15: failure gradient=1.05365, correct gradient=0.85347, target mass=0.30499
- L29H2: failure gradient=0.97600, correct gradient=0.22802, target mass=0.39736
- L33H0: failure gradient=0.94690, correct gradient=0.70778, target mass=0.26907
- L31H5: failure gradient=0.79766, correct gradient=0.25598, target mass=0.26442
- L33H14: failure gradient=0.70134, correct gradient=0.55554, target mass=0.31567
- L26H10: failure gradient=0.68337, correct gradient=0.12489, target mass=0.17567
- L30H11: failure gradient=0.67398, correct gradient=0.32843, target mass=0.57675
- L29H1: failure gradient=0.66115, correct gradient=0.50897, target mass=0.36937
- L31H3: failure gradient=0.54441, correct gradient=0.26490, target mass=0.28111
- L29H3: failure gradient=0.54411, correct gradient=0.36458, target mass=0.35913
- L32H1: failure gradient=0.51931, correct gradient=0.35108, target mass=0.28134
- L31H0: failure gradient=0.50235, correct gradient=0.16386, target mass=0.27406
- L33H3: failure gradient=0.47591, correct gradient=0.53934, target mass=0.30159
- L26H12: failure gradient=0.42418, correct gradient=0.25434, target mass=0.33486
- L27H4: failure gradient=0.32585, correct gradient=0.02748, target mass=0.07406
- L31H12: failure gradient=0.30303, correct gradient=0.10101, target mass=0.64177
- L26H0: failure gradient=0.27232, correct gradient=0.00892, target mass=0.04241
- L32H3: failure gradient=0.27074, correct gradient=0.02445, target mass=0.58008
- L31H15: failure gradient=0.25293, correct gradient=0.11802, target mass=0.81738
- L24H5: failure gradient=0.23141, correct gradient=0.01881, target mass=0.16071
- L33H10: failure gradient=0.23044, correct gradient=-0.01888, target mass=0.14776
- L32H0: failure gradient=0.22541, correct gradient=0.00971, target mass=0.28053
- L33H12: failure gradient=0.21984, correct gradient=0.38382, target mass=0.16433
- L32H6: failure gradient=0.20927, correct gradient=0.00632, target mass=0.26316
- L32H13: failure gradient=0.20747, correct gradient=0.08696, target mass=0.20439
- L27H5: failure gradient=0.19495, correct gradient=0.05964, target mass=0.13618
- L28H10: failure gradient=0.18939, correct gradient=0.06802, target mass=0.23613

## Validation

Selected top-32 at margin=4.0.

Validation accuracy: 0.175 -> 0.967; gain=0.792; preservation=1.000; within-stale correction=0.976.

Opposite-direction gain: -0.175.
