# R14-gemma2-9b-it-causal-head-routing causal-head discovery: google/gemma-2-9b-it

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=120, validation=120, reserved confirmation=960.

Candidate-token audit: 1.000 (109/109).
First-token collision exclusions (discovery only): 0.

## Causally ranked heads

- L28H8: failure gradient=0.42637, correct gradient=0.05429, target mass=0.28764
- L28H2: failure gradient=0.36541, correct gradient=0.11050, target mass=0.19080
- L28H12: failure gradient=0.30521, correct gradient=0.02889, target mass=0.26890
- L35H1: failure gradient=0.28662, correct gradient=0.15449, target mass=0.26459
- L40H14: failure gradient=0.23035, correct gradient=0.22057, target mass=0.08268
- L32H7: failure gradient=0.21628, correct gradient=0.06946, target mass=0.17163
- L39H8: failure gradient=0.21604, correct gradient=0.15609, target mass=0.16672
- L33H1: failure gradient=0.17096, correct gradient=0.03737, target mass=0.16767
- L35H0: failure gradient=0.16395, correct gradient=0.10704, target mass=0.25229
- L37H12: failure gradient=0.15652, correct gradient=0.06129, target mass=0.11685
- L31H15: failure gradient=0.14980, correct gradient=0.02959, target mass=0.56891
- L36H12: failure gradient=0.13237, correct gradient=0.11714, target mass=0.29921
- L30H1: failure gradient=0.12954, correct gradient=0.03463, target mass=0.22085
- L31H13: failure gradient=0.12254, correct gradient=0.03757, target mass=0.22307
- L32H1: failure gradient=0.11991, correct gradient=0.10771, target mass=0.16553
- L28H4: failure gradient=0.11817, correct gradient=0.00585, target mass=0.12124
- L33H7: failure gradient=0.09130, correct gradient=0.02578, target mass=0.13678
- L30H15: failure gradient=0.08924, correct gradient=0.01907, target mass=0.17250
- L35H14: failure gradient=0.08870, correct gradient=0.08363, target mass=0.23546
- L32H13: failure gradient=0.07875, correct gradient=0.03086, target mass=0.17661
- L34H14: failure gradient=0.06853, correct gradient=0.03809, target mass=0.25438
- L26H0: failure gradient=0.05885, correct gradient=0.00562, target mass=0.03311
- L37H1: failure gradient=0.05685, correct gradient=0.04008, target mass=0.07555
- L26H4: failure gradient=0.05508, correct gradient=0.00567, target mass=0.03017
- L29H1: failure gradient=0.05247, correct gradient=0.03805, target mass=0.02753
- L37H6: failure gradient=0.05244, correct gradient=0.02730, target mass=0.04707
- L28H9: failure gradient=0.05176, correct gradient=0.00925, target mass=0.03954
- L32H15: failure gradient=0.04619, correct gradient=0.03125, target mass=0.21471
- L34H15: failure gradient=0.04042, correct gradient=0.03320, target mass=0.16963
- L25H13: failure gradient=0.03978, correct gradient=0.03742, target mass=0.07539
- L25H7: failure gradient=0.03879, correct gradient=0.00663, target mass=0.01190
- L36H2: failure gradient=0.03698, correct gradient=0.01021, target mass=0.12670

## Validation

Selected top-32 at margin=4.0.

Validation accuracy: 0.500 -> 0.908; gain=0.408; preservation=1.000; within-stale correction=0.902.

Opposite-direction gain: -0.308.
