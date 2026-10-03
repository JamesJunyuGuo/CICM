# R13-llama3b-causal-head-routing causal-head discovery: meta-llama/Llama-3.2-3B-Instruct

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=120, validation=120, reserved confirmation=960.

Candidate-token audit: 0.989 (94/95).
First-token collision exclusions (discovery only): 6.

## Causally ranked heads

- L21H20: failure gradient=0.55746, correct gradient=0.17337, target mass=0.73410
- L16H15: failure gradient=0.29419, correct gradient=0.13792, target mass=0.13499
- L13H5: failure gradient=0.29191, correct gradient=0.09396, target mass=0.04089
- L26H10: failure gradient=0.26370, correct gradient=0.36600, target mass=0.15230
- L15H18: failure gradient=0.24450, correct gradient=0.03906, target mass=0.20464
- L26H11: failure gradient=0.22533, correct gradient=0.36569, target mass=0.28996
- L18H9: failure gradient=0.19226, correct gradient=0.27800, target mass=0.07950
- L24H4: failure gradient=0.18453, correct gradient=0.00108, target mass=0.36126
- L18H10: failure gradient=0.17074, correct gradient=0.03993, target mass=0.33190
- L17H9: failure gradient=0.16778, correct gradient=0.07056, target mass=0.20594
- L27H16: failure gradient=0.16505, correct gradient=0.05558, target mass=0.08958
- L18H19: failure gradient=0.15022, correct gradient=0.01585, target mass=0.15161
- L21H18: failure gradient=0.11807, correct gradient=0.06793, target mass=0.47467
- L27H10: failure gradient=0.10739, correct gradient=0.08752, target mass=0.07408
- L26H21: failure gradient=0.10054, correct gradient=0.06647, target mass=0.16372
- L20H20: failure gradient=0.10003, correct gradient=0.00623, target mass=0.24338
- L14H2: failure gradient=0.08803, correct gradient=0.04139, target mass=0.09604
- L13H17: failure gradient=0.08542, correct gradient=0.07355, target mass=0.04763
- L24H3: failure gradient=0.07929, correct gradient=0.04087, target mass=0.30839
- L19H19: failure gradient=0.07482, correct gradient=0.01196, target mass=0.10244
- L19H11: failure gradient=0.07198, correct gradient=0.05039, target mass=0.20570
- L16H14: failure gradient=0.07151, correct gradient=0.03327, target mass=0.04462
- L19H0: failure gradient=0.06919, correct gradient=0.03607, target mass=0.32411
- L22H5: failure gradient=0.06862, correct gradient=0.07566, target mass=0.12610
- L18H18: failure gradient=0.06658, correct gradient=0.00672, target mass=0.12073
- L23H11: failure gradient=0.06648, correct gradient=0.04436, target mass=0.18692
- L21H19: failure gradient=0.06485, correct gradient=0.12315, target mass=0.40422
- L26H9: failure gradient=0.06466, correct gradient=-0.00139, target mass=0.24023
- L22H10: failure gradient=0.06306, correct gradient=0.03984, target mass=0.15476
- L17H0: failure gradient=0.05228, correct gradient=0.01549, target mass=0.37048
- L16H17: failure gradient=0.05137, correct gradient=0.03077, target mass=0.02786
- L12H3: failure gradient=0.05018, correct gradient=0.02305, target mass=0.03716

## Validation

Selected top-32 at margin=4.0.

Validation accuracy: 0.283 -> 0.883; gain=0.600; preservation=1.000; within-stale correction=0.894.

Opposite-direction gain: -0.275.
