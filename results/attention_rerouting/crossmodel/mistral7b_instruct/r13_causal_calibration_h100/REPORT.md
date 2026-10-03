# R13-mistral7b-causal-head-routing causal-head discovery: mistralai/Mistral-7B-Instruct-v0.3

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=120, validation=120, reserved confirmation=960.

Candidate-token audit: 0.977 (86/88).
First-token collision exclusions (discovery only): 6.

## Causally ranked heads

- L19H9: failure gradient=0.95971, correct gradient=0.68441, target mass=0.55454
- L19H16: failure gradient=0.49070, correct gradient=0.17992, target mass=0.22815
- L16H29: failure gradient=0.36440, correct gradient=0.22679, target mass=0.23835
- L31H19: failure gradient=0.32503, correct gradient=0.04983, target mass=0.32625
- L18H3: failure gradient=0.31928, correct gradient=0.22948, target mass=0.30714
- L21H7: failure gradient=0.31263, correct gradient=0.31148, target mass=0.17484
- L19H8: failure gradient=0.26375, correct gradient=0.11513, target mass=0.20691
- L16H1: failure gradient=0.25511, correct gradient=0.01749, target mass=0.03351
- L31H17: failure gradient=0.19007, correct gradient=0.20119, target mass=0.29945
- L18H12: failure gradient=0.17979, correct gradient=0.07078, target mass=0.29666
- L25H29: failure gradient=0.16826, correct gradient=0.18139, target mass=0.47526
- L31H18: failure gradient=0.16539, correct gradient=0.12342, target mass=0.40290
- L31H4: failure gradient=0.16154, correct gradient=0.15310, target mass=0.35086
- L24H5: failure gradient=0.14990, correct gradient=0.07209, target mass=0.62988
- L18H30: failure gradient=0.14885, correct gradient=0.05101, target mass=0.24250
- L30H1: failure gradient=0.14444, correct gradient=-0.00746, target mass=0.21098
- L24H15: failure gradient=0.13539, correct gradient=0.06262, target mass=0.29688
- L26H17: failure gradient=0.12814, correct gradient=0.06294, target mass=0.12656
- L21H11: failure gradient=0.11267, correct gradient=0.04566, target mass=0.11246
- L24H21: failure gradient=0.11215, correct gradient=0.02557, target mass=0.39885
- L31H6: failure gradient=0.10351, correct gradient=0.09208, target mass=0.21009
- L20H14: failure gradient=0.09497, correct gradient=0.10092, target mass=0.08461
- L29H22: failure gradient=0.09339, correct gradient=0.32461, target mass=0.24754
- L27H29: failure gradient=0.09235, correct gradient=-0.00401, target mass=0.54604
- L28H25: failure gradient=0.09105, correct gradient=-0.01351, target mass=0.31075
- L23H14: failure gradient=0.08571, correct gradient=0.23914, target mass=0.19083
- L20H6: failure gradient=0.08308, correct gradient=0.06701, target mass=0.15455
- L30H2: failure gradient=0.08200, correct gradient=0.00551, target mass=0.32812
- L29H9: failure gradient=0.08135, correct gradient=0.00030, target mass=0.31301
- L17H0: failure gradient=0.08114, correct gradient=0.00752, target mass=0.02475
- L31H31: failure gradient=0.08083, correct gradient=0.23456, target mass=0.41998
- L26H6: failure gradient=0.07800, correct gradient=0.00444, target mass=0.42259

## Validation

Selected top-32 at margin=2.0.

Validation accuracy: 0.100 -> 0.775; gain=0.675; preservation=1.000; within-stale correction=0.817.

Opposite-direction gain: -0.075.
