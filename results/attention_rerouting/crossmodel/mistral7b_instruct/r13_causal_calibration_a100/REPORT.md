# R13-mistral7b-causal-head-routing causal-head discovery: mistralai/Mistral-7B-Instruct-v0.3

**Verdict:** `validation_gate_passed_confirmation_not_run`.

Rows: discovery=120, validation=120, reserved confirmation=960.

Candidate-token audit: 0.977 (85/87).
First-token collision exclusions (discovery only): 6.

## Causally ranked heads

- L19H9: failure gradient=0.95851, correct gradient=0.64963, target mass=0.55951
- L19H16: failure gradient=0.49710, correct gradient=0.18717, target mass=0.23061
- L16H29: failure gradient=0.36797, correct gradient=0.21497, target mass=0.24127
- L18H3: failure gradient=0.32738, correct gradient=0.23134, target mass=0.30879
- L31H19: failure gradient=0.32603, correct gradient=0.07719, target mass=0.32558
- L21H7: failure gradient=0.32433, correct gradient=0.31474, target mass=0.17567
- L19H8: failure gradient=0.26679, correct gradient=0.11198, target mass=0.20900
- L16H1: failure gradient=0.25930, correct gradient=0.01490, target mass=0.03406
- L31H17: failure gradient=0.18742, correct gradient=0.20408, target mass=0.30067
- L18H12: failure gradient=0.18211, correct gradient=0.06013, target mass=0.30034
- L25H29: failure gradient=0.16974, correct gradient=0.19104, target mass=0.48068
- L31H18: failure gradient=0.16437, correct gradient=0.12368, target mass=0.40550
- L31H4: failure gradient=0.15785, correct gradient=0.14598, target mass=0.35104
- L24H15: failure gradient=0.15460, correct gradient=0.06092, target mass=0.30049
- L18H30: failure gradient=0.15044, correct gradient=0.05171, target mass=0.24456
- L24H5: failure gradient=0.14684, correct gradient=0.07043, target mass=0.63096
- L30H1: failure gradient=0.14511, correct gradient=-0.00786, target mass=0.21222
- L26H17: failure gradient=0.12705, correct gradient=0.06328, target mass=0.12848
- L21H11: failure gradient=0.11535, correct gradient=0.04789, target mass=0.11388
- L24H21: failure gradient=0.11505, correct gradient=0.02595, target mass=0.40266
- L31H6: failure gradient=0.09983, correct gradient=0.09463, target mass=0.21110
- L20H14: failure gradient=0.09772, correct gradient=0.11253, target mass=0.08602
- L28H25: failure gradient=0.09448, correct gradient=-0.02299, target mass=0.31407
- L29H22: failure gradient=0.09362, correct gradient=0.31183, target mass=0.24957
- L27H29: failure gradient=0.08846, correct gradient=-0.00367, target mass=0.55039
- L29H9: failure gradient=0.08540, correct gradient=0.01686, target mass=0.31634
- L30H2: failure gradient=0.08536, correct gradient=0.00949, target mass=0.33054
- L20H6: failure gradient=0.08364, correct gradient=0.06288, target mass=0.15593
- L17H0: failure gradient=0.08268, correct gradient=0.00593, target mass=0.02528
- L31H31: failure gradient=0.08023, correct gradient=0.23050, target mass=0.42394
- L22H1: failure gradient=0.07867, correct gradient=-0.00229, target mass=0.05433
- L26H6: failure gradient=0.07788, correct gradient=0.00520, target mass=0.42503

## Validation

Selected top-32 at margin=2.0.

Validation accuracy: 0.100 -> 0.783; gain=0.683; preservation=1.000; within-stale correction=0.810.

Opposite-direction gain: -0.083.
