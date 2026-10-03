# Stage R11 smoke adjudication

Date: 2026-09-11

All four paired smoke jobs completed with exit code 0. Route audits and
`beta=0` identity checks were exact for both models. The Qwen2.5-14B config
contains a `sliding_window` field but sets `use_sliding_window=false`; the
generic eager-attention warning is therefore not an active sliding-window
protocol change.

## Qwen2.5-14B-Instruct: proceed to full

Both hardware runs selected the same eight heads (with only a small ordering
swap) and the same `beta=8`. On the 12-row calibration subset, gain was zero at
`beta` 0.5, 1, and 2, then +0.083 at 4 and +0.167 at 8, with 1.000 correct
preservation. First-token logits changed and responses flipped at nonzero
`beta`, so the preregistered smoke validity gate passes. The 48-row confirmation
metrics are underpowered smoke diagnostics, not efficacy results.

## Llama-3.1-8B-Instruct: stop under R11

Both hardware runs selected the same eight heads and reproduced the same
behavior. First-token logits changed increasingly over `beta=0.5..8`, but the
12 calibration responses never changed at any dose. This fails the explicit
smoke gate requiring at least one calibration response flip. The confirmation
route chosen by the all-tied calibration rule (`beta=0.5`) changed retrieval
from 0.438 to 0.417 and corrected no stale responses. Do not launch the frozen
R11 full protocol for Llama.

This stop does not establish that attention routing can never work for Llama.
It establishes that the preregistered eight-head discovery score plus fixed
raw-logit beta grid did not produce an active correction in this smoke. A
scale-normalized or adaptive-margin operator would be a new experiment and
must not be substituted into R11 after observing this result.
