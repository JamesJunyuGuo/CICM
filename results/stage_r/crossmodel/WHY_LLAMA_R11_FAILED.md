# Why the R11 Llama-3.1-8B smoke did not pass

Date: 2026-09-11

## Conclusion

The smoke does not show that Llama lacks stale-binding or that its current
value is absent. It shows that the R11 transfer rule -- select eight heads by a
failure-versus-correct stale/current attention-ratio contrast, then apply the
Qwen balanced routing operator -- did not identify an effective correction
set on Llama. The leading explanation is a mismatch between descriptive head
ranking and causal output influence, with raw-logit scale as a secondary open
possibility.

## What is directly established

- Both H100 and A100 executions completed with exact route audits and exact
  `beta=0` identity behavior. This is not a span or hook-integrity failure.
- Calibration contained six correct and six within-stale rows. For every
  `beta` in `{0.5,1,2,4,8}`, correct preservation was 1.000 but no calibration
  response changed.
- The intervention did reach the network. At `beta=8`, the maximum absolute
  first-token output-logit change was 6.0 and the vocabulary-wide mean absolute
  change was 0.324, yet no answer crossed a decision boundary.
- On the 48-row smoke confirmation split, the selected `beta=0.5` corrected no
  within-stale answer and changed retrieval accuracy from 0.438 to 0.417. The
  opposite-sign arm corrected two rows, but this tiny post-selection sample is
  not evidence that reverse routing is beneficial.

## Why the selected heads are the main suspect

The discovery score uses only a ratio:

`failure mean log(A_stale/A_current) - correct mean log(A_stale/A_current)`.

It does not require large absolute attention mass and does not test the OV
direction or downstream causal effect. This matters on Llama:

- L1H19 ranked first with score 5.63, but its mean total attention to stale plus
  current value tokens on failures was only 0.000055 (0.0055%). A large ratio
  over two almost-unused locations is not a strong transport path.
- Five of the selected eight heads placed less than 1% total failure attention
  on the two target groups. Only three exceeded 10%.
- The selected layers were 1, 3, 6, 16, 19, and 20 of 32. None was in the last
  eleven blocks. Later computation can overwrite or ignore an early/middle
  attention change.

The Qwen2.5-14B contrast is sharp: its eight selected heads lie in layers
35--42 of 48, and every selected head assigns more than 15% mean failure
attention to stale plus current target tokens. Its calibration answers begin
to flip at `beta=4` and improve further at `beta=8`.

Attention direction alone also omits the OV map. A head can attend to the
current token without writing a value direction that increases the current
answer logit; it can even have an inhibitory or routing-only role. The Llama
smoke did not measure this sign before selecting heads.

## What prior Llama results rule out

Stage-L L-1 found the current value decodable on Llama within-stale errors
(true-current probe score 0.822), so the current binding is not simply absent.
Residual-normalized steering at Llama's final block reduced target error by
69.5 percentage points relative to matched random steering, whereas the middle
layer effect was only 1.8 points. That result is qualified by a final-layer
answer-injection risk, but it independently shows that a late answer-selection
lever exists. R11 selected no late Llama head.

## Remaining uncertainty

Raw `beta` is not scale invariant. A larger or adaptive intervention might move
Llama answers, although `beta=8` already caused substantial output-logit
changes without a calibration flip. The present smoke therefore cannot cleanly
separate insufficient scaling from an ineffective head/OV set, but it weighs
more strongly toward the latter.

## Smallest decisive follow-up

Do not reinterpret or enlarge R11 after observing the smoke. A separate
experiment should use calibration-only causal head screening: test each
candidate head's effect on the current-minus-stale answer-logit gap, require
nontrivial absolute target attention, inspect its OV sign, and evaluate the
selected set on held-out rows. In parallel, compare fixed raw `beta` with an
adaptive aggregate-attention-margin operator. If causally selected late heads
work, R11 failed at head discovery; if only adaptive scaling works, the primary
failure was logit-scale mismatch; if neither works, Llama's correction route is
mechanistically different from Qwen's.
