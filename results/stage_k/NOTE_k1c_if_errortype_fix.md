# K1c — fix the IF within-stale ERROR-TYPE matcher (2026-07-20)

## What's fine (do not touch)
- DP is clean after the decode fix: no more `ambiguous`, noforget_reference 0.81,
  within-stale 0.294 with a real other/other_valid/stale distribution. Trustworthy.
- Forget SUCCESS/FAILURE agreement vs `human_llm_align` (DP 0.93, IF 0.92) is solid — the
  pass/fail call is validated.

## The problem: IF within-stale = 1.000 is a matcher artifact
The human-alignment number validated the pass/fail binary, NOT the within-stale-vs-other
error-TYPE split — which is our headline. Sampled IF `within_stale` rows include false
positives:
- id=0: response = "...Could you please clarify which numbers you are referring to?" — a
  DEFLECTION with zero stale values, labeled within_stale.
- id=1: `old_value` extracted as `'s say you'` (garbage from the quote regex on "let's say
  you"); the model gave a DIFFERENT wrong tip, labeled within_stale.
IF `error_type` has NO `other` bucket at all (within_stale=980, correct_forget=14), i.e.
`within_stale` has collapsed to mean "not a refusal." So 1.000 is a tautology, not a finding.
The IF failure RATE (~98.6%, model almost never forgets) is real; the within-stale TYPE
breakdown is inflated.

## Fix (CPU-only re-score of the already-saved, now-clean responses)
1. **Span extraction** (`extract_forgotten_spans`): stop using the apostrophe/quote regex
   that grabs contraction fragments (`'s say you`). Extract the SPECIFIC referent of the
   `forget_instruction` (the named entity / value the user asked to forget), not arbitrary
   quoted fragments; for number-type, match the specific numbers in scope, not every digit.
2. **Add a real `deflect/other` bucket**: a failure that neither refuses NOR restates the
   specific forgotten value (e.g. "which numbers?", or a different wrong answer) must be
   classified `deflect`/`other`, NOT within_stale. within_stale must require the SPECIFIC
   stale value to appear.
3. **Validate the ERROR-TYPE breakdown** (within_stale vs deflect vs other) against a small
   hand-labeled sample (~50 IF failures) — the existing `human_llm_align` set only has
   pass/fail labels, so error-type needs its own spot-check. Report agreement.
4. Re-score IF (no re-generation needed — responses are saved and clean) and re-read the IF
   within-stale fraction.

## Then re-adjudicate — do NOT proceed to K2/K3 until this is done
- Likely: clean IF within-stale drops below 1.000 but stays well above DP's 0.294 → the
  IF-vs-DP dissociation survives, now honest, and a THIRD failure mode (deflection) is
  surfaced — itself mechanistically interesting (neither forgets nor restates).
- If clean IF within-stale is NOT dominant either → the dissociation weakens; re-read.
Only after a clean IF error-type breakdown do we proceed to K2/K3 to explain the contrast.
Do not build mechanism (K2/K3) on the inflated 1.000.
