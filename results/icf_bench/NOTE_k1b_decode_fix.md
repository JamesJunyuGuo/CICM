# K1b — root-cause fix for the K1 measurement bug (2026-07-20)

## Diagnosis (confirmed by inspecting rows)

The K1 numbers are contaminated by a **left-padding decode-slice bug** in
`src/icf_generate.py::generate_texts`, NOT by real model behavior. The tokenizer uses
`padding_side="left"`, so `model.generate` returns `[PAD*p, prompt*r, new*g]` and the new
tokens begin at the **padded input width** (`input_ids.shape[1]`) for every row. The code
instead slices at each row's **non-pad length** `r`:

    prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()   # = r (non-pad len)
    outputs.append(tokenizer.decode(seq[int(prompt_len):], ...))  # BUG: slices into prompt

For any row shorter than the batch max (p>0), `seq[r:]` starts INSIDE the real prompt, so the
captured "response" = prompt tail (e.g. the DP option list, which contains `old_value`) +
the literal `assistant` generation-prompt marker + the true answer. Evidence: sampled DP
`ambiguous` rows all show `...D. <old_value> ... assistant\nA. <current_value>` where the
model actually chose the CORRECT current option; `old_value` only appears because option D
was sliced in. This inflates DP `ambiguous` to 44% and makes DP `within_stale`=0.204
meaningless.

**This same bug also affects IF** (its prompt echo contains the conversation history = the
to-forget facts), so IF `within_stale`=0.913 is ALSO not trustworthy until re-run — though
it is plausibly mostly real (the paper reports Qwen2.5-7B IF SFRR ≈ 0.23%, i.e. it almost
never obeys forget instructions). Do not treat either number as final.

## Fix (one change) — then RE-RUN both scenarios

Raw token ids were not saved, so this cannot be re-parsed from text; re-generate. In
`generate_texts`, replace the per-row non-pad slice with the padded input width:

    input_width = encoded["input_ids"].shape[1]   # new tokens start here for ALL rows
    for seq in generated:
        outputs.append(tokenizer.decode(seq[input_width:], skip_special_tokens=True).strip())

(Optional sanity: assert no decoded response contains the literal `assistant` marker or the
prompt tail after the fix.)

## Then, before any headline

1. Re-generate DP and IF on Qwen2.5-7B with the fixed slice; re-run the K1 classification.
2. **Validate matchers against `external_data/icf_bench/human_llm_align/`** (50×3 per
   scenario) — report matcher-vs-human agreement; a scenario below agreement stays out of
   the headline. This is now mandatory (it would have caught the DP corruption).
3. Re-adjudicate the ≥2-scenario within-stale gate on the clean data.

## Pre-registered readings (unchanged criteria; both branches proceed, do not stop)

- Both IF and DP within-stale dominant (CI lower bound ≥ 0.70) → gate passes 2/2 → proceed
  to K2 (measure the attention/dilution mechanism).
- IF dominant but clean DP NOT dominant (when DP genuinely fails, it picks a distractor/old
  near chance) → a real IF-vs-DP dissociation: suppression-on-command stale-binds,
  preference-override does not. Pivot K2/K3 to explain the contrast (selection-failure in
  IF; recency-override available in DP). Stronger than a clean 2/2.

Do NOT report K1 headline numbers until the slice fix + matcher validation are done.
