# Stage F Report

## Status

Stage F is complete for F-1 and F-2. I did not modify
`results/hard_tier_and_transport/hard_tier/`.

Main artifacts:

- F-1 summary: `results/hard_tier_and_transport/mech_loop_summary.json`
- F-2 summary: `results/hard_tier_and_transport/transport_law_summary.json`
- F9: `results/figures/F9_repaired_p_last_gain.{png,pdf}` plus
  `results/figures/F9_repaired_p_last_gain.data.json`
- F10: `results/figures/F10_transport_share_law.{png,pdf}` plus
  `results/figures/F10_transport_share_law.data.json`

## F-1 Repaired-Model Mechanism Loop

The 0.5B smoke completed successfully on job `19059802`; the full 7B job
completed on job `19060387` in 15:59. The GPU duplicate was cancelled after the
AI full job started healthy.

Hard gate passed for all three adapter paths. The adapter-off identity check
matched the stored baseline predictions exactly on 20 examples for Arm L,
Random-heads, and Generic; all three had zero mismatches.

The repaired arms strongly increase last-write transport on originally stable
wrong examples, but the location of the increase is not a clean grafting story.
The best reading is **mixed/diffuse with partial native-circuit restoration**.

| arm | accuracy on F-1 slice | stable-wrong flips | original top-8 overlap in top-16 gain heads | own-head overlap | original top-8 mean gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arm L | 0.995 | 193/194 | 5 | 5 | 0.235 |
| Random-heads | 0.985 | 189/194 | 3 | 0 | 0.197 |
| Generic | 1.000 | 194/194 | 0 | n/a | 0.261 |

Interpretation:

- Arm L behaves like targeted circuit sharpening: gains include 5 of the
  original top-8 heads, and original-top8 p_last on flipped examples rises from
  0.397 to 0.772.
- Random-heads does not show grafting into its own random heads: own-head
  overlap is 0, and own-head p_last on flipped examples only rises from 0.211
  to 0.259. Instead, random training still increases original-top8 p_last from
  0.401 to 0.717, plus nearby late heads.
- Generic is broad/diffuse: it reaches 194/194 flips and raises original-top8
  p_last from 0.396 to 0.816, but its largest gain heads are mostly other
  late-layer heads.

This helps explain the C4 specificity gap: random-head training can recover
in-distribution behavior by indirectly restoring late selection, but it does
not install a clean, portable random-head selection circuit. Targeted Arm L
more directly sharpens the native circuit.

## F-2 Transport-Share Law

The per-example transport-share analysis uses existing Qwen and Llama stable
extractions only. The main predictor is mean p_last over each model's own top-8
A1 heads, with all-head mean and I-only as controls.

| model | n | accuracy | top-8 p_last AUC | all-head p_last AUC | I-only AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | 346 | 0.679 | 0.842 | 0.744 | 0.715 |
| Llama3.1-8B | 439 | 0.383 | 0.640 | 0.616 | 0.553 |

The monotone law is strong for Qwen and present but weaker for Llama. Within-I
checks support that this is not only an interference-load confound: Qwen top-8
AUC is 0.817/0.739/0.879 for I=2/4/8, while Llama top-8 AUC is
0.633/0.722/0.583 for I=2/4/8.

Verdict: per-example current-write transport share predicts success above the
I-only baseline, especially in Qwen. For Llama the effect is weaker, so the
single-example law should be stated as supportive rather than the sole basis of
the mechanism claim.
