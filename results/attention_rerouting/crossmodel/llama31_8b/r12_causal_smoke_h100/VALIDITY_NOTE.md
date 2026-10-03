# R12 smoke validity note

This smoke is pipeline-valid but not scientifically eligible for freezing the
held-out confirmation configuration. The original implementation applied
`--limit-per-cell 20` before constructing the canonical 240/960 split. Of the
24 discovery/validation rows, 23 belong to the canonical calibration split and
one (`stage_l_natfact_1_00043`) belongs to the canonical confirmation split.

Therefore the reported validation gain (0.50 to 1.00 for top-16, margin 4) is
only evidence that causal-gradient head ranking can move Llama outputs. It must
not select a configuration for the 960-row confirmation. The implementation is
repaired to split first and limit only within calibration; a fresh calibration
run must freeze the head set and margin before confirmation.
