import unittest

import torch

from intervene_stage_b import (
    AttentionBiasContext,
    apply_attention_bias,
    build_example_sets,
    build_scope_heads,
    resolve_sanity_baseline_mode,
)


class StageBInterventionTest(unittest.TestCase):
    def test_attention_bias_targets_only_configured_heads_and_span_columns(self):
        scores = torch.zeros((1, 4, 3, 6), dtype=torch.float32)
        ctx = AttentionBiasContext(
            alpha=4.0,
            span_token_start=2,
            span_token_end=5,
            heads_by_layer={3: (1, 3)},
        )

        biased = apply_attention_bias(scores, layer_idx=3, ctx=ctx)

        expected = torch.zeros_like(scores)
        expected[:, [1, 3], :, 2:5] = torch.log(torch.tensor(4.0))
        self.assertTrue(torch.equal(biased, expected))

    def test_attention_bias_alpha_one_is_bitwise_identity(self):
        scores = torch.randn((1, 2, 2, 5), dtype=torch.float32)
        ctx = AttentionBiasContext(
            alpha=1.0,
            span_token_start=1,
            span_token_end=4,
            heads_by_layer={0: (0, 1)},
        )

        biased = apply_attention_bias(scores, layer_idx=0, ctx=ctx)

        self.assertIs(biased, scores)
        self.assertTrue(torch.equal(biased, scores))

    def test_build_scope_heads_uses_spec_scopes(self):
        top_summary = {
            "top_heads_by_correct_p_last": [
                {"layer": 23, "head": 11, "mean_p_last_correct": 0.9},
                {"layer": 20, "head": 23, "mean_p_last_correct": 0.8},
            ]
        }

        late = build_scope_heads("late", num_layers=30, num_heads=32, top_summary=top_summary)
        self.assertEqual(set(late.keys()), set(range(18, 28)))
        self.assertEqual(late[18], tuple(range(32)))

        topk = build_scope_heads("topk", num_layers=30, num_heads=32, top_summary=top_summary, top_k=2)
        self.assertEqual(topk, {23: (11,), 20: (23,)})

        all_heads = build_scope_heads("all", num_layers=2, num_heads=3, top_summary=top_summary)
        self.assertEqual(all_heads, {0: (0, 1, 2), 1: (0, 1, 2)})

    def test_topk_scope_skips_heads_outside_smoke_model_shape(self):
        top_summary = {
            "top_heads_by_correct_p_last": [
                {"layer": 20, "head": 23, "mean_p_last_correct": 0.9},
                {"layer": 23, "head": 11, "mean_p_last_correct": 0.8},
                {"layer": 22, "head": 2, "mean_p_last_correct": 0.7},
            ]
        }

        topk = build_scope_heads("topk", num_layers=24, num_heads=14, top_summary=top_summary, top_k=2)

        self.assertEqual(topk, {23: (11,), 22: (2,)})

    def test_sanity_baseline_mode_auto_uses_same_model_for_smoke(self):
        self.assertEqual(
            resolve_sanity_baseline_mode("auto", "Qwen/Qwen2.5-7B-Instruct"),
            "stage_a",
        )
        self.assertEqual(
            resolve_sanity_baseline_mode("auto", "Qwen/Qwen2.5-0.5B-Instruct"),
            "same_model",
        )
        self.assertEqual(
            resolve_sanity_baseline_mode("stage_a", "Qwen/Qwen2.5-0.5B-Instruct"),
            "stage_a",
        )

    def test_build_example_sets_matches_stage_b_primary_cells(self):
        extract_rows = [
            {"id": 1, "condition": "interference", "interference_load": 4, "correct": 0, "answer_type": "stale"},
            {"id": 2, "condition": "interference", "interference_load": 2, "correct": 1, "answer_type": "correct"},
            {"id": 3, "condition": "simple", "interference_load": 0, "correct": 1, "answer_type": "correct"},
            {"id": 4, "condition": "interference", "interference_load": 8, "correct": 1, "answer_type": "correct"},
        ]
        stable_rows = [
            {"id": 1, "stable_label": "stable_wrong"},
            {"id": 2, "stable_label": "stable_correct"},
            {"id": 3, "stable_label": "stable_correct"},
            {"id": 4, "stable_label": "flipped"},
        ]

        sets = build_example_sets(extract_rows, stable_rows, correct_subsample_n=10, seed=7)

        self.assertEqual([row["id"] for row in sets["W"]], [1])
        self.assertEqual([row["id"] for row in sets["C"]], [2])
        self.assertEqual([row["id"] for row in sets["S"]], [3])

    def test_build_example_sets_can_filter_wrong_to_within_stale(self):
        extract_rows = [
            {"id": 1, "condition": "interference", "interference_load": 4, "correct": 0, "answer_type": "stale"},
            {"id": 2, "condition": "interference", "interference_load": 4, "correct": 0, "answer_type": "other"},
        ]
        stable_rows = [
            {"id": 1, "stable_label": "stable_wrong"},
            {"id": 2, "stable_label": "stable_wrong"},
        ]

        sets = build_example_sets(
            extract_rows,
            stable_rows,
            correct_subsample_n=10,
            seed=7,
            wrong_filter="within_stale",
        )

        self.assertEqual([row["id"] for row in sets["W"]], [1])


if __name__ == "__main__":
    unittest.main()
