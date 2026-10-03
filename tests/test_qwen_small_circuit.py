import sys
import unittest
from collections import Counter

import torch

sys.path.insert(0, "/anvil/scratch/x-jguo7/agent_eval/Contextual_management/src")

from qwen_small_circuit import (
    PART_D_SPECS,
    apply_query_head_scale,
    compute_caps,
    bounded_sequence_length,
    query_to_kv_head,
    select_exact_pairs,
    stable_stale_subset,
    stale_competitor,
)
from pythia_gen import TEMPLATES


class QwenSmallCircuitTests(unittest.TestCase):
    def test_caps_respect_variable_value_and_structural_limits(self):
        caps = compute_caps(n_variables=18, n_task_values=30, target_n_lines=15)
        self.assertEqual(caps["n_lines"], 15)
        self.assertEqual(caps["k_cap"], 10)
        self.assertEqual(caps["k_grid"], [0, 2, 4, 6, 8, 10])

    def test_qwen_identity_hook_is_exact(self):
        hidden = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12)
        query = torch.tensor([3, 2])
        scaled = apply_query_head_scale(hidden, query, [0, 2], 1.0, 3)
        self.assertTrue(torch.equal(hidden, scaled))

    def test_qwen_head_scaling_uses_query_heads(self):
        hidden = torch.ones(1, 3, 12)
        scaled = apply_query_head_scale(hidden, torch.tensor([2]), [2], 0.0, 3)
        self.assertTrue(torch.equal(scaled[0, 2, :8], torch.ones(8)))
        self.assertTrue(torch.equal(scaled[0, 2, 8:], torch.zeros(4)))

    def test_part_d_matrix_contains_authorized_gated_models(self):
        keys = {key for key, _label, _model in PART_D_SPECS}
        self.assertEqual(
            {"llama32_1b", "gemma2_2b", "gemma2_9b"} - keys,
            set(),
        )

    def test_query_to_kv_head_uses_contiguous_gqa_groups(self):
        self.assertEqual(
            [query_to_kv_head(head, 12, 2) for head in range(12)],
            [0] * 6 + [1] * 6,
        )
        with self.assertRaises(ValueError):
            query_to_kv_head(0, 12, 5)

    def test_exact_pair_selection_balances_template_and_split(self):
        rows = []
        for template in TEMPLATES:
            for split in ("discovery", "evaluation"):
                for index in range(20):
                    rows.append(
                        {
                            "id": f"{template}-{split}-{index:02d}",
                            "template": template,
                            "split": split,
                        }
                    )
        chosen = select_exact_pairs(rows, per_template_split=16)
        counts = Counter((row["template"], row["split"]) for row in chosen)
        self.assertEqual(len(chosen), 96)
        self.assertEqual(set(counts.values()), {16})

    def test_stale_competitor_uses_the_observed_prediction(self):
        row = {
            "prediction": " front",
            "pred_token_id": 4065,
            "stale_values": ["dance", "front"],
            "stale_token_ids": [15254, 4065],
            "strongest_stale_token_id": 15254,
        }
        self.assertEqual(stale_competitor(row), (4065, "front"))

        row["pred_token_id"] = 999
        with self.assertRaisesRegex(ValueError, "not among stale token ids"):
            stale_competitor(row)

    def test_stable_stale_subset_reports_baseline_flips(self):
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        stable, mismatches = stable_stale_subset(
            rows, ["within_stale", "correct_current", "within_stale"]
        )
        self.assertEqual([row["id"] for row in stable], ["a", "c"])
        self.assertEqual(
            mismatches,
            [{"id": "b", "baseline_label": "correct_current"}],
        )

    def test_induction_length_is_bounded_by_unique_token_pool(self):
        self.assertEqual(bounded_sequence_length(24, 17), 17)
        self.assertEqual(bounded_sequence_length(12, 17), 12)
        with self.assertRaisesRegex(ValueError, "at least two"):
            bounded_sequence_length(24, 1)


if __name__ == "__main__":
    unittest.main()
