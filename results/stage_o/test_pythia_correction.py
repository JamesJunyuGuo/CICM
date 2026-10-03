import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, "/anvil/scratch/x-jguo7/agent_eval/Contextual_management/src")

from pythia_correction import apply_head_scale, adjudicate_gamma


class PythiaCorrectionTests(unittest.TestCase):
    def test_identity_scale_is_exact(self):
        hidden = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        query = torch.tensor([2, 1])
        scaled = apply_head_scale(hidden, query, [1], 1.0, n_heads=4)
        self.assertTrue(torch.equal(hidden, scaled))

    def test_zero_scale_only_changes_selected_query_slice(self):
        hidden = torch.ones(2, 3, 8)
        query = torch.tensor([2, 1])
        scaled = apply_head_scale(hidden, query, [1], 0.0, n_heads=4)
        expected = hidden.clone()
        expected[0, 2, 2:4] = 0
        expected[1, 1, 2:4] = 0
        self.assertTrue(torch.equal(expected, scaled))

    def test_operating_gate_requires_all_controls(self):
        row = {
            "gamma": 0.5,
            "targeted": {
                "correction": {"mean": 0.3},
                "correct_preservation": 0.96,
                "k0_accuracy": 0.98,
                "lm_delta": 0.01,
            },
            "random": {
                "correction": {"quantile95": [0.0, 0.2]},
                "lm_delta": {"mean": 0.02},
            },
        }
        verdict = adjudicate_gamma(row, k0_baseline=0.989)
        self.assertTrue(verdict["three_hard_controls_pass"])
        row["targeted"]["correct_preservation"] = 0.94
        self.assertFalse(adjudicate_gamma(row, 0.989)["three_hard_controls_pass"])


if __name__ == "__main__":
    unittest.main()
