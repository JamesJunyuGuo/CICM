import unittest

import numpy as np

from analyze_write_attention import (
    compute_p_last,
    stale_answer_ratios,
    wrong_mask_for_rows as a1_wrong_mask_for_rows,
)
from analyze_logit_lens import (
    should_drop_shared_first_token,
    wrong_mask_for_rows as a2_wrong_mask_for_rows,
)


class AnalysisHelpersTest(unittest.TestCase):
    def test_compute_p_last_uses_per_example_write_count(self):
        attn = np.zeros((2, 1, 1, 4), dtype=np.float32)
        attn[0, 0, 0, :3] = [1.0, 1.0, 2.0]
        attn[1, 0, 0, :2] = [3.0, 1.0]
        p_last = compute_p_last(attn, np.array([3, 2]))
        self.assertAlmostEqual(float(p_last[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(p_last[1, 0, 0]), 0.25)

    def test_stale_answer_ratios_compare_answered_write_to_current(self):
        attn = np.zeros((1, 1, 1, 4), dtype=np.float32)
        attn[0, 0, 0, :3] = [0.2, 0.6, 0.2]
        rows = [{"answer_type": "stale", "answered_write_idx": 1, "write_count": 3}]
        ratios = stale_answer_ratios(attn, rows)
        self.assertAlmostEqual(float(ratios["answered"][0, 0, 0]), 0.6)
        self.assertAlmostEqual(float(ratios["last"][0, 0, 0]), 0.2)

    def test_shared_first_token_drop_rule(self):
        self.assertTrue(should_drop_shared_first_token(5, [1, 5, 9]))
        self.assertFalse(should_drop_shared_first_token(5, [1, 2, 9]))

    def test_within_stale_wrong_filter_excludes_other_wrong(self):
        rows = [
            {"correct": 1, "answer_type": "correct"},
            {"correct": 0, "answer_type": "stale"},
            {"correct": 0, "answer_type": "other"},
        ]

        self.assertEqual(a1_wrong_mask_for_rows(rows, "within_stale").tolist(), [False, True, False])
        self.assertEqual(a2_wrong_mask_for_rows(rows, "within_stale"), [False, True, False])


if __name__ == "__main__":
    unittest.main()
