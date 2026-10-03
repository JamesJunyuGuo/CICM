import math
import unittest

import numpy as np

from stage_f_mech_loop import (
    classify_mechanism_reading,
    mean_head_values,
    top_gain_heads,
)
from stage_f_transport_law import auc_score, decile_points, fit_logistic_1d


class StageFMechanismTest(unittest.TestCase):
    def test_top_gain_heads_reports_layer_head_pairs(self):
        gain = np.zeros((3, 4), dtype=np.float32)
        gain[2, 1] = 0.9
        gain[0, 3] = 0.5

        heads = top_gain_heads(gain, top_k=2)

        self.assertEqual(
            heads,
            [
                {"layer": 2, "head": 1, "gain": 0.8999999761581421},
                {"layer": 0, "head": 3, "gain": 0.5},
            ],
        )

    def test_mean_head_values_uses_selected_pairs(self):
        values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

        selected = mean_head_values(
            values,
            [{"layer": 0, "head": 1}, {"layer": 2, "head": 3}],
        )

        expected = np.array([(1 + 11) / 2, (13 + 23) / 2], dtype=np.float32)
        np.testing.assert_allclose(selected, expected)

    def test_classify_mechanism_reading_prefers_grafting_for_random_own_heads(self):
        arm_summaries = {
            "arm_l": {"original_top8_overlap_in_top16": 6, "own_head_overlap_in_top16": 6},
            "random_heads": {
                "original_top8_overlap_in_top16": 1,
                "own_head_overlap_in_top16": 5,
            },
            "generic": {"original_top8_overlap_in_top16": 2, "own_head_overlap_in_top16": None},
        }

        self.assertEqual(classify_mechanism_reading(arm_summaries), "grafting")


class StageFTransportLawTest(unittest.TestCase):
    def test_auc_score_handles_perfect_and_tied_rankings(self):
        self.assertEqual(auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(auc_score([0, 1], [0.5, 0.5]), 0.5)
        self.assertTrue(math.isnan(auc_score([1, 1], [0.1, 0.2])))

    def test_fit_logistic_1d_predicts_monotone_probabilities(self):
        fit = fit_logistic_1d(
            np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0], dtype=np.float64),
            np.array([0, 0, 0, 1, 1, 1], dtype=np.float64),
        )

        self.assertGreater(fit["coef"], 0)
        self.assertLess(fit["predict"]([0.1])[0], fit["predict"]([0.9])[0])

    def test_decile_points_reports_empirical_accuracy_by_bin(self):
        points = decile_points(
            np.linspace(0.0, 1.0, 20),
            np.array([0] * 10 + [1] * 10),
            n_bins=4,
        )

        self.assertEqual(len(points), 4)
        self.assertEqual(points[0]["n"], 5)
        self.assertEqual(points[-1]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
