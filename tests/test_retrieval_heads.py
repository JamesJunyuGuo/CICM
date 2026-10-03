import math
import unittest

from retrieval_heads import rankdata, spearman, verdict_from_overlap


class RetrievalHeadsTest(unittest.TestCase):
    def test_rankdata_averages_ties(self):
        self.assertEqual(rankdata([10, 20, 20, 40]).tolist(), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_handles_monotone_inputs(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_spearman_returns_nan_for_constant_rank(self):
        self.assertTrue(math.isnan(spearman([1, 1, 1], [1, 2, 3])))

    def test_verdict_labels_overlap_regimes(self):
        self.assertEqual(verdict_from_overlap(8, 8), "same heads")
        self.assertEqual(verdict_from_overlap(0, 8), "disjoint")
        self.assertEqual(verdict_from_overlap(3, 8), "partial overlap")


if __name__ == "__main__":
    unittest.main()
