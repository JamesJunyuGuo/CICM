import unittest

from build_stable_labels import build_stable_rows, summarize_stability


class StableLabelsTest(unittest.TestCase):
    def test_build_stable_rows_classifies_agreement_and_flips(self):
        extract = [
            {"row_index": 0, "id": 10, "condition": "simple", "n_lines": 40, "interference_load": 0, "correct": 1},
            {"row_index": 1, "id": 11, "condition": "interference", "n_lines": 40, "interference_load": 4, "correct": 0},
            {"row_index": 2, "id": 12, "condition": "interference", "n_lines": 40, "interference_load": 8, "correct": 1},
        ]
        sdpa = {
            10: {"correct": 1, "pred": 7, "raw": "7"},
            11: {"correct": 0, "pred": 8, "raw": "8"},
            12: {"correct": 0, "pred": 9, "raw": "9"},
        }

        rows = build_stable_rows(extract, sdpa)
        self.assertEqual([r["stable_label"] for r in rows], ["stable_correct", "stable_wrong", "flipped"])
        self.assertEqual([r["stable_primary"] for r in rows], [1, 1, 0])

    def test_summarize_stability_reports_flip_rates(self):
        rows = [
            {"condition": "simple", "n_lines": 40, "interference_load": 0, "stable_label": "stable_correct"},
            {"condition": "simple", "n_lines": 40, "interference_load": 0, "stable_label": "flipped"},
            {"condition": "interference", "n_lines": 40, "interference_load": 4, "stable_label": "stable_wrong"},
        ]
        summary = summarize_stability(rows)
        self.assertEqual(summary["counts"]["stable_correct"], 1)
        self.assertEqual(summary["counts"]["stable_wrong"], 1)
        self.assertEqual(summary["counts"]["flipped"], 1)
        simple = [c for c in summary["by_cell"] if c["condition"] == "simple"][0]
        self.assertAlmostEqual(simple["flip_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
