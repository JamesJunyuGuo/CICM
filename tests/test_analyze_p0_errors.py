import unittest

from analyze_p0_errors import summarize_p0_errors


class AnalyzeP0ErrorsTest(unittest.TestCase):
    def test_summarizes_within_stale_cross_and_other_errors(self):
        data_rows = [
            {
                "id": "a",
                "condition": "interference",
                "n_lines": 40,
                "interference_load": 4,
                "gold": 50,
                "stale_values": [10, 20],
            },
            {
                "id": "b",
                "condition": "interference",
                "n_lines": 40,
                "interference_load": 4,
                "gold": 60,
                "stale_values": [30, 40],
            },
            {
                "id": "c",
                "condition": "interference",
                "n_lines": 40,
                "interference_load": 4,
                "gold": 70,
                "stale_values": [55, 80],
            },
        ]
        result_rows = [
            {"id": "a", "pred": 10, "correct": 0},
            {"id": "b", "pred": 9999, "correct": 0},
            {"id": "c", "pred": 50, "correct": 0},
        ]

        summary = summarize_p0_errors(data_rows, result_rows, model="m")

        self.assertEqual(summary["overall"]["n_wrong"], 3)
        self.assertEqual(summary["overall"]["within_stale"], 1)
        self.assertEqual(summary["overall"]["cross_value"], 1)
        self.assertEqual(summary["overall"]["other"], 1)
        self.assertAlmostEqual(summary["overall"]["within_stale_rate_among_wrong"], 1 / 3)
        self.assertEqual(summary["model"], "m")


if __name__ == "__main__":
    unittest.main()
