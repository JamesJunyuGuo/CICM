import unittest

from openrouter_sweep import build_eval_rows, summarize


class OpenRouterSweepTest(unittest.TestCase):
    def test_build_eval_rows_takes_fixed_per_cell_p0_and_ood(self):
        p0 = []
        for condition, load in [("simple", 0), ("interference", 2)]:
            for i in range(3):
                p0.append(
                    {
                        "id": f"p0-{condition}-{i}",
                        "condition": condition,
                        "n_lines": 20,
                        "interference_load": load,
                        "prompt": "What is X?",
                        "gold": i,
                        "stale_values": [i + 10],
                    }
                )
        stage_c = []
        for load in [12, 16, 4]:
            for i in range(3):
                stage_c.append(
                    {
                        "id": f"c-{load}-{i}",
                        "condition": "interference",
                        "n_lines": 160,
                        "interference_load": load,
                        "prompt": "What is X?",
                        "gold": i,
                        "stale_values": [i + 10],
                    }
                )

        rows = build_eval_rows(p0, stage_c, per_cell=2, seed=5)

        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["openrouter_source"] == "p0" for row in rows), 4)
        self.assertEqual(sum(row["openrouter_source"] == "stage_c_ood" for row in rows), 4)
        self.assertFalse(any(row["interference_load"] == 4 for row in rows))

    def test_summarize_reports_within_stale_among_errors(self):
        results = [
            {
                "model": "m",
                "openrouter_source": "p0",
                "condition": "interference",
                "n_lines": 20,
                "interference_load": 2,
                "correct": 1,
                "answer_type": "correct",
            },
            {
                "model": "m",
                "openrouter_source": "p0",
                "condition": "interference",
                "n_lines": 20,
                "interference_load": 2,
                "correct": 0,
                "answer_type": "within_stale",
            },
        ]

        summary = summarize(results, {"meta": True})

        self.assertEqual(summary["cells"][0]["accuracy"], 0.5)
        self.assertEqual(summary["cells"][0]["within_stale_rate_among_errors"], 1.0)


if __name__ == "__main__":
    unittest.main()
