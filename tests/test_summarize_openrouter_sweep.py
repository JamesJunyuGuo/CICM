import unittest

from summarize_openrouter_sweep import summarize


class SummarizeOpenRouterSweepTest(unittest.TestCase):
    def test_summarize_marks_completion_and_error_mix(self):
        rows = [
            {
                "model": "m",
                "openrouter_source": "p0",
                "condition": "interference",
                "n_lines": 40,
                "interference_load": 4,
                "correct": 1,
                "answer_type": "correct",
            },
            {
                "model": "m",
                "openrouter_source": "p0",
                "condition": "interference",
                "n_lines": 40,
                "interference_load": 4,
                "correct": 0,
                "answer_type": "within_stale",
            },
        ]
        out = summarize(rows, {"n_calls": 2, "models": ["m"]})
        self.assertTrue(out["complete"])
        self.assertEqual(out["counts_by_model"], {"m": 2})
        self.assertEqual(out["cells"][0]["accuracy"], 0.5)
        self.assertEqual(out["cells"][0]["within_stale_rate_among_errors"], 1.0)


if __name__ == "__main__":
    unittest.main()
