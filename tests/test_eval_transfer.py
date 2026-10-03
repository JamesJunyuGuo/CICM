import unittest

from eval_transfer import build_adapter_args, score_stage_d_row, summarize_results


class StageDEvalTest(unittest.TestCase):
    def test_scores_integer_answer(self):
        row = {"gold": 42, "answer_type": "int", "stale_values": [17]}

        scored = score_stage_d_row(row, "42")

        self.assertEqual(scored["pred"], 42)
        self.assertEqual(scored["correct"], 1)
        self.assertEqual(scored["error_type"], "correct")

    def test_classifies_stale_integer_answer(self):
        row = {"gold": 42, "answer_type": "int", "stale_values": [17]}

        scored = score_stage_d_row(row, "17")

        self.assertEqual(scored["correct"], 0)
        self.assertEqual(scored["error_type"], "stale")

    def test_scores_normalized_string_answer(self):
        row = {"gold": "Avery", "answer_type": "string", "stale_values": ["Blair"]}

        scored = score_stage_d_row(row, "Avery.")

        self.assertEqual(scored["pred"], "avery")
        self.assertEqual(scored["correct"], 1)

    def test_scores_unordered_list_set(self):
        row = {
            "gold": ["red", "blue"],
            "answer_type": "list",
            "stage_d_cell": "list_set",
            "stale_values": ["green"],
        }

        scored = score_stage_d_row(row, "blue, red")

        self.assertEqual(scored["correct"], 1)

    def test_scores_ordered_latent_list(self):
        row = {
            "gold": ["red", "blue"],
            "answer_type": "list",
            "stage_d_cell": "latent_list",
            "stale_values": [],
        }

        self.assertEqual(score_stage_d_row(row, "red, blue")["correct"], 1)
        self.assertEqual(score_stage_d_row(row, "blue, red")["correct"], 0)

    def test_scores_dict_entity_values(self):
        row = {
            "gold": {"budget": 10, "server": 20},
            "answer_type": "dict",
            "stale_values": [99],
        }

        scored = score_stage_d_row(row, "budget=10, server=20")

        self.assertEqual(scored["correct"], 1)
        self.assertEqual(scored["n_correct"], 2)

    def test_build_adapter_args_requires_kind_with_dir(self):
        with self.assertRaises(ValueError):
            build_adapter_args("adapters/foo", None)

    def test_build_adapter_args_omits_empty_adapter(self):
        self.assertEqual(build_adapter_args(None, None), {})

    def test_summary_preserves_naturalistic_family_load_and_length(self):
        results = [
            {
                "task_family": "naturalistic",
                "stage_d_cell": "dialogue",
                "answer_type": "int",
                "condition": "interference",
                "interference_load": 4,
                "length_variant": "short",
                "correct": 1,
                "error_type": "correct",
            },
            {
                "task_family": "naturalistic",
                "stage_d_cell": "dialogue",
                "answer_type": "int",
                "condition": "interference",
                "interference_load": 4,
                "length_variant": "short",
                "correct": 0,
                "error_type": "stale",
            },
            {
                "task_family": "naturalistic",
                "stage_d_cell": "dialogue",
                "answer_type": "int",
                "condition": "simple",
                "interference_load": 0,
                "length_variant": "long",
                "correct": 1,
                "error_type": "correct",
            },
        ]

        summary = summarize_results(results, {"stage": "E"})

        cells = {
            (
                cell["stage_d_cell"],
                cell["condition"],
                cell["interference_load"],
                cell["length_variant"],
            ): cell
            for cell in summary["cells"]
        }
        natural_cell = cells[("dialogue", "interference", 4, "short")]
        self.assertEqual(natural_cell["n"], 2)
        self.assertEqual(natural_cell["accuracy"], 0.5)
        self.assertEqual(natural_cell["stale_rate"], 0.5)
        self.assertEqual(natural_cell["within_stale_rate_among_errors"], 1.0)
        self.assertEqual(cells[("dialogue", "simple", 0, "long")]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
