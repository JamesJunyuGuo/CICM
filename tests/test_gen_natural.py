import json
import tempfile
import unittest
from pathlib import Path

from gen_natural import make_natural_rows, write_natural_outputs


class GenNaturalTest(unittest.TestCase):
    def test_generation_is_deterministic_and_covers_grid(self):
        rows_a = make_natural_rows(n_per_cell=2, seed=20)
        rows_b = make_natural_rows(n_per_cell=2, seed=20)

        self.assertEqual(rows_a, rows_b)
        self.assertEqual(len(rows_a), 3 * (4 * 2 + 2) * 2)
        self.assertEqual(
            {row["natural_family"] for row in rows_a},
            {"dialogue", "agent_log", "document"},
        )
        self.assertEqual(
            {
                (row["natural_family"], row["condition"], row["interference_load"], row["length_variant"])
                for row in rows_a
            },
            {
                (family, "interference", load, length)
                for family in ["dialogue", "agent_log", "document"]
                for load in [1, 2, 4, 8]
                for length in ["short", "long"]
            }
            | {
                (family, "simple", 0, length)
                for family in ["dialogue", "agent_log", "document"]
                for length in ["short", "long"]
            },
        )

    def test_rows_preserve_stale_binding_metadata_and_design_rules(self):
        rows = make_natural_rows(n_per_cell=1, seed=7)

        for row in rows:
            with self.subTest(row_id=row["id"]):
                self.assertEqual(row["task_family"], "naturalistic")
                self.assertEqual(row["stage_d_cell"], row["natural_family"])
                self.assertEqual(row["answer_type"], "int")
                self.assertIn("prompt", row)
                self.assertIn("Question:", row["prompt"])
                self.assertIsInstance(row["gold"], int)
                self.assertEqual(len(str(row["gold"])), 4)
                self.assertEqual(len(set(row["all_values"])), len(row["all_values"]))
                self.assertIn(row["gold"], row["all_values"])
                self.assertEqual(set(row["stale_values"]).isdisjoint({row["gold"]}), True)
                self.assertEqual(len(row["target_update_positions"]), len(row["stale_values"]) + 1)
                self.assertEqual(len(row["target_update_char_spans"]), len(row["target_update_positions"]))
                self.assertGreaterEqual(row["tail_units_after_final_update"], 3)

                expected_values = [str(v) for v in row["stale_values"] + [row["gold"]]]
                for span, expected in zip(row["target_update_char_spans"], expected_values):
                    self.assertEqual(row["prompt"][span["char_start"] : span["char_end"]], expected)

    def test_write_outputs_manifest_and_family_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_natural_outputs(Path(tmp), n_per_cell=1, seed=13)

            self.assertEqual(manifest["stage"], "E")
            self.assertEqual(manifest["track"], "naturalistic")
            self.assertTrue((Path(tmp) / "eval_all.jsonl").exists())
            for family in ["dialogue", "agent_log", "document"]:
                self.assertTrue((Path(tmp) / f"eval_{family}.jsonl").exists())

            rows = [
                json.loads(line)
                for line in (Path(tmp) / "eval_all.jsonl").read_text().splitlines()
            ]
            self.assertEqual(manifest["n_total"], len(rows))


if __name__ == "__main__":
    unittest.main()
