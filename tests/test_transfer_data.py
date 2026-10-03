import json
import tempfile
import unittest
from pathlib import Path

from build_transfer_data import make_stage_d_rows, write_stage_d_outputs
from check_transfer_data import check_rows


class StageDDataTest(unittest.TestCase):
    def test_stage_d_generation_is_deterministic(self):
        rows_a = make_stage_d_rows(n_per_cell=2, seed=17)
        rows_b = make_stage_d_rows(n_per_cell=2, seed=17)

        self.assertEqual(rows_a, rows_b)

    def test_stage_d_rows_have_required_schema_and_valid_spans(self):
        rows = make_stage_d_rows(n_per_cell=1, seed=3)

        errors = check_rows(rows)

        self.assertEqual(errors, [])
        for row in rows:
            self.assertIn("prompt", row)
            self.assertIn("gold", row)
            self.assertIn("current_spans", row)
            self.assertIn("stale_spans", row)
            self.assertIn("answer_type", row)
            lines = row["context_lines"]
            for span in row["current_spans"] + row["stale_spans"]:
                self.assertGreaterEqual(span["line"], 0)
                self.assertLess(span["line"], len(lines))

    def test_stage_d_covers_broader_context_management_families(self):
        rows = make_stage_d_rows(n_per_cell=1, seed=5)
        families = {row["task_family"] for row in rows}

        self.assertEqual(
            families,
            {
                "ruler_style",
                "babilong_style",
                "michelangelo_style",
                "state_suite",
            },
        )
        cells = {row["stage_d_cell"] for row in rows}
        self.assertIn("multi_hop_tracing", cells)
        self.assertIn("list_set", cells)
        self.assertIn("multi_round_coreference", cells)
        self.assertIn("implicit_invalidation", cells)

    def test_stage_d_outputs_manifest_and_all_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = write_stage_d_outputs(out_dir, n_per_cell=1, seed=11)

            all_rows = [
                json.loads(line)
                for line in (out_dir / "eval_all.jsonl").read_text().splitlines()
            ]

            self.assertEqual(manifest["n_total"], len(all_rows))
            self.assertTrue((out_dir / "eval_ruler_style.jsonl").exists())
            self.assertTrue((out_dir / "eval_babilong_style.jsonl").exists())
            self.assertTrue((out_dir / "eval_michelangelo_style.jsonl").exists())
            self.assertTrue((out_dir / "eval_state_suite.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
