import tempfile
import unittest
from pathlib import Path

from check_extraction_gate import compare_to_anchor, compute_cells, load_anchor_cells


class StageAGateTest(unittest.TestCase):
    def test_compute_cells_from_index_rows(self):
        rows = [
            {"condition": "simple", "n_lines": 40, "interference_load": 0, "correct": 1},
            {"condition": "simple", "n_lines": 40, "interference_load": 0, "correct": 0},
            {"condition": "interference", "n_lines": 40, "interference_load": 4, "correct": 1},
        ]
        cells = compute_cells(rows)
        self.assertEqual(cells[("simple", 40, 0)]["n"], 2)
        self.assertAlmostEqual(cells[("simple", 40, 0)]["accuracy"], 0.5)
        self.assertAlmostEqual(cells[("interference", 40, 4)]["accuracy"], 1.0)

    def test_compare_to_anchor_enforces_delta_and_simple_floor(self):
        observed = {
            ("simple", 40, 0): {"accuracy": 0.96, "n": 150},
            ("interference", 40, 4): {"accuracy": 0.58, "n": 150},
        }
        anchor = {
            ("simple", 40, 0): {"accuracy": 0.9867, "n": 150},
            ("interference", 40, 4): {"accuracy": 0.6067, "n": 150},
        }
        result = compare_to_anchor(observed, anchor, tolerance=0.04, simple_floor=0.95)
        self.assertTrue(result["passed"])

        observed[("simple", 40, 0)]["accuracy"] = 0.94
        result = compare_to_anchor(observed, anchor, tolerance=0.04, simple_floor=0.95)
        self.assertFalse(result["passed"])
        self.assertIn("simple@40", result["failures"][0])


if __name__ == "__main__":
    unittest.main()
