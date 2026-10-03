import torch
import unittest

from extract_overwrite_attention import (
    classify_prediction,
    filter_rows_by_index_order,
    is_stage_a_row,
    resolve_torch_dtype,
)


class ExtractStageATest(unittest.TestCase):
    def test_stage_a_row_filter_matches_spec_cells(self):
        keep = [
            {"condition": "interference", "n_lines": 40, "interference_load": 2},
            {"condition": "interference", "n_lines": 40, "interference_load": 4},
            {"condition": "interference", "n_lines": 40, "interference_load": 8},
            {"condition": "interference", "n_lines": 20, "interference_load": 4},
            {"condition": "interference", "n_lines": 80, "interference_load": 4},
            {"condition": "simple", "n_lines": 40, "interference_load": 0},
        ]
        drop = [
            {"condition": "simple", "n_lines": 20, "interference_load": 0},
            {"condition": "simple", "n_lines": 80, "interference_load": 0},
            {"condition": "interference", "n_lines": 20, "interference_load": 2},
            {"condition": "interference", "n_lines": 80, "interference_load": 8},
        ]
        self.assertTrue(all(is_stage_a_row(row) for row in keep))
        self.assertFalse(any(is_stage_a_row(row) for row in drop))

    def test_classify_prediction_records_answered_stale_write(self):
        row = {"gold": 44, "stale_values": [11, 22, 33]}
        label = classify_prediction(row, 22)
        self.assertFalse(label["correct"])
        self.assertEqual(label["answer_type"], "stale")
        self.assertEqual(label["answered_write_idx"], 1)

        self.assertEqual(classify_prediction(row, 44)["answer_type"], "correct")
        self.assertEqual(classify_prediction(row, 99)["answer_type"], "other")

    def test_resolve_torch_dtype_defaults_to_float32_reference(self):
        self.assertIs(resolve_torch_dtype("float32"), torch.float32)
        self.assertIs(resolve_torch_dtype("bfloat16"), torch.bfloat16)

    def test_filter_rows_by_index_order_preserves_reference_order(self):
        rows = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
        index_rows = [{"id": "a"}, {"id": "c"}]

        filtered = filter_rows_by_index_order(rows, index_rows)

        self.assertEqual([row["id"] for row in filtered], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
