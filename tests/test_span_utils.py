import json
import os
import random
import unittest

from transformers import AutoTokenizer

from gen_tasks import make_example
from span_utils import build_chat_prompt, map_target_value_spans


MODEL = os.environ.get("SPAN_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LLAMA_MODEL = os.environ.get("SPAN_TEST_LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


class SpanUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def test_target_value_spans_decode_to_exact_values(self):
        rng = random.Random(123)
        rows = [
            make_example("simple", 40, 0, rng),
            make_example("interference", 40, 2, rng),
            make_example("interference", 40, 4, rng),
            make_example("interference", 80, 4, rng),
            make_example("interference", 40, 8, rng),
        ]

        for idx, row in enumerate(rows):
            with self.subTest(idx=idx, I=row["interference_load"]):
                mapped = map_target_value_spans(row, self.tokenizer)
                templated = build_chat_prompt(row["prompt"], self.tokenizer)
                input_ids = self.tokenizer(templated)["input_ids"]
                expected_values = [str(v) for v in row["stale_values"] + [row["gold"]]]
                self.assertEqual(len(mapped["writes"]), len(expected_values))

                for write, expected in zip(mapped["writes"], expected_values):
                    span_ids = input_ids[write["token_start"] : write["token_end"]]
                    decoded = self.tokenizer.decode(span_ids)
                    self.assertEqual(decoded, expected)

    def test_target_value_spans_decode_with_llama_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        rng = random.Random(321)
        row = make_example("interference", 40, 4, rng)

        mapped = map_target_value_spans(row, tokenizer)
        templated = build_chat_prompt(row["prompt"], tokenizer)
        input_ids = tokenizer(templated)["input_ids"]

        expected_values = [str(v) for v in row["stale_values"] + [row["gold"]]]
        self.assertEqual(len(mapped["writes"]), len(expected_values))
        for write, expected in zip(mapped["writes"], expected_values):
            span_ids = input_ids[write["token_start"] : write["token_end"]]
            decoded = tokenizer.decode(span_ids)
            self.assertEqual(decoded, expected)


if __name__ == "__main__":
    unittest.main()
