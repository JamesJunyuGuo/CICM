import unittest

import torch

from gen_paraphrase import make_paraphrase_example
from head_adapters import HeadSliceLoRALinear
from head_temperature_arm import TemperatureContext, apply_temperature
from adapter_eval import score_parallel, score_single_int


class StageCTest(unittest.TestCase):
    def test_paraphrase_generator_preserves_metadata_contract(self):
        row = make_paraphrase_example("interference", n_lines=12, I=3, rng_seed=123)

        self.assertEqual(row["condition"], "interference")
        self.assertEqual(row["task_family"], "paraphrase")
        self.assertEqual(len(row["stale_values"]), 3)
        self.assertEqual(len(row["target_write_positions"]), 4)
        self.assertIn(str(row["gold"]), row["prompt"])
        self.assertIn(row["target_var"], row["prompt"])

    def test_single_int_scoring_uses_first_integer(self):
        row = {"gold": 42}
        scored = score_single_int(row, "The answer is 42, not 17.")

        self.assertEqual(scored["pred"], 42)
        self.assertEqual(scored["correct"], 1)

    def test_parallel_scoring_reports_per_stream_and_all_correct(self):
        row = {"gold": {"A": 10, "B": 20, "C": 30}}
        scored = score_parallel(row, "A=10, B=99, C=30")

        self.assertEqual(scored["n_correct"], 2)
        self.assertEqual(scored["n_streams"], 3)
        self.assertEqual(scored["all_correct"], 0)

    def test_temperature_tau_one_is_bitwise_identity(self):
        scores = torch.randn((1, 4, 2, 5))
        ctx = TemperatureContext(tau=1.0, heads_by_layer={2: (1, 3)})

        out = apply_temperature(scores, layer_idx=2, ctx=ctx)

        self.assertIs(out, scores)
        self.assertTrue(torch.equal(out, scores))

    def test_temperature_targets_only_selected_heads(self):
        scores = torch.ones((1, 4, 2, 5))
        ctx = TemperatureContext(tau=0.5, heads_by_layer={2: (1, 3)})

        out = apply_temperature(scores, layer_idx=2, ctx=ctx)

        expected = torch.ones_like(scores)
        expected[:, [1, 3], :, :] = 2.0
        self.assertTrue(torch.equal(out, expected))

    def test_head_slice_lora_adapter_off_is_identity(self):
        base = torch.nn.Linear(3, 5, bias=False)
        wrapped = HeadSliceLoRALinear(base, selected_rows=[1, 4], rank=2, alpha=4)
        x = torch.randn((2, 3))

        out = wrapped(x)

        self.assertTrue(torch.equal(out, base(x)))

    def test_head_slice_lora_parameters_follow_base_device(self):
        base = torch.nn.Linear(3, 5, bias=False)
        wrapped = HeadSliceLoRALinear(base, selected_rows=[1, 4], rank=2, alpha=4)

        self.assertEqual(wrapped.lora_a.device, base.weight.device)
        self.assertEqual(wrapped.lora_b.device, base.weight.device)
        self.assertEqual(wrapped.row_index.device, base.weight.device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_head_slice_lora_cuda_forward_uses_one_device(self):
        base = torch.nn.Linear(3, 5, bias=False).cuda()
        wrapped = HeadSliceLoRALinear(base, selected_rows=[1, 4], rank=2, alpha=4)
        x = torch.randn((2, 3), device="cuda")

        out = wrapped(x)

        self.assertEqual(out.device.type, "cuda")

    def test_head_slice_lora_updates_only_selected_output_rows(self):
        base = torch.nn.Linear(3, 5, bias=False)
        wrapped = HeadSliceLoRALinear(base, selected_rows=[1, 4], rank=1, alpha=1)
        with torch.no_grad():
            wrapped.lora_a.fill_(1.0)
            wrapped.lora_b.fill_(1.0)
        x = torch.ones((2, 3))

        delta = wrapped(x) - base(x)

        self.assertTrue(torch.equal(delta[:, [0, 2, 3]], torch.zeros((2, 3))))
        self.assertTrue(torch.equal(delta[:, [1, 4]], torch.full((2, 2), 3.0)))


if __name__ == "__main__":
    unittest.main()
