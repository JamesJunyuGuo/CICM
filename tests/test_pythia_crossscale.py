import sys
import unittest

import numpy as np

sys.path.insert(0, "/anvil/scratch/x-jguo7/agent_eval/Contextual_management/src")

from pythia_crossscale import (
    adjudicate_behavior_rows,
    rank_stale_promoting_heads,
    summarize_induction_overlap,
)


def make_rows(k0_correct=True, all_overwrite_easy=False):
    rows = []
    for k in (0, 2, 4):
        for seed in (11, 29, 47):
            for template in ("arrow", "current", "latest"):
                for index in range(72):
                    if k == 0:
                        correct = k0_correct or index > 7
                    elif all_overwrite_easy:
                        correct = index < 65
                    else:
                        correct = index < (50 if k == 2 else 32)
                    rows.append(
                        {
                            "id": f"{k}-{seed}-{template}-{index}",
                            "semantic_id": f"s-{seed}-{template}-{index}",
                            "k": k,
                            "seed": seed,
                            "template": template,
                            "variant": "single",
                            "label": "correct_current" if correct else "within_stale",
                        }
                    )
    return rows


class CrossScaleTests(unittest.TestCase):
    def test_behavior_gate_selects_powered_k(self):
        verdict = adjudicate_behavior_rows(make_rows(), model="m")
        self.assertEqual(verdict["status"], "mechanism_ready")
        self.assertTrue(verdict["k0_gate"]["pass"])
        self.assertEqual(verdict["chosen_k"], 4)

    def test_behavior_gate_reports_infeasible_at_cap(self):
        verdict = adjudicate_behavior_rows(
            make_rows(all_overwrite_easy=True), model="m"
        )
        self.assertEqual(verdict["status"], "substrate_infeasible_at_cap")

    def test_stale_ranking_uses_discovery_only(self):
        labels = np.asarray(
            ["correct_current", "within_stale", "correct_current", "within_stale"]
        )
        splits = np.asarray(["discovery", "discovery", "evaluation", "evaluation"])
        stale_ratio = np.asarray([[[0.1, 0.2]], [[0.8, 0.3]], [[0.9, 0.1]], [[0.0, 1.0]]])
        head_dla = np.asarray([[[-1.0, -1.0]], [[-1.0, -1.0]], [[-1.0, -1.0]], [[-1.0, -1.0]]])
        ranking = rank_stale_promoting_heads(stale_ratio, head_dla, labels, splits)
        self.assertEqual((ranking[0]["layer"], ranking[0]["head"]), (0, 0))

    def test_induction_overlap_is_top_decile_fraction(self):
        summary = summarize_induction_overlap(
            {"circuit_heads": [{"percentile_among_heads": 0.95}, {"percentile_among_heads": 0.2}]}
        )
        self.assertEqual(summary["top_decile_count"], 1)
        self.assertEqual(summary["top_decile_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
