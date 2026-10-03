import json
import tempfile
import unittest
from pathlib import Path

from summarize_natural_transfer import summarize_arm


class SummarizeNaturalTransferTest(unittest.TestCase):
    def test_summarize_arm_pools_high_i_by_family(self):
        summary = {
            "cells": [
                {
                    "stage_d_cell": "dialogue",
                    "condition": "interference",
                    "interference_load": 4,
                    "n": 100,
                    "accuracy": 0.8,
                    "stale_rate": 0.2,
                },
                {
                    "stage_d_cell": "dialogue",
                    "condition": "interference",
                    "interference_load": 8,
                    "n": 100,
                    "accuracy": 0.6,
                    "stale_rate": 0.4,
                },
                {
                    "stage_d_cell": "dialogue",
                    "condition": "interference",
                    "interference_load": 2,
                    "n": 100,
                    "accuracy": 1.0,
                    "stale_rate": 0.0,
                },
            ]
        }

        out, _ = summarize_arm(summary, {4, 8})

        self.assertAlmostEqual(out["dialogue"]["accuracy"], 0.7)
        self.assertAlmostEqual(out["dialogue"]["stale_rate"], 0.3)
        self.assertAlmostEqual(out["dialogue"]["within_stale_rate_among_errors"], 1.0)


if __name__ == "__main__":
    unittest.main()
