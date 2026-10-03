import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figs import save_bundle


class FigsTest(unittest.TestCase):
    def test_save_bundle_writes_png_pdf_data_and_manifest_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            fig, ax = plt.subplots()
            ax.plot([1, 2], [0.5, 0.7])

            entry = save_bundle(
                out_dir=out_dir,
                name="F_test",
                fig=fig,
                data={"x": [1, 2], "y": [0.5, 0.7]},
                description="test figure",
                sources=["a.json"],
            )

            self.assertTrue((out_dir / "F_test.png").exists())
            self.assertTrue((out_dir / "F_test.pdf").exists())
            self.assertEqual(
                json.loads((out_dir / "F_test.data.json").read_text()),
                {"x": [1, 2], "y": [0.5, 0.7]},
            )
            self.assertIn("F_test", entry)
            self.assertIn("a.json", entry)


if __name__ == "__main__":
    unittest.main()
