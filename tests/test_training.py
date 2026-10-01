"""CPU smoke tests for contrastive training artifact generation."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.train_embedding import main as train_main


class TrainingSmokeTests(unittest.TestCase):
    def test_one_epoch_writes_checkpoint_and_reproducibility_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "deeplure"
            image_root.mkdir()
            rows = []
            for design_id, color, color_rgb in (
                ("D001", "red", (180, 40, 50)),
                ("D001", "blue", (30, 50, 180)),
                ("D002", "red", (180, 40, 50)),
                ("D002", "blue", (30, 50, 180)),
            ):
                relative_path = f"{design_id}_{color}.png"
                Image.new("RGB", (72, 72), color_rgb).save(image_root / relative_path)
                rows.append(
                    {
                        "image_id": f"{design_id}_{color}",
                        "relative_path": relative_path,
                        "source_dataset": "deeplure",
                        "design_id": design_id,
                        "colorway": color,
                    }
                )
            split_path = root / "split.csv"
            pd.DataFrame(rows).to_csv(split_path, index=False)
            output = root / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                result = train_main(
                    [
                        "--train-csv", str(split_path),
                        "--validation-csv", str(split_path),
                        "--deeplure-root", str(image_root),
                        "--output-dir", str(output),
                        "--epochs", "1",
                        "--batch-size", "2",
                        "--pairs-per-epoch", "4",
                        "--validation-pairs", "4",
                        "--embedding-dim", "16",
                        "--image-size", "64",
                        "--device", "cpu",
                        "--no-pretrained",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "training_config.json").is_file())
            self.assertTrue((output / "training_history.csv").is_file())


if __name__ == "__main__":
    unittest.main()