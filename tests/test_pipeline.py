"""Synthetic integration tests; all images are generated in temporary folders."""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from scripts.generate_metadata import main as generate_metadata_main
from scripts.create_splits import main as create_splits_main
from scripts.split_utils import load_labeled_dataset, validate_metadata_only
from scripts.validate_dataset import main as validate_dataset_main


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.deeplure_root = self.root / "deeplure"
        self.kaggle_root = self.root / "kaggle"
        self.deeplure_root.mkdir()
        self.kaggle_root.mkdir()
        for design_number in range(1, 7):
            for colorway, color in (("red", (190, 30, 40)), ("blue", (30, 70, 190)), ("green", (30, 150, 80))):
                image_path = self.deeplure_root / f"D{design_number:03d}" / f"{colorway}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (48, 48), color)
                draw = ImageDraw.Draw(image)
                if design_number % 2:
                    draw.ellipse((10, 10, 38, 38), outline="white", width=3)
                else:
                    draw.rectangle((10, 10, 38, 38), outline="white", width=3)
                draw.text((18, 18), str(design_number), fill="white")
                image.save(image_path)
        shutil.copyfile(
            self.deeplure_root / "D001" / "red.png", self.kaggle_root / "copy.png"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_metadata_duplicate_report_and_blank_template(self) -> None:
        metadata_dir = self.root / "combined_metadata"
        reports_dir = self.root / "combined_reports"
        labels_file = metadata_dir / "design_labels.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            generate_metadata_main(
                [
                    "--deeplure-root", str(self.deeplure_root),
                    "--kaggle-root", str(self.kaggle_root),
                    "--metadata-dir", str(metadata_dir),
                    "--reports-dir", str(reports_dir),
                    "--labels-file", str(labels_file),
                ]
            )
        metadata = pd.read_csv(metadata_dir / "image_metadata.csv")
        duplicates = pd.read_csv(reports_dir / "duplicate_report.csv")
        labels = pd.read_csv(metadata_dir / "design_labels_template.csv")
        self.assertEqual(len(metadata), 19)
        self.assertEqual(set(metadata["source_dataset"]), {"deeplure", "kaggle"})
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(int(duplicates.iloc[0]["duplicate_count"]), 2)
        self.assertTrue(labels["design_id"].isna().all())
        with self.assertRaisesRegex(ValueError, "Exact duplicate"):
            validate_metadata_only(
                metadata_dir / "image_metadata.csv",
                {"deeplure": self.deeplure_root, "kaggle": self.kaggle_root},
            )

    def test_synthetic_labels_generate_leakage_free_splits(self) -> None:
        metadata_dir = self.root / "metadata"
        reports_dir = self.root / "reports"
        with contextlib.redirect_stdout(io.StringIO()):
            generate_metadata_main(
                [
                    "--deeplure-root", str(self.deeplure_root),
                    "--metadata-dir", str(metadata_dir),
                    "--reports-dir", str(reports_dir),
                ]
            )
        labels_path = metadata_dir / "design_labels.csv"
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        labels["design_id"] = labels["relative_path"].str.split("/").str[0]
        labels["colorway"] = labels["relative_path"].str.split("/").str[1].str.removesuffix(".png")
        labels["label_confidence"] = "high"
        labels["label_source"] = "synthetic test annotation"
        labels["notes"] = "Generated test-only motif assignment"
        labels.to_csv(labels_path, index=False)

        dataset = load_labeled_dataset(
            metadata_dir / "image_metadata.csv", labels_path, None, self.deeplure_root, None
        )
        self.assertEqual(
            validate_metadata_only(
                metadata_dir / "image_metadata.csv",
                {"deeplure": self.deeplure_root},
            ),
            [],
        )
        split_dir = self.root / "splits"
        with contextlib.redirect_stdout(io.StringIO()):
            split_result = create_splits_main(
                [
                    "--metadata", str(metadata_dir / "image_metadata.csv"),
                    "--labels", str(labels_path),
                    "--deeplure-root", str(self.deeplure_root),
                    "--output-dir", str(split_dir),
                    "--seed", "42",
                    "--evaluation-fraction", "0.3",
                ]
            )
            validation_result = validate_dataset_main(
                [
                    "--metadata", str(metadata_dir / "image_metadata.csv"),
                    "--labels", str(labels_path),
                    "--deeplure-root", str(self.deeplure_root),
                    "--split-dir", str(split_dir),
                ]
            )
        self.assertEqual(split_result, 0)
        self.assertEqual(validation_result, 0)
        for filename in (
            "train.csv",
            "validation.csv",
            "gallery.csv",
            "query.csv",
            "verification_pairs.csv",
        ):
            self.assertTrue((split_dir / filename).is_file())
        pairs = pd.read_csv(split_dir / "verification_pairs.csv")
        self.assertGreater((pairs["label"] == 1).sum(), 0)
        self.assertGreater((pairs["label"] == 0).sum(), 0)
        self.assertTrue((pairs["color_relationship"] == "different").any())

    def test_blank_design_ids_fail_loudly(self) -> None:
        metadata_dir = self.root / "metadata"
        reports_dir = self.root / "reports"
        with contextlib.redirect_stdout(io.StringIO()):
            generate_metadata_main(
                [
                    "--deeplure-root", str(self.deeplure_root),
                    "--metadata-dir", str(metadata_dir),
                    "--reports-dir", str(reports_dir),
                ]
            )
        with self.assertRaisesRegex(ValueError, "Blank design IDs"):
            load_labeled_dataset(
                metadata_dir / "image_metadata.csv",
                metadata_dir / "design_labels.csv",
                None,
                self.deeplure_root,
                None,
            )


if __name__ == "__main__":
    unittest.main()