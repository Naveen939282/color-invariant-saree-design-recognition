import tempfile
import unittest
from pathlib import Path

from PIL import Image

from public_kaggle_experiment.scripts.audit_kaggle_dataset import (
    EXPECTED_CLASSES,
    create_metadata,
    discover_classes,
    find_exact_duplicate_groups,
    find_split_overlaps,
    normalize_source_filename,
)


class DatasetAuditTests(unittest.TestCase):
    def test_source_filename_normalization_removes_only_roboflow_suffix(self):
        self.assertEqual(
            normalize_source_filename("Image6_jpeg.rf." + "a" * 32 + ".jpg"),
            "image6.jpeg",
        )
        self.assertEqual(
            normalize_source_filename("image31_jpg.rf." + "b" * 32 + ".jpg"),
            "image31.jpg",
        )

    def test_metadata_creation_reads_image_and_assigns_candidate_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                for class_name in EXPECTED_CLASSES:
                    (root / split / class_name).mkdir(parents=True)
            image = Image.new("RGB", (8, 6), (40, 80, 120))
            names = ["sample_jpg.rf." + char * 32 + ".jpg" for char in "abc"]
            for split, name in zip(("train", "train", "valid"), names):
                image.save(root / split / "Banarasi" / name)
            rows, _ = create_metadata(root)
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["source_variant_count"] for row in rows}, {3})
            self.assertEqual({row["width"] for row in rows}, {8})
            self.assertEqual({row["height"] for row in rows}, {6})
            self.assertEqual({row["mode"] for row in rows}, {"RGB"})
            self.assertEqual({row["normalized_source_filename"] for row in rows}, {"sample.jpg"})

    def test_duplicate_detection_groups_exact_md5_only(self):
        rows = [
            {"md5": "same", "relative_path": "a.jpg"},
            {"md5": "same", "relative_path": "b.jpg"},
            {"md5": "different", "relative_path": "c.jpg"},
        ]
        groups = find_exact_duplicate_groups(rows)
        self.assertEqual(list(groups), ["same"])
        self.assertEqual(len(groups["same"]), 2)

    def test_split_overlap_detection_finds_shared_keys(self):
        rows = [
            {"source_id": "one", "split": "train"},
            {"source_id": "one", "split": "valid"},
            {"source_id": "two", "split": "test"},
        ]
        overlaps = find_split_overlaps(rows, "source_id")
        self.assertEqual(overlaps["train/valid"], ["one"])
        self.assertEqual(overlaps["train/test"], [])

    def test_expected_class_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                for class_name in EXPECTED_CLASSES:
                    (root / split / class_name).mkdir(parents=True)
            self.assertEqual(discover_classes(root), set(EXPECTED_CLASSES))


if __name__ == "__main__":
    unittest.main()