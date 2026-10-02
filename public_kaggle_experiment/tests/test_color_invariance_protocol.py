import tempfile
import unittest
from pathlib import Path

from PIL import Image

from public_kaggle_experiment.scripts.create_canonical_images import select_canonical_rows
from public_kaggle_experiment.scripts.create_color_invariance_manifest import (
    build_test_gallery,
    build_test_verification_pairs,
    build_transform_manifest,
    read_canonical_rows,
)
from public_kaggle_experiment.src.color_transforms import (
    TRANSFORMATION_CONFIG,
    TRANSFORMATION_NAMES,
    apply_color_transform,
)


def canonical_fixture():
    return [
        {
            "identity_id": "banarasi::silk-a.jpg",
            "source_id": "banarasi::silk-a.jpg",
            "class_name": "Banarasi",
            "split": "train",
            "canonical_path": "train/Banarasi/silk-a.jpg",
            "canonical_filename": "silk-a.jpg",
            "canonical_md5": "a" * 32,
            "original_group_size": "1",
            "selection_rule": "lexicographically_smallest_relative_path",
        },
        {
            "identity_id": "banarasi::silk-b.jpg",
            "source_id": "banarasi::silk-b.jpg",
            "class_name": "Banarasi",
            "split": "test",
            "canonical_path": "test/Banarasi/silk-b.jpg",
            "canonical_filename": "silk-b.jpg",
            "canonical_md5": "b" * 32,
            "original_group_size": "3",
            "selection_rule": "lexicographically_smallest_relative_path",
        },
        {
            "identity_id": "ikat::woven-c.jpg",
            "source_id": "ikat::woven-c.jpg",
            "class_name": "Ikat",
            "split": "test",
            "canonical_path": "test/Ikat/woven-c.jpg",
            "canonical_filename": "woven-c.jpg",
            "canonical_md5": "c" * 32,
            "original_group_size": "1",
            "selection_rule": "lexicographically_smallest_relative_path",
        },
    ]


class ColorTransformTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (17, 11))
        for x in range(self.image.width):
            for y in range(self.image.height):
                self.image.putpixel((x, y), (x * 13, y * 19, (x + y) * 7))

    def test_transformations_are_deterministic(self):
        for name in TRANSFORMATION_NAMES:
            with self.subTest(transformation=name):
                first = apply_color_transform(self.image, name)
                second = apply_color_transform(self.image, name)
                self.assertEqual(first.tobytes(), second.tobytes())

    def test_valid_names_and_unknown_name_rejection(self):
        self.assertEqual(len(TRANSFORMATION_NAMES), 11)
        self.assertEqual(set(TRANSFORMATION_NAMES), set(TRANSFORMATION_CONFIG))
        with self.assertRaisesRegex(ValueError, "Unknown transformation"):
            apply_color_transform(self.image, "random_color")

    def test_grayscale_is_three_channel_rgb_and_all_transforms_preserve_size(self):
        grayscale = apply_color_transform(self.image, "grayscale")
        self.assertEqual(grayscale.mode, "RGB")
        self.assertEqual(len(grayscale.getbands()), 3)
        for name in TRANSFORMATION_NAMES:
            with self.subTest(transformation=name):
                transformed = apply_color_transform(self.image, name)
                self.assertEqual(transformed.size, self.image.size)
                self.assertEqual(transformed.mode, "RGB")


class CanonicalManifestTests(unittest.TestCase):
    def metadata_rows(self):
        records = []
        for path, source_id, split, class_name, group_size in (
            ("train/Banarasi/z.jpg", "banarasi::one.jpg", "train", "Banarasi", 3),
            ("train/Banarasi/a.jpg", "banarasi::one.jpg", "train", "Banarasi", 3),
            ("train/Banarasi/m.jpg", "banarasi::one.jpg", "train", "Banarasi", 3),
            ("test/Banarasi/same-class.jpg", "banarasi::two.jpg", "test", "Banarasi", 1),
        ):
            records.append(
                {
                    "source_id": source_id,
                    "class_name": class_name,
                    "split": split,
                    "relative_path": path,
                    "filename": Path(path).name,
                    "md5": "d" * 32,
                    "source_variant_count": str(group_size),
                }
            )
        return records

    def test_canonical_selection_is_lexicographic_and_deterministic(self):
        records = self.metadata_rows()
        selected = select_canonical_rows(records)
        selected_again = select_canonical_rows(reversed(records))
        self.assertEqual(selected, selected_again)
        first = next(row for row in selected if row["source_id"] == "banarasi::one.jpg")
        self.assertEqual(first["canonical_path"], "train/Banarasi/a.jpg")
        self.assertEqual(first["original_group_size"], "3")

    def test_one_candidate_identity_per_group_and_class_is_not_identity(self):
        selected = select_canonical_rows(self.metadata_rows())
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["identity_id"] for row in selected}), 2)
        same_class = [row for row in selected if row["class_name"] == "Banarasi"]
        self.assertEqual(len(same_class), 2)
        self.assertNotEqual(same_class[0]["identity_id"], same_class[1]["identity_id"])
        self.assertTrue(all(row["identity_id"] == row["source_id"] for row in selected))

    def test_candidate_identity_cross_split_is_rejected(self):
        rows = self.metadata_rows()
        conflicting = dict(rows[0], split="test")
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            select_canonical_rows(rows + [conflicting])

    def test_manifest_has_each_expected_transformation_per_identity(self):
        canonical = canonical_fixture()
        manifest = build_transform_manifest(canonical)
        self.assertEqual(len(manifest), len(canonical) * len(TRANSFORMATION_NAMES))
        for identity in canonical:
            rows = [row for row in manifest if row["identity_id"] == identity["identity_id"]]
            self.assertEqual({row["transformation"] for row in rows}, set(TRANSFORMATION_NAMES))
            self.assertTrue(all(row["identity_id"] != row["class_name"] for row in rows))
            self.assertTrue(all(row["transformed_path"] == "" for row in rows))

    def test_manifest_rejects_duplicate_canonical_identity_rows(self):
        rows = canonical_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.csv"
            import csv

            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([rows[0], rows[0]])
            with self.assertRaisesRegex(ValueError, "more than one row"):
                read_canonical_rows(path)

    def test_test_gallery_and_verification_pairs_use_candidate_identity_only(self):
        canonical = canonical_fixture()
        manifest = build_transform_manifest(canonical)
        test_queries = [row for row in manifest if row["split"] == "test"]
        gallery = build_test_gallery(canonical)
        pairs = build_test_verification_pairs(test_queries, gallery)
        self.assertEqual(len(gallery), 2)
        self.assertEqual(len(pairs), len(test_queries) * len(gallery))
        for pair in pairs:
            expected = pair["query_identity_id"] == pair["gallery_identity_id"]
            self.assertEqual(pair["same_identity"] == "1", expected)
            self.assertEqual(pair["pair_type"], "positive" if expected else "negative")


if __name__ == "__main__":
    unittest.main()