"""Unit tests for embedding data, transforms, model, and metric-learning loss."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from scripts.embedding_dataset import (
    ContrastivePairDataset,
    SareeImageDataset,
    build_transform,
)
from scripts.embedding_model import (
    GrayscaleResNet18Baseline,
    ResNet18Embedding,
    cosine_similarity,
    contrastive_loss,
)
from scripts.train_embedding import seed_everything


class EmbeddingPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = []
        for design_id, colorways in (("D001", ("red", "blue")), ("D002", ("red", "blue"))):
            for colorway in colorways:
                relative_path = f"{design_id}/{colorway}.png"
                path = self.root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (72, 60), (190, 30, 40) if colorway == "red" else (30, 60, 190)).save(path)
                self.rows.append(
                    {
                        "image_id": f"{design_id}_{colorway}",
                        "relative_path": relative_path,
                        "source_dataset": "deeplure",
                        "design_id": design_id,
                        "colorway": colorway,
                    }
                )
        self.csv_path = self.root / "split.csv"
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_grayscale_and_color_transforms_return_rgb_tensor(self) -> None:
        image = Image.new("RGB", (72, 60), (150, 50, 20))
        for mode in ("grayscale", "color_aug"):
            tensor = build_transform(mode, train=True, image_size=64)(image)
            self.assertEqual(tuple(tensor.shape), (3, 64, 64))
            self.assertTrue(torch.isfinite(tensor).all())

    def test_dataset_loads_image_using_existing_root_resolver(self) -> None:
        dataset = SareeImageDataset(
            self.csv_path,
            transform=transforms.ToTensor(),
            deeplure_root=self.root,
        )
        item = dataset[0]
        self.assertEqual(tuple(item["image"].shape), (3, 60, 72))
        self.assertEqual(item["design_id"], "D001")
        self.assertEqual(item["colorway"], "red")

    def test_path_traversal_is_rejected(self) -> None:
        bad_frame = pd.DataFrame(
            [{**self.rows[0], "relative_path": "../outside.png"}]
        )
        bad_path = self.root / "bad.csv"
        bad_frame.to_csv(bad_path, index=False)
        dataset = SareeImageDataset(
            bad_path, transform=transforms.ToTensor(), deeplure_root=self.root
        )
        with self.assertRaisesRegex(ValueError, "escapes configured data root"):
            dataset[0]

    def test_resnet_embedding_is_fixed_dimensional_and_l2_normalized(self) -> None:
        model = ResNet18Embedding(embedding_dim=32, pretrained=False, freeze_backbone=True)
        output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 32))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        model.set_backbone_trainable(True)
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))

    def test_grayscale_baseline_emits_normalized_512d_features(self) -> None:
        model = GrayscaleResNet18Baseline(pretrained=False).eval()
        output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 512))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_cosine_similarity_and_contrastive_loss(self) -> None:
        first = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        second = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
        similarity = cosine_similarity(first, second)
        self.assertTrue(torch.allclose(similarity, torch.tensor([1.0, 0.8]), atol=1e-5))
        loss = contrastive_loss(first, second, torch.tensor([1.0, 0.0]), margin=1.0)
        self.assertAlmostEqual(float(loss), 0.0675, places=3)

    def test_pair_sampling_is_deterministic_for_same_seed_and_epoch(self) -> None:
        dataset = SareeImageDataset(
            self.csv_path, transform=transforms.ToTensor(), deeplure_root=self.root
        )
        first = ContrastivePairDataset(dataset, pairs_per_epoch=8, seed=17)
        second = ContrastivePairDataset(dataset, pairs_per_epoch=8, seed=17)
        first_item = first[0]
        second_item = second[0]
        self.assertTrue(torch.equal(first_item[0], second_item[0]))
        self.assertTrue(torch.equal(first_item[1], second_item[1]))
        self.assertTrue(torch.equal(first_item[2], second_item[2]))
        self.assertEqual(float(first[0][2]), 1.0)
        self.assertEqual(float(first[1][2]), 0.0)

    def test_training_seed_repeats_torch_random_sequence(self) -> None:
        seed_everything(123)
        first = torch.rand(5)
        seed_everything(123)
        second = torch.rand(5)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()