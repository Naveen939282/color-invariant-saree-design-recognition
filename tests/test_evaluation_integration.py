"""Synthetic end-to-end test for trained-model evaluation and local reports."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from scripts.embedding_model import ResNet18Embedding
from scripts.evaluate_retrieval import main as evaluate_main
from scripts.generate_retrieval_report import main as report_main
from scripts.split_utils import generate_pairs
from scripts.visualize_embeddings import main as visualize_main


class EvaluationIntegrationTests(unittest.TestCase):
    def test_trained_eval_metrics_pca_and_html_use_synthetic_images(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "deeplure"
            image_root.mkdir()
            rows = []
            for design_id, colorway, color in (
                ("D001", "red", (180, 20, 30)),
                ("D001", "blue", (20, 30, 180)),
                ("D002", "red", (180, 20, 30)),
                ("D002", "blue", (20, 30, 180)),
                ("D003", "red", (180, 20, 30)),
                ("D003", "blue", (20, 30, 180)),
                ("D003", "green", (20, 160, 40)),
                ("D004", "red", (180, 20, 30)),
                ("D004", "blue", (20, 30, 180)),
                ("D004", "green", (20, 160, 40)),
            ):
                relative_path = f"{design_id}/{colorway}.png"
                path = image_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (72, 72), color).save(path)
                rows.append(
                    {
                        "image_id": f"{design_id}_{colorway}",
                        "relative_path": relative_path,
                        "filename": f"{colorway}.png",
                        "source_dataset": "deeplure",
                        "design_id": design_id,
                        "colorway": colorway,
                    }
                )
            frame = pd.DataFrame(rows)
            validation = frame[frame["design_id"].isin(["D001", "D002"])].copy()
            evaluation = frame[frame["design_id"].isin(["D003", "D004"])].copy()
            query = evaluation[evaluation["colorway"] == "red"].copy()
            gallery = evaluation[evaluation["colorway"] != "red"].copy()
            pairs = generate_pairs(gallery, query, max_pairs_per_class=100, seed=42)

            validation_path = root / "validation.csv"
            gallery_path = root / "gallery.csv"
            query_path = root / "query.csv"
            pairs_path = root / "pairs.csv"
            labels_path = root / "labels.csv"
            for path, data in (
                (validation_path, validation),
                (gallery_path, gallery),
                (query_path, query),
                (pairs_path, pairs),
                (labels_path, frame.assign(label_confidence="high", label_source="synthetic", notes="")),
            ):
                data.to_csv(path, index=False)

            model = ResNet18Embedding(embedding_dim=16, pretrained=False, freeze_backbone=True)
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": {"embedding_dim": 16, "freeze_backbone": True},
                },
                checkpoint_path,
            )
            output_dir = root / "evaluation"
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_main(
                    [
                        "--deeplure-root", str(image_root),
                        "--validation-csv", str(validation_path),
                        "--gallery-csv", str(gallery_path),
                        "--query-csv", str(query_path),
                        "--pairs-csv", str(pairs_path),
                        "--checkpoint", str(checkpoint_path),
                        "--mode", "trained",
                        "--output-dir", str(output_dir),
                        "--batch-size", "4",
                        "--image-size", "64",
                        "--device", "cpu",
                    ]
                )
            self.assertEqual(result, 0)
            metrics = json.loads((output_dir / "evaluation_metrics.json").read_text())
            trained_result = metrics["evaluations"]["trained_color_aug_resnet18"]
            self.assertIsNotNone(trained_result["verification"]["roc_auc"])
            self.assertEqual(trained_result["retrieval"]["query_count"], 2)
            self.assertTrue((output_dir / "retrieval_trained_color_aug_resnet18.csv").is_file())

            figure_dir = root / "figures"
            with contextlib.redirect_stdout(io.StringIO()):
                visualize_main(
                    [
                        "--embeddings", str(output_dir / "embeddings_trained_color_aug_resnet18.npz"),
                        "--labels", str(labels_path),
                        "--output-dir", str(figure_dir),
                    ]
                )
                report_main(
                    [
                        "--ranking-csv", str(output_dir / "retrieval_trained_color_aug_resnet18.csv"),
                        "--query-csv", str(query_path),
                        "--gallery-csv", str(gallery_path),
                        "--deeplure-root", str(image_root),
                        "--output", str(root / "report" / "index.html"),
                        "--top-k", "2",
                    ]
                )
            self.assertTrue((figure_dir / "pca_by_design_id.png").is_file())
            report = (root / "report" / "index.html").read_text(encoding="utf-8")
            self.assertIn("file://", report)
            self.assertIn("different design", report)


if __name__ == "__main__":
    unittest.main()