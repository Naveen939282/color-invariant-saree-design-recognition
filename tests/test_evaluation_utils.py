"""Unit tests for validation threshold, verification, and retrieval metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.evaluation_utils import (
    retrieval_metrics,
    select_threshold,
    verification_metrics,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_threshold_uses_validation_scores_and_maximizes_f1(self) -> None:
        selection = select_threshold([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
        self.assertEqual(selection["threshold"], 0.8)
        self.assertEqual(selection["validation_f1"], 1.0)
        self.assertIn("validation", selection["method"])

    def test_verification_metrics_include_auc_eer_and_threshold_metrics(self) -> None:
        pairs = pd.DataFrame(
            {
                "label": [0, 0, 1, 1],
                "similarity": [0.1, 0.2, 0.8, 0.9],
            }
        )
        metrics = verification_metrics(pairs, threshold=0.5)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["equal_error_rate"], 0.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_single_class_or_empty_validation_reports_unavailable_threshold(self) -> None:
        self.assertIsNone(select_threshold([1, 1], [0.8, 0.9])["threshold"])
        self.assertIsNone(select_threshold([], [])["threshold"])
        metrics = verification_metrics(
            pd.DataFrame({"label": [1, 1], "similarity": [0.8, 0.9]}),
            threshold=None,
        )
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["equal_error_rate"])
        self.assertIn("both positive and negative", metrics["metric_note"])

    def test_retrieval_reports_normal_and_cross_color_metrics(self) -> None:
        query = pd.DataFrame(
            [
                {"image_id": "q1", "design_id": "D001", "colorway": "red"},
                {"image_id": "q2", "design_id": "D002", "colorway": "blue"},
            ]
        )
        gallery = pd.DataFrame(
            [
                {"image_id": "g1", "design_id": "D001", "colorway": "blue"},
                {"image_id": "g2", "design_id": "D001", "colorway": "red"},
                {"image_id": "g3", "design_id": "D002", "colorway": "red"},
                {"image_id": "g4", "design_id": "D003", "colorway": "blue"},
            ]
        )
        embeddings = {
            "q1": np.asarray([1.0, 0.0]),
            "q2": np.asarray([0.0, 1.0]),
            "g1": np.asarray([0.99, 0.01]),
            "g2": np.asarray([0.8, 0.6]),
            "g3": np.asarray([0.0, 0.99]),
            "g4": np.asarray([-1.0, 0.0]),
        }
        metrics, ranking = retrieval_metrics(query, gallery, embeddings)
        self.assertAlmostEqual(metrics["recall@1"], 0.75)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["cross_color"]["query_count_with_cross_color_positive"], 2)
        self.assertEqual(metrics["cross_color"]["cross_color_recall@1"], 1.0)
        self.assertEqual(len(ranking), len(query) * len(gallery))

    def test_query_without_relevant_gallery_is_reported(self) -> None:
        query = pd.DataFrame(
            [{"image_id": "q", "design_id": "D001", "colorway": "red"}]
        )
        gallery = pd.DataFrame(
            [{"image_id": "g", "design_id": "D002", "colorway": "blue"}]
        )
        metrics, _ = retrieval_metrics(
            query,
            gallery,
            {"q": np.asarray([1.0, 0.0]), "g": np.asarray([0.0, 1.0])},
        )
        self.assertEqual(metrics["queries_without_gallery_positive"], 1)
        self.assertIsNone(metrics["recall@1"])


if __name__ == "__main__":
    unittest.main()