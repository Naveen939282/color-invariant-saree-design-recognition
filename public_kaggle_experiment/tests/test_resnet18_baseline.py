import unittest

import numpy as np
import torch

from public_kaggle_experiment.src.baseline_metrics import (
    assert_query_gallery_identity_match,
    build_verification_scores,
    cosine_similarity_matrix,
    retrieval_recall_at_k,
    select_validation_f1_threshold,
    verification_metrics,
)
from public_kaggle_experiment.src.resnet18_baseline import FrozenResNet18Baseline


class FrozenEmbeddingTests(unittest.TestCase):
    def test_embedding_shape_normalization_and_frozen_eval_mode(self):
        model = FrozenResNet18Baseline(weights=None)
        output = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 512))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)
        model.train()
        self.assertFalse(model.training)
        self.assertFalse(output.requires_grad)


class BaselineMetricTests(unittest.TestCase):
    def setUp(self):
        self.gallery_ids = ["candidate-a", "candidate-b", "candidate-c"]
        self.query_ids = ["candidate-a", "candidate-c"]
        self.similarities = torch.tensor(
            [[0.9, 0.2, 0.1], [0.3, 0.8, 0.7]], dtype=torch.float32
        )

    def test_gallery_query_identity_matching(self):
        assert_query_gallery_identity_match(self.query_ids, self.gallery_ids, expected_gallery_size=3)
        with self.assertRaisesRegex(ValueError, "missing from gallery"):
            assert_query_gallery_identity_match(["missing"], self.gallery_ids)

    def test_cosine_similarity(self):
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        gallery = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        scores = cosine_similarity_matrix(queries, gallery)
        torch.testing.assert_close(scores[0, 0], torch.tensor(1.0))
        torch.testing.assert_close(scores[1, 0], torch.tensor(0.0))
        torch.testing.assert_close(scores[1, 1], torch.tensor(2**-0.5))

    def test_recall_at_k(self):
        recalls = retrieval_recall_at_k(
            self.similarities, self.query_ids, self.gallery_ids, (1, 2, 3)
        )
        self.assertEqual(recalls, {1: 0.5, 2: 1.0, 3: 1.0})

    def test_verification_scores_use_candidate_identity_equality(self):
        scores, labels = build_verification_scores(
            self.similarities, self.query_ids, self.gallery_ids
        )
        np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(scores, [0.9, 0.2, 0.1, 0.3, 0.8, 0.7])

    def test_validation_threshold_is_selected_from_validation_pairs(self):
        validation_scores = np.array([0.95, 0.85, 0.75, 0.20, 0.10, 0.05])
        validation_labels = np.array([1, 1, 1, 0, 0, 0])
        threshold, validation_f1 = select_validation_f1_threshold(
            validation_scores, validation_labels
        )
        self.assertEqual(threshold, 0.75)
        self.assertEqual(validation_f1, 1.0)
        test = verification_metrics(np.array([0.70, 0.80]), np.array([1, 0]), threshold)
        self.assertEqual(test["threshold"], threshold)
        self.assertEqual(test["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()