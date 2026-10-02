import unittest

import torch

from public_kaggle_experiment.scripts.train_color_invariant import _score_split
from public_kaggle_experiment.src.contrastive_learning import (
    checkpoint_is_better,
    deterministic_training_transform,
    pair_identity_labels,
    supervised_contrastive_loss,
    validate_identity_splits,
)
from public_kaggle_experiment.src.trained_model import ColorInvariantEmbeddingModel


class TrainedModelTests(unittest.TestCase):
    def test_overall_non_identity_recall_keeps_similarity_identity_alignment(self):
        from public_kaggle_experiment.src.color_transforms import TRANSFORMATION_NAMES

        identities = [f"candidate-{index}" for index in range(5)]
        rows = [
            {"identity_id": identity, "transformation": transformation}
            for identity in identities
            for transformation in TRANSFORMATION_NAMES
        ]
        encoded = {
            (identity, transformation): torch.nn.functional.one_hot(
                torch.tensor(index), num_classes=len(identities)
            ).float()
            for identity in identities
            for index in [identities.index(identity)]
            for transformation in TRANSFORMATION_NAMES
        }
        gallery = {
            identity: torch.nn.functional.one_hot(
                torch.tensor(index), num_classes=len(identities)
            ).float()
            for index, identity in enumerate(identities)
        }
        metrics, _, _, _ = _score_split(rows, encoded, gallery, threshold=None)
        self.assertEqual(metrics[-1]["transformation"], "overall_non_identity")
        self.assertEqual(metrics[-1]["recall_at_1"], 1.0)

    def test_model_embedding_is_128d_normalized_and_trainable(self):
        model = ColorInvariantEmbeddingModel(weights=None, embedding_dim=128)
        output = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 128))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        output.sum().backward()
        self.assertIsNotNone(model.projection.weight.grad)
        self.assertIsNotNone(model.backbone.conv1.weight.grad)

    def test_supervised_contrastive_loss_pulls_positive_views_together(self):
        embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            requires_grad=True,
        )
        loss = supervised_contrastive_loss(embeddings, ["a", "a", "b", "b"])
        wrong_pairs_loss = supervised_contrastive_loss(embeddings, ["a", "b", "a", "b"])
        self.assertLess(loss.item(), wrong_pairs_loss.item())
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_pair_identity_labels_and_fixed_epoch_transform_schedule(self):
        self.assertEqual(pair_identity_labels(["candidate-a", "candidate-b"]), [
            "candidate-a", "candidate-a", "candidate-b", "candidate-b"
        ])
        first_identity_schedule = [
            deterministic_training_transform(0, epoch) for epoch in range(10)
        ]
        self.assertEqual(len(set(first_identity_schedule)), 10)
        self.assertEqual(
            deterministic_training_transform(0, 0),
            deterministic_training_transform(0, 0),
        )

    def test_identity_split_integrity_rejects_candidate_and_path_overlap(self):
        rows = [
            {"identity_id": "a", "source_id": "a", "split": "train", "canonical_path": "train/a.jpg"},
            {"identity_id": "b", "source_id": "b", "split": "valid", "canonical_path": "valid/b.jpg"},
            {"identity_id": "c", "source_id": "c", "split": "test", "canonical_path": "test/c.jpg"},
        ]
        splits = validate_identity_splits(rows)
        self.assertEqual({key: len(value) for key, value in splits.items()}, {"train": 1, "valid": 1, "test": 1})
        with self.assertRaisesRegex(ValueError, "overlap train/test"):
            validate_identity_splits(rows + [
                {"identity_id": "c", "source_id": "c", "split": "train", "canonical_path": "train/c.jpg"}
            ])

    def test_checkpoint_selection_uses_only_validation_metric_and_keeps_earliest_tie(self):
        self.assertTrue(checkpoint_is_better(0.8, None))
        self.assertTrue(checkpoint_is_better(0.9, 0.8))
        self.assertFalse(checkpoint_is_better(0.8, 0.8))
        self.assertFalse(checkpoint_is_better(0.7, 0.8))


if __name__ == "__main__":
    unittest.main()