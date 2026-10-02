"""Paired-view supervised contrastive helpers with deterministic scheduling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F

from public_kaggle_experiment.src.color_transforms import TRANSFORMATION_NAMES


TRAINING_TRANSFORMATIONS = tuple(
    name for name in TRANSFORMATION_NAMES if name != "identity"
)


def deterministic_training_transform(identity_position: int, epoch_index: int) -> str:
    if identity_position < 0 or epoch_index < 0:
        raise ValueError("Identity position and epoch index must be non-negative")
    return TRAINING_TRANSFORMATIONS[(identity_position + epoch_index) % len(TRAINING_TRANSFORMATIONS)]


def pair_identity_labels(identity_ids: Sequence[str]) -> list[str]:
    if not identity_ids:
        raise ValueError("At least one candidate experiment identity is required")
    return [identity_id for identity_id in identity_ids for _ in range(2)]


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    identity_ids: Sequence[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    """Treat both views of each candidate identity as positives; all others are negatives."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(identity_ids):
        raise ValueError("Embeddings must be a 2D tensor aligned with identity labels")
    if embeddings.shape[0] < 2:
        raise ValueError("At least two views are required")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    normalized = F.normalize(embeddings, p=2, dim=1)
    logits = normalized @ normalized.T / temperature
    diagonal = torch.eye(len(identity_ids), dtype=torch.bool, device=embeddings.device)
    positive_mask = torch.tensor(
        [
            [left == right for right in identity_ids]
            for left in identity_ids
        ],
        dtype=torch.bool,
        device=embeddings.device,
    ) & ~diagonal
    positive_counts = positive_mask.sum(dim=1)
    if torch.any(positive_counts == 0):
        raise ValueError("Every view must have another view of the same candidate identity")

    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probability = (log_probabilities * positive_mask).sum(dim=1)
    return -(positive_log_probability / positive_counts).mean()


def validate_identity_splits(
    canonical_rows: Sequence[dict[str, str]],
) -> dict[str, set[str]]:
    split_ids: dict[str, set[str]] = {split: set() for split in ("train", "valid", "test")}
    paths_by_split: dict[str, set[str]] = {split: set() for split in split_ids}
    for row in canonical_rows:
        split = row["split"]
        if split not in split_ids:
            raise ValueError(f"Unknown canonical split: {split}")
        identity_id = row["identity_id"]
        if identity_id != row["source_id"]:
            raise ValueError(f"Candidate identity must match source_id: {identity_id}")
        if identity_id in split_ids[split]:
            raise ValueError(f"Duplicate canonical candidate identity: {identity_id}")
        split_ids[split].add(identity_id)
        canonical_path = row["canonical_path"]
        if canonical_path in paths_by_split[split]:
            raise ValueError(f"Duplicate canonical image path in {split}: {canonical_path}")
        paths_by_split[split].add(canonical_path)

    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = split_ids[first] & split_ids[second]
        if overlap:
            raise ValueError(f"Candidate identities overlap {first}/{second}: {sorted(overlap)[:5]}")
        path_overlap = paths_by_split[first] & paths_by_split[second]
        if path_overlap:
            raise ValueError(f"Canonical images overlap {first}/{second}: {sorted(path_overlap)[:5]}")
    return split_ids


def checkpoint_is_better(candidate_metric: float, best_metric: float | None) -> bool:
    """Higher validation recall@1 wins; exact ties retain the earlier checkpoint."""
    if best_metric is None:
        return True
    return candidate_metric > best_metric