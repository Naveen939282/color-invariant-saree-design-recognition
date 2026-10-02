"""Metric helpers for the frozen controlled color-invariance baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def assert_query_gallery_identity_match(
    query_identity_ids: Sequence[str],
    gallery_identity_ids: Sequence[str],
    *,
    expected_gallery_size: int | None = None,
) -> None:
    if len(set(gallery_identity_ids)) != len(gallery_identity_ids):
        raise ValueError("Gallery contains duplicate candidate identities")
    if expected_gallery_size is not None and len(gallery_identity_ids) != expected_gallery_size:
        raise ValueError(
            f"Expected {expected_gallery_size} gallery identities, got {len(gallery_identity_ids)}"
        )
    gallery_ids = set(gallery_identity_ids)
    missing = sorted(set(query_identity_ids) - gallery_ids)
    if missing:
        raise ValueError(f"Query identities missing from gallery: {missing[:5]}")


def cosine_similarity_matrix(
    query_embeddings: torch.Tensor, gallery_embeddings: torch.Tensor
) -> torch.Tensor:
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("Query and gallery embeddings must be 2D tensors")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("Query and gallery embedding dimensions do not match")
    if not torch.isfinite(query_embeddings).all() or not torch.isfinite(gallery_embeddings).all():
        raise ValueError("Embeddings contain non-finite values")
    queries = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    gallery = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)
    return queries @ gallery.T


def retrieval_recall_at_k(
    similarities: torch.Tensor | np.ndarray,
    query_identity_ids: Sequence[str],
    gallery_identity_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[int, float]:
    assert_query_gallery_identity_match(query_identity_ids, gallery_identity_ids)
    matrix = (
        similarities.detach().cpu().numpy()
        if isinstance(similarities, torch.Tensor)
        else np.asarray(similarities)
    )
    if matrix.shape != (len(query_identity_ids), len(gallery_identity_ids)):
        raise ValueError("Similarity matrix shape does not match query/gallery identity counts")
    if not np.isfinite(matrix).all():
        raise ValueError("Similarity matrix contains non-finite values")
    gallery_positions = {identity_id: index for index, identity_id in enumerate(gallery_identity_ids)}
    rankings = np.argsort(-matrix, axis=1, kind="stable")
    recalls: dict[int, float] = {}
    targets = np.asarray([gallery_positions[identity_id] for identity_id in query_identity_ids])
    for k in k_values:
        if k < 1 or k > len(gallery_identity_ids):
            raise ValueError(f"Recall@{k} requires 1 <= k <= gallery size")
        recalls[k] = float(np.any(rankings[:, :k] == targets[:, None], axis=1).mean())
    return recalls


def build_verification_scores(
    similarities: torch.Tensor | np.ndarray,
    query_identity_ids: Sequence[str],
    gallery_identity_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    assert_query_gallery_identity_match(query_identity_ids, gallery_identity_ids)
    matrix = (
        similarities.detach().cpu().numpy()
        if isinstance(similarities, torch.Tensor)
        else np.asarray(similarities)
    )
    if matrix.shape != (len(query_identity_ids), len(gallery_identity_ids)):
        raise ValueError("Similarity matrix shape does not match query/gallery identity counts")
    labels = np.equal(
        np.asarray(query_identity_ids, dtype=object)[:, None],
        np.asarray(gallery_identity_ids, dtype=object)[None, :],
    ).astype(np.int8)
    return matrix.reshape(-1).astype(np.float64), labels.reshape(-1)


def select_validation_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if scores.size != labels.size or scores.size == 0:
        raise ValueError("Validation scores and labels must have the same non-zero length")
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation scores must be finite and labels must contain both classes")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Could not derive a validation threshold")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best_f1 = float(f1_values.max())
    tied = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0.0, atol=1e-12))
    chosen_index = int(tied[-1])
    return float(thresholds[chosen_index]), best_f1


def verification_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if scores.size != labels.size or scores.size == 0:
        raise ValueError("Verification scores and labels must have the same non-zero length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Verification labels must contain both positive and negative pairs")
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, scores >= threshold, zero_division=0)),
        "threshold": float(threshold),
    }