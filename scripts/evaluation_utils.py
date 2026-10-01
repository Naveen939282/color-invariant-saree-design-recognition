"""Metric helpers for verification and query-to-gallery retrieval."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from .split_utils import color_relationship
except ImportError:
    from split_utils import color_relationship


def make_validation_pairs(validation: pd.DataFrame) -> pd.DataFrame:
    """Create all unordered validation pairs for threshold selection only."""
    rows = []
    records = validation.to_dict("records")
    for first, second in itertools.combinations(records, 2):
        same_design = first["design_id"] == second["design_id"]
        relation = color_relationship(first.get("colorway", ""), second.get("colorway", ""))
        rows.append(
            {
                "image_id_1": first["image_id"],
                "image_id_2": second["image_id"],
                "label": int(same_design),
                "pair_type": "positive" if same_design else "negative",
                "color_relationship": relation,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["image_id_1", "image_id_2", "label", "pair_type", "color_relationship"],
    )


def score_pairs(
    pairs: pd.DataFrame, embeddings: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Add cosine similarities to rows while retaining supplied pair labels."""
    output = pairs.copy()
    if output.empty:
        output["similarity"] = pd.Series(dtype=float)
        return output
    missing_ids = set(output["image_id_1"]) | set(output["image_id_2"])
    missing_ids -= set(embeddings)
    if missing_ids:
        raise ValueError(f"Pairs reference images without embeddings: {sorted(missing_ids)[:10]}")
    output["similarity"] = [
        float(np.dot(embeddings[first], embeddings[second]))
        for first, second in zip(output["image_id_1"], output["image_id_2"])
    ]
    return output


def select_threshold(labels: list[int], similarities: list[float]) -> dict[str, float | str | None]:
    """Choose a cosine threshold by maximum F1 on validation pairs only."""
    y_true = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(similarities, dtype=np.float64)
    if len(y_true) != len(scores) or len(y_true) == 0:
        return {"threshold": None, "validation_f1": None, "method": "unavailable"}
    if len(np.unique(y_true)) < 2:
        return {"threshold": None, "validation_f1": None, "method": "unavailable: validation has one class"}
    thresholds = np.unique(np.concatenate(([scores.min() - 1e-6], scores, [scores.max() + 1e-6])))
    candidates = []
    for threshold in thresholds:
        prediction = (scores >= threshold).astype(np.int64)
        candidates.append((f1_score(y_true, prediction, zero_division=0), float(threshold)))
    best_f1, best_threshold = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "threshold": best_threshold,
        "validation_f1": float(best_f1),
        "method": "maximum validation-pair F1; ties select the higher cosine threshold",
    }


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def verification_metrics(
    scored_pairs: pd.DataFrame, threshold: float | None
) -> dict[str, Any]:
    """Calculate threshold and ranking metrics without tuning on these pairs."""
    labels = scored_pairs["label"].astype(int).to_numpy()
    scores = scored_pairs["similarity"].astype(float).to_numpy()
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    result: dict[str, Any] = {
        "pair_count": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "positive_similarity": _distribution(positive),
        "negative_similarity": _distribution(negative),
        "threshold": threshold,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "equal_error_rate": None,
        "metric_note": None,
    }
    if threshold is not None and len(labels) > 0:
        predicted = (scores >= threshold).astype(np.int64)
        result.update(
            accuracy=float(accuracy_score(labels, predicted)),
            precision=float(precision_score(labels, predicted, zero_division=0)),
            recall=float(recall_score(labels, predicted, zero_division=0)),
            f1=float(f1_score(labels, predicted, zero_division=0)),
        )
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        false_negative_rate = 1.0 - true_positive_rate
        eer_index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
        result["equal_error_rate"] = float(
            (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2.0
        )
    else:
        result["metric_note"] = "ROC-AUC and EER require both positive and negative pairs."
    if len(labels) == 0:
        result["metric_note"] = "No verification pairs were available."
    return result


def retrieval_metrics(
    query: pd.DataFrame,
    gallery: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    ks: tuple[int, ...] = (1, 3, 5),
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute macro query Recall@K/MRR and colorway-stratified retrieval."""
    ranking_rows = []
    per_query: list[dict[str, Any]] = []
    no_gallery_positive = 0
    for query_row in query.to_dict("records"):
        query_id = query_row["image_id"]
        if query_id not in embeddings:
            raise ValueError(f"No embedding for query image {query_id}.")
        similarities = []
        for gallery_row in gallery.to_dict("records"):
            gallery_id = gallery_row["image_id"]
            if gallery_id not in embeddings:
                raise ValueError(f"No embedding for gallery image {gallery_id}.")
            similarity = float(np.dot(embeddings[query_id], embeddings[gallery_id]))
            same_design = query_row["design_id"] == gallery_row["design_id"]
            relation = color_relationship(
                query_row.get("colorway", ""), gallery_row.get("colorway", "")
            )
            similarities.append((similarity, gallery_row, same_design, relation))
        similarities.sort(key=lambda item: (-item[0], item[1]["image_id"]))
        all_relevant_ranks = [
            rank
            for rank, (_, _, same_design, _) in enumerate(similarities, start=1)
            if same_design
        ]
        cross_color_ranks = [
            rank
            for rank, (_, _, same_design, relation) in enumerate(similarities, start=1)
            if same_design and relation == "different"
        ]
        same_color_ranks = [
            rank
            for rank, (_, _, same_design, relation) in enumerate(similarities, start=1)
            if same_design and relation == "same"
        ]
        if not all_relevant_ranks:
            no_gallery_positive += 1
        record: dict[str, Any] = {
            "image_id": query_id,
            "design_id": query_row["design_id"],
            "relevant_gallery_count": len(all_relevant_ranks),
            "cross_color_relevant_count": len(cross_color_ranks),
            "same_color_relevant_count": len(same_color_ranks),
            "mrr": 1.0 / all_relevant_ranks[0] if all_relevant_ranks else None,
            "cross_color_mrr": 1.0 / cross_color_ranks[0] if cross_color_ranks else None,
        }
        for k in ks:
            record[f"recall@{k}"] = (
                sum(rank <= k for rank in all_relevant_ranks) / len(all_relevant_ranks)
                if all_relevant_ranks
                else None
            )
            record[f"cross_color_recall@{k}"] = (
                sum(rank <= k for rank in cross_color_ranks) / len(cross_color_ranks)
                if cross_color_ranks
                else None
            )
        per_query.append(record)
        for rank, (similarity, gallery_row, same_design, relation) in enumerate(
            similarities, start=1
        ):
            ranking_rows.append(
                {
                    "query_image_id": query_id,
                    "query_design_id": query_row["design_id"],
                    "query_colorway": query_row.get("colorway", ""),
                    "rank": rank,
                    "gallery_image_id": gallery_row["image_id"],
                    "gallery_design_id": gallery_row["design_id"],
                    "gallery_colorway": gallery_row.get("colorway", ""),
                    "similarity": similarity,
                    "same_design": int(same_design),
                    "color_relationship": relation,
                }
            )

    per_query_frame = pd.DataFrame(per_query)
    metrics: dict[str, Any] = {
        "query_count": int(len(query)),
        "gallery_count": int(len(gallery)),
        "queries_without_gallery_positive": no_gallery_positive,
        "query_metrics": per_query,
    }
    if per_query_frame.empty:
        for column in [*(f"recall@{k}" for k in ks), "mrr"]:
            metrics[column] = None
        metrics["cross_color"] = {
            "query_count_with_cross_color_positive": 0,
            **{f"cross_color_recall@{k}": None for k in ks},
            "cross_color_mrr": None,
        }
        metrics["colorway_matched_positive_query_count"] = 0
        return metrics, pd.DataFrame(ranking_rows)
    for column in [*(f"recall@{k}" for k in ks), "mrr"]:
        values = pd.to_numeric(per_query_frame[column], errors="coerce")
        metrics[column] = float(values.mean()) if values.notna().any() else None
    cross_summary: dict[str, Any] = {
        "query_count_with_cross_color_positive": int(
            per_query_frame["cross_color_relevant_count"].gt(0).sum()
        )
    }
    for column in [*(f"cross_color_recall@{k}" for k in ks), "cross_color_mrr"]:
        values = pd.to_numeric(per_query_frame[column], errors="coerce")
        cross_summary[column] = float(values.mean()) if values.notna().any() else None
    metrics["cross_color"] = cross_summary
    metrics["colorway_matched_positive_query_count"] = int(
        per_query_frame["same_color_relevant_count"].gt(0).sum()
    )
    return metrics, pd.DataFrame(ranking_rows)