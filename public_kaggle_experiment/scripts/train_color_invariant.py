"""Train and evaluate a contrastive ResNet18 on controlled color transforms."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from public_kaggle_experiment.scripts.create_color_invariance_manifest import (  # noqa: E402
    read_canonical_rows,
)
from public_kaggle_experiment.scripts.evaluate_resnet18_baseline import (  # noqa: E402
    CanonicalTransformDataset,
    EXPECTED_IDENTITIES,
    _load_csv,
    encode_records,
    measure_inference_latency,
    resolve_device,
    set_deterministic_seed,
    validate_test_gallery,
    validate_transform_manifest,
)
from public_kaggle_experiment.src.baseline_metrics import (  # noqa: E402
    assert_query_gallery_identity_match,
    build_verification_scores,
    cosine_similarity_matrix,
    retrieval_recall_at_k,
    select_validation_f1_threshold,
    verification_metrics,
)
from public_kaggle_experiment.src.color_transforms import (  # noqa: E402
    TRANSFORMATION_CONFIG,
    TRANSFORMATION_NAMES,
)
from public_kaggle_experiment.src.contrastive_learning import (  # noqa: E402
    TRAINING_TRANSFORMATIONS,
    checkpoint_is_better,
    deterministic_training_transform,
    pair_identity_labels,
    supervised_contrastive_loss,
    validate_identity_splits,
)
from public_kaggle_experiment.src.trained_model import (  # noqa: E402
    ColorInvariantEmbeddingModel,
)


EVALUATION_BATCH_SIZE = 32
IDENTITY_TRANSFORM = "identity"
NON_IDENTITY_TRANSFORMS = tuple(
    name for name in TRANSFORMATION_NAMES if name != IDENTITY_TRANSFORM
)


class PairedCanonicalTrainingDataset(Dataset):
    """Yield one canonical and one scheduled transformed view per train identity."""

    def __init__(self, rows: list[dict[str, str]], dataset_root: Path, preprocess: Any):
        self.rows = rows
        self.dataset_root = dataset_root.resolve()
        self.preprocess = preprocess
        self.epoch_index = 0
        self._reader: CanonicalTransformDataset | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch_index: int) -> None:
        self.epoch_index = epoch_index

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.rows[index]
        transform = deterministic_training_transform(index, self.epoch_index)
        canonical = {
            "identity_id": row["identity_id"],
            "canonical_path": row["canonical_path"],
            "transformation": IDENTITY_TRANSFORM,
        }
        transformed = {
            "identity_id": row["identity_id"],
            "canonical_path": row["canonical_path"],
            "transformation": transform,
        }
        if self._reader is None:
            self._reader = CanonicalTransformDataset(
                [canonical, transformed], self.dataset_root, self.preprocess
            )
        else:
            self._reader.rows = [canonical, transformed]
        canonical_tensor, _, _ = self._reader[0]
        transformed_tensor, _, _ = self._reader[1]
        return torch.stack((canonical_tensor, transformed_tensor)), row["identity_id"], transform

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _validation_records(
    canonical_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]], split: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    gallery = [
        {
            "identity_id": row["identity_id"],
            "source_id": row["source_id"],
            "canonical_path": row["canonical_path"],
            "transformation": "canonical",
        }
        for row in canonical_rows
        if row["split"] == split
    ]
    queries = sorted(
        manifest_rows,
        key=lambda row: (row["identity_id"], TRANSFORMATION_NAMES.index(row["transformation"])),
    )
    return gallery, queries


def _score_split(
    query_rows: list[dict[str, str]],
    encoded: dict[tuple[str, str], torch.Tensor],
    gallery: dict[str, torch.Tensor],
    *,
    threshold: float | None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], list[np.ndarray], list[np.ndarray]]:
    gallery_ids = list(gallery)
    gallery_vectors = torch.stack([gallery[identity_id] for identity_id in gallery_ids])
    result_rows: list[dict[str, Any]] = []
    similarities_by_transform: dict[str, torch.Tensor] = {}
    non_identity_scores: list[np.ndarray] = []
    non_identity_labels: list[np.ndarray] = []
    non_identity_query_ids: list[str] = []

    for transformation in TRANSFORMATION_NAMES:
        rows = [row for row in query_rows if row["transformation"] == transformation]
        query_ids = [row["identity_id"] for row in rows]
        if len(query_ids) != len(gallery_ids) or set(query_ids) != set(gallery_ids):
            raise ValueError(f"{transformation} query identities do not match the {len(gallery_ids)}-item gallery")
        query_vectors = torch.stack(
            [encoded[(identity_id, transformation)] for identity_id in query_ids]
        )
        similarities = cosine_similarity_matrix(query_vectors, gallery_vectors)
        similarities_by_transform[transformation] = similarities
        recall = retrieval_recall_at_k(similarities, query_ids, gallery_ids, (1, 3, 5))
        scores, labels = build_verification_scores(similarities, query_ids, gallery_ids)
        row: dict[str, Any] = {
            "transformation": transformation,
            "query_count": len(query_ids),
            "gallery_count": len(gallery_ids),
            "positive_pairs": int(labels.sum()),
            "negative_pairs": int(len(labels) - labels.sum()),
            "recall_at_1": recall[1],
            "recall_at_3": recall[3],
            "recall_at_5": recall[5],
            "roc_auc": float(roc_auc_score(labels, scores)),
            "f1": "",
            "threshold": "",
        }
        if threshold is not None:
            verification = verification_metrics(scores, labels, threshold)
            row["f1"] = verification["f1"]
            row["threshold"] = threshold
        result_rows.append(row)
        if transformation != IDENTITY_TRANSFORM:
            non_identity_scores.append(scores)
            non_identity_labels.append(labels)
            non_identity_query_ids.extend(query_ids)

    non_identity_matrices = [
        similarities_by_transform[name] for name in NON_IDENTITY_TRANSFORMS
    ]
    pooled_similarities = torch.cat(non_identity_matrices, dim=0)
    pooled_recall = retrieval_recall_at_k(
        pooled_similarities, non_identity_query_ids, gallery_ids, (1, 3, 5)
    )
    pooled_scores = np.concatenate(non_identity_scores)
    pooled_labels = np.concatenate(non_identity_labels)
    overall: dict[str, Any] = {
        "transformation": "overall_non_identity",
        "query_count": len(non_identity_query_ids),
        "gallery_count": len(gallery_ids),
        "positive_pairs": int(pooled_labels.sum()),
        "negative_pairs": int(len(pooled_labels) - pooled_labels.sum()),
        "recall_at_1": pooled_recall[1],
        "recall_at_3": pooled_recall[3],
        "recall_at_5": pooled_recall[5],
        "roc_auc": float(roc_auc_score(pooled_labels, pooled_scores)),
        "f1": "",
        "threshold": "",
    }
    if threshold is not None:
        pooled_verification = verification_metrics(pooled_scores, pooled_labels, threshold)
        overall["f1"] = pooled_verification["f1"]
        overall["threshold"] = threshold
    return result_rows + [overall], similarities_by_transform, non_identity_scores, non_identity_labels


def _validation_threshold(
    query_rows: list[dict[str, str]],
    similarities_by_transform: dict[str, torch.Tensor],
    gallery_ids: list[str],
) -> tuple[float, float, int]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for transformation in NON_IDENTITY_TRANSFORMS:
        rows = [row for row in query_rows if row["transformation"] == transformation]
        transform_scores, transform_labels = build_verification_scores(
            similarities_by_transform[transformation],
            [row["identity_id"] for row in rows],
            gallery_ids,
        )
        scores.append(transform_scores)
        labels.append(transform_labels)
    pooled_scores = np.concatenate(scores)
    pooled_labels = np.concatenate(labels)
    threshold, validation_f1 = select_validation_f1_threshold(pooled_scores, pooled_labels)
    return threshold, validation_f1, int(pooled_labels.size)


def _encode_split(
    canonical_rows: list[dict[str, str]],
    query_rows: list[dict[str, str]],
    split: str,
    dataset_root: Path,
    preprocess: Any,
    model: ColorInvariantEmbeddingModel,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str], torch.Tensor], torch.Tensor, dict[str, float]]:
    gallery_rows, sorted_queries = _validation_records(canonical_rows, query_rows, split)
    records = gallery_rows + sorted_queries
    encoded, timing, timing_batch = encode_records(
        records,
        dataset_root,
        preprocess,
        model,  # The shared encoder uses only the embedding_dim/forward contract.
        device,
        EVALUATION_BATCH_SIZE,
        num_workers,
        seed,
    )
    gallery = {
        row["identity_id"]: encoded[(row["identity_id"], "canonical")]
        for row in gallery_rows
    }
    return gallery, encoded, timing_batch, timing


def _baseline_comparison(
    baseline_csv: Path, trained_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with baseline_csv.open(newline="", encoding="utf-8") as source:
        baseline_rows = list(csv.DictReader(source))
    baseline_by_transform = {row["transformation"]: row for row in baseline_rows}
    trained_by_transform = {row["transformation"]: row for row in trained_rows}
    expected = set(TRANSFORMATION_NAMES) | {"overall_non_identity"}
    if set(baseline_by_transform) != expected or set(trained_by_transform) != expected:
        raise ValueError("Baseline/trained summary rows do not match the fixed transformation protocol")

    comparisons: list[dict[str, Any]] = []
    outcomes = {"improvements": 0, "unchanged": 0, "regressions": 0}
    for transformation in (*TRANSFORMATION_NAMES, "overall_non_identity"):
        baseline = baseline_by_transform[transformation]
        trained = trained_by_transform[transformation]
        row: dict[str, Any] = {"transformation": transformation}
        for key in ("recall_at_1", "recall_at_3", "recall_at_5", "roc_auc", "f1"):
            baseline_value = float(baseline[key])
            trained_value = float(trained[key])
            delta = trained_value - baseline_value
            row[f"baseline_{key}"] = baseline_value
            row[f"trained_{key}"] = trained_value
            row[f"delta_{key}"] = delta
            if delta > 1e-12:
                outcomes["improvements"] += 1
            elif delta < -1e-12:
                outcomes["regressions"] += 1
            else:
                outcomes["unchanged"] += 1
        comparisons.append(row)
    return comparisons, outcomes


def _write_report(
    output_path: Path,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    best_validation: dict[str, Any],
    threshold_info: dict[str, Any],
    test_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    comparison_outcomes: dict[str, int],
    timing: dict[str, Any],
) -> None:
    lines = [
        "# Trained Color-Invariant Embedding Experiment",
        "",
        "This is a controlled synthetic color-invariance evaluation over candidate filename-derived identities, not verified saree designs or real colorways. It is not evidence of real-world cross-color generalization.",
        "",
        f"- Best checkpoint: epoch {best_validation['epoch']} selected by validation overall non-identity Recall@1 = {best_validation['validation_overall_non_identity_recall_at_1']:.4f}",
        f"- Validation-selected global verification threshold: {threshold_info['threshold']:.8f} (pooled non-identity validation F1 {threshold_info.get('validation_f1', threshold_info.get('validation_f1_at_threshold', float('nan'))):.4f})",
        f"- Pretrained backbone: {config['pretrained_weights']}; projection: Linear(512, 128) then L2 normalization",
        f"- Parameters: {config['total_parameters']:,}; trainable: {config['trainable_parameters']:,}; embedding dimension: {config['embedding_dim']}",
        f"- Device: {config['device']}; batch size: {config['batch_size']}; epochs: {config['epochs']}",
        f"- Training-loop runtime (sum of epoch measurements): {timing['optimization_runtime_seconds']:.1f}s; validation runtime (sum of epoch measurements): {timing['validation_epoch_runtime_seconds']:.1f}s",
        f"- Combined recorded epoch time: {timing['training_and_validation_epoch_seconds']:.1f}s; excludes initialization and final checkpoint/test evaluation",
        f"- Inference median: {timing['pure_inference']['median_batch_latency_ms']:.3f}ms/batch ({timing['pure_inference']['median_latency_per_image_ms']:.3f}ms/image)",
        "",
        "## Held-Out Test Results",
        "",
        "| Transformation | Recall@1 | Recall@3 | Recall@5 | ROC-AUC | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            f"| {row['transformation']} | {row['recall_at_1']:.4f} | {row['recall_at_3']:.4f} | "
            f"{row['recall_at_5']:.4f} | {row['roc_auc']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            "Positive deltas are higher metric values; regressions are shown rather than hidden. No winner is declared by the script.",
            "",
            "| Transformation | Baseline R@1 | Trained R@1 | Baseline R@3 | Trained R@3 | Baseline R@5 | Trained R@5 | Baseline AUC | Trained AUC | Baseline F1 | Trained F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['transformation']} | {row['baseline_recall_at_1']:.4f} | {row['trained_recall_at_1']:.4f} | "
            f"{row['baseline_recall_at_3']:.4f} | {row['trained_recall_at_3']:.4f} | "
            f"{row['baseline_recall_at_5']:.4f} | {row['trained_recall_at_5']:.4f} | "
            f"{row['baseline_roc_auc']:.4f} | {row['trained_roc_auc']:.4f} | "
            f"{row['baseline_f1']:.4f} | {row['trained_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Across the {len(comparisons) * 5} transformation/metric comparisons: {comparison_outcomes['improvements']} improvements, {comparison_outcomes['unchanged']} unchanged, {comparison_outcomes['regressions']} regressions.",
            "",
            "## Protocol and Limitations",
            "",
            "- Training used only 431 train candidate experiment identities, with a canonical view paired against one of the ten fixed non-identity transforms each epoch. Transform assignment rotates deterministically across epochs; no extra augmentation was used.",
            "- The 115 validation identities select checkpoints by pooled overall non-identity Recall@1. Strict ties retain the earliest checkpoint.",
            "- After checkpoint selection, a single threshold maximizes pooled validation F1 across non-identity validation pairs; ties choose the largest threshold. It is applied unchanged to every test transformation.",
            "- The test set contains 60 canonical candidate identities and 660 transform queries; test data was not used for training, checkpoint selection, or threshold selection.",
            "- Candidate identities are filename-derived `source_id` groups, not verified design IDs. Same-design/different-color pairs are absent; all findings are controlled synthetic color-invariance results.",
            "- High scores reflect transformed views of the same source image. They do not establish real-world saree cross-color recognition.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_training(
    dataset_root: Path,
    canonical_csv: Path,
    manifest_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    embedding_dim: int = 128,
    temperature: float = 0.07,
    seed: int = 42,
    requested_device: str = "auto",
    num_workers: int = 0,
    baseline_summary: Path | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 4 or batch_size % 2:
        raise ValueError("epochs must be positive and batch-size must be an even integer >= 4")
    if learning_rate <= 0 or weight_decay < 0 or embedding_dim < 1 or temperature <= 0:
        raise ValueError("Invalid optimizer, embedding, or contrastive temperature configuration")
    if num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists() or (dataset_root.is_file() and dataset_root.suffix.casefold() != ".zip"):
        raise ValueError(f"Dataset root must be an extracted directory or ZIP archive: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_rows = read_canonical_rows(canonical_csv)
    split_ids = validate_identity_splits(canonical_rows)
    if {split: len(ids) for split, ids in split_ids.items()} != EXPECTED_IDENTITIES:
        raise ValueError("Canonical candidate identity counts differ from the audited protocol")
    canonical_paths = [row["canonical_path"] for row in canonical_rows]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError("Canonical image path is reused across candidate identities/splits")
    canonical_by_id = {row["identity_id"]: row for row in canonical_rows}

    train_manifest = _load_csv(manifest_dir / "train_manifest.csv")
    valid_manifest = _load_csv(manifest_dir / "valid_manifest.csv")
    validate_transform_manifest(train_manifest, canonical_by_id, "train", 431)
    validate_transform_manifest(valid_manifest, canonical_by_id, "valid", 115)
    train_rows = sorted(
        (row for row in canonical_rows if row["split"] == "train"),
        key=lambda row: row["identity_id"],
    )
    valid_gallery_rows, valid_query_rows = _validation_records(
        canonical_rows, valid_manifest, "valid"
    )
    if len(train_rows) != 431 or len(valid_gallery_rows) != 115:
        raise ValueError("Training/validation identity count invariant failed")

    device = resolve_device(requested_device)
    set_deterministic_seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = ColorInvariantEmbeddingModel(weights=weights, embedding_dim=embedding_dim).to(device)
    model.train()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    config: dict[str, Any] = {
        "seed": seed,
        "model_architecture": "torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), classifier replaced with Identity, Linear(512, embedding_dim), L2 normalization",
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_url": weights.url,
        "embedding_dim": embedding_dim,
        "projection_head": "Linear(512, 128) followed by L2 normalization; trained jointly with the ResNet18 backbone",
        "temperature": temperature,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "device": str(device),
        "num_workers": num_workers,
        "dataset_split": {split: len(values) for split, values in split_ids.items()},
        "identity_scope": "candidate experiment identity from audited filename-derived source_id; not verified design identity",
        "training_transforms": list(TRAINING_TRANSFORMATIONS),
        "transformation_definitions": TRANSFORMATION_CONFIG,
        "training_view_policy": "canonical plus one deterministic transform per candidate identity each epoch; transform position rotates across the ten non-identity definitions",
        "training_objective": "supervised contrastive loss; two views of same source_id are positives, all other in-batch source_ids are negatives",
        "checkpoint_selection_rule": "maximum validation overall non-identity Recall@1 pooled across all ten transforms; exact ties retain earliest epoch",
        "threshold_selection_rule": "after checkpoint selection, maximize pooled validation non-identity pair F1; ties choose largest threshold; freeze for all test transformations",
        "validation_threshold_excludes_identity_control": True,
        "test_used_for_training_or_model_selection": False,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
        "pillow_version": PIL.__version__,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_parameters": "all backbone and projection parameters trainable",
    }
    _write_json(output_dir / "config.json", config)

    training_dataset = PairedCanonicalTrainingDataset(train_rows, dataset_root, preprocess)
    identities_per_batch = batch_size // 2
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_validation: dict[str, Any] = {"epoch": 0, "validation_overall_non_identity_recall_at_1": -1.0}
    optimization_seconds = 0.0
    wall_started = time.perf_counter()

    try:
        for epoch_index in range(epochs):
            training_dataset.set_epoch(epoch_index)
            generator = torch.Generator().manual_seed(seed + epoch_index)
            loader = DataLoader(
                training_dataset,
                batch_size=identities_per_batch,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=device.type == "cuda",
                generator=generator,
            )
            model.train()
            train_loss_sum = 0.0
            trained_identity_count = 0
            epoch_train_started = time.perf_counter()
            for paired_views, identity_ids, _ in loader:
                flat_views = paired_views.flatten(start_dim=0, end_dim=1).to(
                    device, non_blocking=device.type == "cuda"
                )
                batch_identities = list(identity_ids)
                view_identity_ids = pair_identity_labels(batch_identities)
                optimizer.zero_grad(set_to_none=True)
                embeddings = model(flat_views)
                loss = supervised_contrastive_loss(
                    embeddings, view_identity_ids, temperature=temperature
                )
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss at epoch {epoch_index + 1}")
                loss.backward()
                optimizer.step()
                batch_identity_count = len(batch_identities)
                train_loss_sum += float(loss.detach()) * batch_identity_count
                trained_identity_count += batch_identity_count
            epoch_train_seconds = time.perf_counter() - epoch_train_started
            optimization_seconds += epoch_train_seconds

            model.eval()
            valid_embeddings, validation_encoding_timing, _ = encode_records(
                valid_gallery_rows + valid_query_rows,
                dataset_root,
                preprocess,
                model,  # Shared baseline encoder invokes inference only and reads no labels.
                device,
                EVALUATION_BATCH_SIZE,
                num_workers,
                seed + 1000 + epoch_index,
            )
            validation_gallery = {
                row["identity_id"]: valid_embeddings[(row["identity_id"], "canonical")]
                for row in valid_gallery_rows
            }
            epoch_validation_rows, validation_similarities, _, _ = _score_split(
                valid_query_rows,
                valid_embeddings,
                validation_gallery,
                threshold=None,
            )
            overall_validation = epoch_validation_rows[-1]
            selection_metric = float(overall_validation["recall_at_1"])
            selected = checkpoint_is_better(
                selection_metric,
                float(best_validation["validation_overall_non_identity_recall_at_1"]),
            )
            epoch_number = epoch_index + 1
            for validation_row in epoch_validation_rows:
                validation_history.append({"epoch": epoch_number, **validation_row})

            epoch_row = {
                "epoch": epoch_number,
                "mean_training_loss": train_loss_sum / trained_identity_count,
                "validation_overall_non_identity_recall_at_1": selection_metric,
                "checkpoint_selected": int(selected),
                "training_seconds": epoch_train_seconds,
                "validation_seconds": sum(validation_encoding_timing.values()),
                "validation_identity_control_recall_at_1": epoch_validation_rows[0]["recall_at_1"],
            }
            history.append(epoch_row)
            if selected:
                best_validation = {
                    "epoch": epoch_number,
                    "validation_overall_non_identity_recall_at_1": selection_metric,
                }
                torch.save(
                    {
                        "epoch": epoch_number,
                        "model_state_dict": model.state_dict(),
                        "validation_overall_non_identity_recall_at_1": selection_metric,
                        "config": config,
                    },
                    output_dir / "best_model.pt",
                )
            print(
                f"epoch {epoch_number}/{epochs}: loss={epoch_row['mean_training_loss']:.5f} "
                f"validation_nonidentity_R@1={selection_metric:.4f} "
                f"selected={bool(selected)} train_s={epoch_train_seconds:.1f}",
                flush=True,
            )
    finally:
        training_dataset.close()

    wall_training_seconds = time.perf_counter() - wall_started
    if best_validation["epoch"] == 0:
        raise RuntimeError("Training completed without selecting a checkpoint")
    _write_csv(
        output_dir / "training_history.csv",
        history,
        (
            "epoch", "mean_training_loss", "validation_overall_non_identity_recall_at_1",
            "checkpoint_selected", "training_seconds", "validation_seconds",
            "validation_identity_control_recall_at_1",
        ),
    )
    _write_csv(
        output_dir / "validation_metrics.csv",
        validation_history,
        (
            "epoch", "transformation", "query_count", "gallery_count", "positive_pairs",
            "negative_pairs", "recall_at_1", "recall_at_3", "recall_at_5", "roc_auc", "f1", "threshold",
        ),
    )

    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(
            output_dir / "best_model.pt", map_location=device, weights_only=True
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Select the verification threshold only after loading the validation-selected checkpoint.
    best_valid_embeddings, _, _ = encode_records(
        valid_gallery_rows + valid_query_rows,
        dataset_root,
        preprocess,
        model,
        device,
        EVALUATION_BATCH_SIZE,
        num_workers,
        seed + 2000,
    )
    best_valid_gallery = {
        row["identity_id"]: best_valid_embeddings[(row["identity_id"], "canonical")]
        for row in valid_gallery_rows
    }
    best_valid_rows, best_valid_similarities, _, _ = _score_split(
        valid_query_rows, best_valid_embeddings, best_valid_gallery, threshold=None
    )
    valid_gallery_ids = list(best_valid_gallery)
    threshold, validation_f1, validation_pair_count = _validation_threshold(
        valid_query_rows, best_valid_similarities, valid_gallery_ids
    )
    threshold_info = {
        "threshold": threshold,
        "selection_split": "valid",
        "checkpoint_epoch": best_validation["epoch"],
        "validation_f1_at_threshold": validation_f1,
        "validation_pair_count": validation_pair_count,
        "method": "maximize pooled validation F1 over all non-identity query/gallery cosine pairs",
        "identity_control_excluded": True,
        "tie_break": "largest threshold among thresholds with equal validation F1",
        "test_labels_used": False,
    }
    _write_json(output_dir / "selected_validation_threshold.json", threshold_info)

    # Test manifests and image paths are opened only after checkpoint and threshold selection.
    test_manifest = _load_csv(manifest_dir / "test_manifest.csv")
    test_gallery_rows = validate_test_gallery(
        _load_csv(manifest_dir / "test_gallery.csv"), canonical_by_id
    )
    validate_transform_manifest(test_manifest, canonical_by_id, "test", 60)
    for row in test_manifest:
        if row["identity_id"] not in split_ids["test"]:
            raise ValueError(f"Non-test identity found in test manifest: {row['identity_id']}")
    test_gallery_records = [
        {
            "identity_id": row["identity_id"],
            "canonical_path": row["canonical_path"],
            "transformation": "canonical",
        }
        for row in test_gallery_rows
    ]
    test_queries = sorted(
        test_manifest,
        key=lambda row: (row["identity_id"], TRANSFORMATION_NAMES.index(row["transformation"])),
    )
    test_encoded, test_encoding_timing, timing_batch = encode_records(
        test_gallery_records + test_queries,
        dataset_root,
        preprocess,
        model,
        device,
        EVALUATION_BATCH_SIZE,
        num_workers,
        seed + 3000,
    )
    test_gallery_embeddings = {
        key: value for key, value in test_encoded.items() if key[1] == "canonical"
    }
    test_query_embeddings = {
        key: value for key, value in test_encoded.items() if key[1] != "canonical"
    }
    test_gallery = {
        row["identity_id"]: test_gallery_embeddings[(row["identity_id"], "canonical")]
        for row in test_gallery_rows
    }
    if len(test_gallery) != 60 or any(
        int(sum(row["transformation"] == name for row in test_queries)) != 60
        for name in TRANSFORMATION_NAMES
    ):
        raise ValueError("Held-out test gallery/query count sanity check failed")
    test_metrics, _, _, _ = _score_split(
        test_queries, test_query_embeddings, test_gallery, threshold=threshold
    )

    default_baseline = manifest_dir.parent / "baseline_resnet18" / "summary.csv"
    baseline_path = baseline_summary or default_baseline
    if not baseline_path.is_file():
        raise ValueError(f"Frozen baseline summary not found: {baseline_path}")
    comparisons, comparison_outcomes = _baseline_comparison(baseline_path, test_metrics)

    pure_inference = measure_inference_latency(model, timing_batch, device)
    inference_parameters = sum(parameter.numel() for parameter in model.parameters())
    config.update(
        {
            "total_parameters": inference_parameters,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "best_checkpoint_epoch": best_validation["epoch"],
            "best_validation_overall_non_identity_recall_at_1": best_validation[
                "validation_overall_non_identity_recall_at_1"
            ],
            "validation_threshold": threshold_info,
            "baseline_summary_path": str(baseline_path),
            "optimizer": "AdamW",
            "loss": "supervised contrastive / NT-Xent-style over paired views",
            "evaluation_preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
        }
    )
    timing = {
        "optimization_runtime_seconds": optimization_seconds,
        "wall_runtime_seconds": wall_training_seconds,
        "inference_encoding_dataset_wait_and_preprocessing_seconds": sum(test_encoding_timing.values())
        - test_encoding_timing.get("forward_pass_seconds", 0.0)
        - test_encoding_timing.get("tensor_transfer_seconds", 0.0),
        "inference_encoding_forward_seconds": test_encoding_timing.get("forward_pass_seconds", 0.0),
        "pure_inference": pure_inference,
        "pure_inference_excludes_loading_preprocessing_and_transfer": True,
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "timing.json", timing)
    _write_csv(
        output_dir / "test_metrics.csv",
        test_metrics,
        (
            "transformation", "query_count", "gallery_count", "positive_pairs", "negative_pairs",
            "recall_at_1", "recall_at_3", "recall_at_5", "roc_auc", "f1", "threshold",
        ),
    )
    comparison_fields = ("transformation",) + tuple(
        field
        for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "roc_auc", "f1")
        for field in (f"baseline_{metric}", f"trained_{metric}", f"delta_{metric}")
    )
    _write_csv(output_dir / "baseline_comparison.csv", comparisons, comparison_fields)
    _write_json(
        output_dir / "comparison_summary.json",
        {**comparison_outcomes, "metric_comparisons": len(comparisons) * 5},
    )
    _write_report(
        output_dir / "training_report.md",
        config,
        history,
        best_validation,
        threshold_info,
        test_metrics,
        comparisons,
        comparison_outcomes,
        timing,
    )
    return {
        "config": config,
        "history": history,
        "best_validation": best_validation,
        "threshold": threshold_info,
        "test_metrics": test_metrics,
        "comparisons": comparisons,
        "comparison_outcomes": comparison_outcomes,
        "timing": timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--canonical-csv", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    args = parser.parse_args()
    try:
        results = run_training(
            args.dataset_root,
            args.canonical_csv,
            args.manifest_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            embedding_dim=args.embedding_dim,
            temperature=args.temperature,
            seed=args.seed,
            requested_device=args.device,
            num_workers=args.num_workers,
            baseline_summary=args.baseline_summary,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    print(f"Best epoch: {results['best_validation']['epoch']}")
    print(f"Best validation overall non-identity Recall@1: {results['best_validation']['validation_overall_non_identity_recall_at_1']:.4f}")
    print(f"Validation threshold: {results['threshold']['threshold']:.8f}")
    print(f"Training report: {args.output_dir / 'training_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())