"""Evaluate a frozen ImageNet ResNet18 on the controlled color transforms."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from public_kaggle_experiment.scripts.create_color_invariance_manifest import (  # noqa: E402
    read_canonical_rows,
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
    apply_color_transform,
)
from public_kaggle_experiment.src.resnet18_baseline import (  # noqa: E402
    FrozenResNet18Baseline,
)


EXPECTED_IDENTITIES = {"train": 431, "valid": 115, "test": 60}
EXPECTED_GALLERY_SIZE = 60
EXPECTED_QUERIES_PER_TRANSFORM = 60
IDENTITY_TRANSFORM = "identity"
NON_IDENTITY_TRANSFORMS = tuple(
    name for name in TRANSFORMATION_NAMES if name != IDENTITY_TRANSFORM
)
WARMUP_RUNS = 10
TIMED_RUNS = 50


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset-relative image path: {value}")
    return relative


class CanonicalTransformDataset(Dataset):
    """Read canonical images and apply their manifest transform on demand."""

    def __init__(self, rows: list[dict[str, str]], dataset_root: Path, preprocess: Any):
        self.rows = rows
        self.dataset_root = dataset_root.resolve()
        self.preprocess = preprocess
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def _read_bytes(self, relative_path: str) -> bytes:
        relative = _safe_relative_path(relative_path)
        if self.dataset_root.is_file() and self.dataset_root.suffix.casefold() == ".zip":
            if self._archive is None:
                self._archive = zipfile.ZipFile(self.dataset_root)
            try:
                return self._archive.read(relative.as_posix())
            except KeyError as error:
                raise ValueError(f"Canonical image is missing from archive: {relative_path}") from error

        image_path = (self.dataset_root / Path(*relative.parts)).resolve()
        try:
            image_path.relative_to(self.dataset_root)
        except ValueError as error:
            raise ValueError(f"Canonical image escapes dataset root: {relative_path}") from error
        return image_path.read_bytes()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.rows[index]
        try:
            with Image.open(io.BytesIO(self._read_bytes(row["canonical_path"]))) as opened:
                opened.load()
                image = opened.convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode canonical image {row['canonical_path']}: {error}") from error

        transformation = row.get("transformation", "canonical")
        if transformation != "canonical":
            image = apply_color_transform(image, transformation)
        return self.preprocess(image), row["identity_id"], transformation

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Required manifest file not found: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def validate_split_integrity(canonical_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = {split: set() for split in EXPECTED_IDENTITIES}
    source_ids: set[str] = set()
    for row in canonical_rows:
        identity_id = row["identity_id"]
        if identity_id != row["source_id"]:
            raise ValueError(f"Identity/source ID mismatch: {identity_id}")
        if identity_id in source_ids:
            raise ValueError(f"More than one canonical image for identity: {identity_id}")
        source_ids.add(identity_id)
        by_split[row["split"]].add(identity_id)
    for split, expected_count in EXPECTED_IDENTITIES.items():
        if len(by_split[split]) != expected_count:
            raise ValueError(
                f"Expected {expected_count} canonical identities in {split}, got {len(by_split[split])}"
            )
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = by_split[first] & by_split[second]
        if overlap:
            raise ValueError(f"Candidate identities cross {first}/{second}: {sorted(overlap)[:5]}")
    return by_split


def validate_transform_manifest(
    rows: list[dict[str, str]],
    canonical_by_id: dict[str, dict[str, str]],
    split: str,
    expected_query_count: int,
) -> None:
    if not rows:
        raise ValueError(f"No {split} query specifications found")
    expected_ids = {
        identity_id
        for identity_id, row in canonical_by_id.items()
        if row["split"] == split
    }
    per_transform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        transformation = row.get("transformation", "")
        if transformation not in TRANSFORMATION_NAMES:
            raise ValueError(f"Unknown {split} transformation: {transformation}")
        identity_id = row["identity_id"]
        canonical = canonical_by_id.get(identity_id)
        if canonical is None or canonical["split"] != split:
            raise ValueError(f"{split} query identity is not in the canonical split: {identity_id}")
        if row["source_id"] != identity_id or row["canonical_path"] != canonical["canonical_path"]:
            raise ValueError(f"Query specification does not match canonical identity: {identity_id}")
        expected_config = json.dumps(
            TRANSFORMATION_CONFIG[transformation], sort_keys=True, separators=(",", ":")
        )
        if row["transformation_config_json"] != expected_config:
            raise ValueError(f"Transformation config changed for {transformation}")
        if row.get("transformed_path", ""):
            raise ValueError("Baseline must transform canonical images on demand, not use saved variants")
        per_transform[transformation].append(row)

    if set(per_transform) != set(TRANSFORMATION_NAMES):
        raise ValueError(f"{split} manifest does not contain all expected transformations")
    for transformation, transform_rows in per_transform.items():
        ids = [row["identity_id"] for row in transform_rows]
        if len(ids) != expected_query_count or set(ids) != expected_ids or len(set(ids)) != len(ids):
            raise ValueError(
                f"{split}/{transformation} must have one query for each of "
                f"{expected_query_count} identities"
            )


def validate_test_gallery(
    gallery_rows: list[dict[str, str]], canonical_by_id: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    expected = {
        identity_id: row
        for identity_id, row in canonical_by_id.items()
        if row["split"] == "test"
    }
    if len(gallery_rows) != EXPECTED_GALLERY_SIZE:
        raise ValueError(
            f"Expected exactly {EXPECTED_GALLERY_SIZE} test gallery identities, got {len(gallery_rows)}"
        )
    if len({row["identity_id"] for row in gallery_rows}) != EXPECTED_GALLERY_SIZE:
        raise ValueError("Test gallery contains duplicate identities")
    for row in gallery_rows:
        canonical = expected.get(row["identity_id"])
        if canonical is None or row["source_id"] != row["identity_id"]:
            raise ValueError(f"Gallery identity does not match a test canonical: {row['identity_id']}")
        if row["canonical_path"] != canonical["canonical_path"]:
            raise ValueError(f"Gallery path differs from canonical path: {row['identity_id']}")
    return sorted(gallery_rows, key=lambda row: row["identity_id"])


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu or auto")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, and auto")
    return device


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def encode_records(
    rows: list[dict[str, str]],
    dataset_root: Path,
    preprocess: Any,
    model: FrozenResNet18Baseline,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, float], torch.Tensor]:
    dataset = CanonicalTransformDataset(rows, dataset_root, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    embeddings: dict[tuple[str, str], torch.Tensor] = {}
    elapsed = Counter()
    first_input: torch.Tensor | None = None
    iterator = iter(loader)
    try:
        while True:
            wait_started = time.perf_counter()
            try:
                images, identity_ids, transformations = next(iterator)
            except StopIteration:
                break
            elapsed["dataset_wait_seconds"] += time.perf_counter() - wait_started

            transfer_started = time.perf_counter()
            images = images.to(device, non_blocking=device.type == "cuda")
            _synchronize(device)
            elapsed["tensor_transfer_seconds"] += time.perf_counter() - transfer_started
            if first_input is None:
                first_input = images[: min(batch_size, len(images))].detach()

            _synchronize(device)
            inference_started = time.perf_counter()
            with torch.inference_mode():
                output = model(images)
            _synchronize(device)
            elapsed["forward_pass_seconds"] += time.perf_counter() - inference_started
            if output.ndim != 2 or output.shape[1] != model.embedding_dim:
                raise ValueError(f"Unexpected embedding shape: {tuple(output.shape)}")
            if not torch.isfinite(output).all():
                raise ValueError("Model emitted non-finite embeddings")
            norms = output.norm(p=2, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                raise ValueError("Model emitted embeddings that are not L2-normalized")
            for identity_id, transformation, embedding in zip(
                identity_ids, transformations, output.detach().cpu()
            ):
                key = (identity_id, transformation)
                if key in embeddings:
                    raise ValueError(f"Duplicate encoded identity/transformation: {key}")
                embeddings[key] = embedding
    finally:
        dataset.close()
    if first_input is None:
        raise ValueError("Cannot encode an empty image set")
    return dict(embeddings), dict(elapsed), first_input


def measure_inference_latency(
    model: FrozenResNet18Baseline,
    input_batch: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int]:
    for _ in range(WARMUP_RUNS):
        with torch.inference_mode():
            model(input_batch)
    _synchronize(device)
    durations_ms: list[float] = []
    for _ in range(TIMED_RUNS):
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            model(input_batch)
        _synchronize(device)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    median_ms = statistics.median(durations_ms)
    return {
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "batch_size": int(input_batch.shape[0]),
        "median_batch_latency_ms": median_ms,
        "mean_batch_latency_ms": statistics.mean(durations_ms),
        "median_latency_per_image_ms": median_ms / int(input_batch.shape[0]),
    }


def _embeddings_for(
    rows: list[dict[str, str]],
    encoded: dict[tuple[str, str], torch.Tensor],
    gallery: dict[str, torch.Tensor],
    transformation: str,
) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
    query_ids = [row["identity_id"] for row in rows]
    query_vectors = torch.stack([encoded[(identity_id, transformation)] for identity_id in query_ids])
    gallery_ids = list(gallery)
    gallery_vectors = torch.stack([gallery[identity_id] for identity_id in gallery_ids])
    return query_vectors, query_ids, gallery_vectors, gallery_ids


def _score_transformation(
    rows: list[dict[str, str]],
    encoded: dict[tuple[str, str], torch.Tensor],
    gallery: dict[str, torch.Tensor],
    transformation: str,
    threshold: float,
) -> tuple[dict[str, Any], torch.Tensor, np.ndarray, np.ndarray]:
    query_vectors, query_ids, gallery_vectors, gallery_ids = _embeddings_for(
        rows, encoded, gallery, transformation
    )
    assert_query_gallery_identity_match(query_ids, gallery_ids)
    similarities = cosine_similarity_matrix(query_vectors, gallery_vectors)
    recall = retrieval_recall_at_k(similarities, query_ids, gallery_ids, (1, 3, 5))
    scores, labels = build_verification_scores(similarities, query_ids, gallery_ids)
    verification = verification_metrics(scores, labels, threshold)
    retrieval_row = {
        "transformation": transformation,
        "query_count": len(query_ids),
        "gallery_count": len(gallery_ids),
        "recall_at_1": recall[1],
        "recall_at_3": recall[3],
        "recall_at_5": recall[5],
    }
    verification_row = {
        "transformation": transformation,
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int(len(labels) - labels.sum()),
        "roc_auc": verification["roc_auc"],
        "f1": verification["f1"],
        "threshold": verification["threshold"],
    }
    return retrieval_row | verification_row, similarities, scores, labels


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _preprocessing_config(preprocess: Any) -> dict[str, Any]:
    values: dict[str, Any] = {"transform_repr": repr(preprocess)}
    for attribute in ("resize_size", "crop_size", "mean", "std", "antialias"):
        value = getattr(preprocess, attribute, None)
        if value is not None:
            values[attribute] = list(value) if isinstance(value, (tuple, list)) else value
    interpolation = getattr(preprocess, "interpolation", None)
    if interpolation is not None:
        values["interpolation"] = str(interpolation)
    return values


def _write_baseline_report(
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    threshold_info: dict[str, Any],
    config: dict[str, Any],
    timing: dict[str, Any],
) -> None:
    lines = [
        "# Frozen ResNet18 Controlled Color-Invariance Baseline",
        "",
        "This is a controlled synthetic color-invariance evaluation over candidate filename-derived identities, not verified designs or real colorways. It is not evidence of real-world cross-color saree recognition.",
        "",
        f"- Weights: `{config['weights']}`",
        f"- Device: `{config['device']}`; batch size: {config['batch_size']}",
        f"- Embedding: native {config['embedding_dimension']}-D pooled ResNet18 feature, L2-normalized; no projection",
        f"- Parameters: {config['feature_extractor_parameters']:,} used / {config['original_resnet18_parameters']:,} standard ResNet18 including discarded classifier; trainable {config['trainable_parameters']:,}",
        f"- Global validation-selected threshold: {threshold_info['threshold']:.8f} (validation F1 {threshold_info['validation_f1']:.4f})",
        f"- Pure forward median latency: {timing['pure_inference']['median_batch_latency_ms']:.3f} ms/batch ({timing['pure_inference']['median_latency_per_image_ms']:.3f} ms/image), after {timing['pure_inference']['warmup_runs']} warm-ups and {timing['pure_inference']['timed_runs']} timed runs",
        "",
        "| Transformation | Recall@1 | Recall@3 | Recall@5 | ROC-AUC | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['transformation']} | {row['recall_at_1']:.4f} | {row['recall_at_3']:.4f} | "
            f"{row['recall_at_5']:.4f} | {row['roc_auc']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The single verification threshold maximizes F1 on all validation non-identity query/gallery pairs pooled together; ties use the largest threshold. The identity control is excluded from threshold selection and reported with the same frozen threshold. No test labels are used to select or change the threshold.",
            "",
            "Preprocessing is the transform attached to the selected torchvision ImageNet weights (resize, center crop, RGB tensor conversion, and ImageNet normalization); exact parameters are recorded in `configuration.json`.",
            "",
            "Results use only the public Kaggle canonical images and fixed synthetic transforms. Candidate source IDs are filename-based, not verified design IDs. The identity transformation is an unchanged control and is not a color-invariance result.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_baseline(
    canonical_csv: Path,
    manifest_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    requested_device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    if dataset_root.is_file() and dataset_root.suffix.casefold() != ".zip":
        raise ValueError("Dataset root file must be a ZIP archive")

    canonical_rows = read_canonical_rows(canonical_csv)
    split_ids = validate_split_integrity(canonical_rows)
    canonical_by_id = {row["identity_id"]: row for row in canonical_rows}
    validation_manifest = _load_csv(manifest_dir / "valid_manifest.csv")
    test_manifest = _load_csv(manifest_dir / "test_manifest.csv")
    validate_transform_manifest(validation_manifest, canonical_by_id, "valid", 115)
    validate_transform_manifest(test_manifest, canonical_by_id, "test", EXPECTED_QUERIES_PER_TRANSFORM)
    gallery_rows = validate_test_gallery(
        _load_csv(manifest_dir / "test_gallery.csv"), canonical_by_id
    )
    test_ids = [row["identity_id"] for row in gallery_rows]
    for transformation in TRANSFORMATION_NAMES:
        transform_ids = [
            row["identity_id"] for row in test_manifest if row["transformation"] == transformation
        ]
        assert_query_gallery_identity_match(
            transform_ids, test_ids, expected_gallery_size=EXPECTED_GALLERY_SIZE
        )
        if len(transform_ids) != EXPECTED_QUERIES_PER_TRANSFORM:
            raise ValueError(f"Expected 60 {transformation} test queries, got {len(transform_ids)}")
    if len(split_ids["test"]) != EXPECTED_GALLERY_SIZE:
        raise ValueError("Test gallery identity count does not match the expected 60")

    device = resolve_device(requested_device)
    set_deterministic_seed(seed)
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = FrozenResNet18Baseline(weights=weights).to(device)
    model.eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen baseline invariant failed: model must be eval with no trainable params")

    validation_gallery_records = [
        {
            "identity_id": row["identity_id"],
            "canonical_path": row["canonical_path"],
            "transformation": "canonical",
        }
        for row in canonical_rows
        if row["split"] == "valid"
    ]
    test_gallery_records = [
        {
            "identity_id": row["identity_id"],
            "canonical_path": row["canonical_path"],
            "transformation": "canonical",
        }
        for row in gallery_rows
    ]
    validation_queries = sorted(
        validation_manifest, key=lambda row: (row["identity_id"], TRANSFORMATION_NAMES.index(row["transformation"]))
    )
    test_queries = sorted(
        test_manifest, key=lambda row: (row["identity_id"], TRANSFORMATION_NAMES.index(row["transformation"]))
    )

    all_encoding_times: dict[str, float] = defaultdict(float)
    validation_gallery_embeddings, timings, _ = encode_records(
        validation_gallery_records,
        dataset_root,
        preprocess,
        model,
        device,
        batch_size,
        num_workers,
        seed,
    )
    for key, value in timings.items():
        all_encoding_times[key] += value
    validation_query_embeddings, timings, _ = encode_records(
        validation_queries,
        dataset_root,
        preprocess,
        model,
        device,
        batch_size,
        num_workers,
        seed + 1,
    )
    for key, value in timings.items():
        all_encoding_times[key] += value
    test_gallery_embeddings, timings, timing_batch = encode_records(
        test_gallery_records,
        dataset_root,
        preprocess,
        model,
        device,
        batch_size,
        num_workers,
        seed + 2,
    )
    for key, value in timings.items():
        all_encoding_times[key] += value
    test_query_embeddings, timings, _ = encode_records(
        test_queries,
        dataset_root,
        preprocess,
        model,
        device,
        batch_size,
        num_workers,
        seed + 3,
    )
    for key, value in timings.items():
        all_encoding_times[key] += value

    val_gallery = {
        identity_id: validation_gallery_embeddings[(identity_id, "canonical")]
        for identity_id in sorted(split_ids["valid"])
    }
    test_gallery = {
        identity_id: test_gallery_embeddings[(identity_id, "canonical")]
        for identity_id in sorted(split_ids["test"])
    }
    validation_scores: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    for transformation in NON_IDENTITY_TRANSFORMS:
        rows = [row for row in validation_queries if row["transformation"] == transformation]
        vectors, query_ids, gallery_vectors, gallery_ids = _embeddings_for(
            rows, validation_query_embeddings, val_gallery, transformation
        )
        assert_query_gallery_identity_match(query_ids, gallery_ids, expected_gallery_size=115)
        similarities = cosine_similarity_matrix(vectors, gallery_vectors)
        scores, labels = build_verification_scores(similarities, query_ids, gallery_ids)
        validation_scores.append(scores)
        validation_labels.append(labels)
    threshold, validation_f1 = select_validation_f1_threshold(
        np.concatenate(validation_scores), np.concatenate(validation_labels)
    )
    threshold_info = {
        "threshold": threshold,
        "validation_f1": validation_f1,
        "selection_split": "valid",
        "selection_transformations": list(NON_IDENTITY_TRANSFORMS),
        "identity_control_excluded": True,
        "method": "maximize pooled validation F1 over all non-identity query/gallery cosine scores",
        "tie_break": "largest threshold among thresholds with equal validation F1",
        "validation_pair_count": int(sum(len(scores) for scores in validation_scores)),
        "test_labels_used_for_selection": False,
    }

    retrieval_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    non_identity_scores: list[np.ndarray] = []
    non_identity_labels: list[np.ndarray] = []
    non_identity_similarity_rows: list[torch.Tensor] = []
    non_identity_query_ids: list[str] = []
    for transformation in TRANSFORMATION_NAMES:
        rows = [row for row in test_queries if row["transformation"] == transformation]
        result, similarities, scores, labels = _score_transformation(
            rows, test_query_embeddings, test_gallery, transformation, threshold
        )
        retrieval_rows.append(
            {key: result[key] for key in ("transformation", "query_count", "gallery_count", "recall_at_1", "recall_at_3", "recall_at_5")}
        )
        verification_rows.append(
            {key: result[key] for key in ("transformation", "positive_pairs", "negative_pairs", "roc_auc", "f1", "threshold")}
        )
        if transformation != IDENTITY_TRANSFORM:
            non_identity_scores.append(scores)
            non_identity_labels.append(labels)
            non_identity_similarity_rows.append(similarities)
            non_identity_query_ids.extend(row["identity_id"] for row in rows)

    pooled_similarities = torch.cat(non_identity_similarity_rows, dim=0)
    pooled_query_ids = non_identity_query_ids
    pooled_gallery_ids = list(test_gallery)
    pooled_recall = retrieval_recall_at_k(
        pooled_similarities, pooled_query_ids, pooled_gallery_ids, (1, 3, 5)
    )
    pooled_scores = np.concatenate(non_identity_scores)
    pooled_labels = np.concatenate(non_identity_labels)
    pooled_verification = verification_metrics(pooled_scores, pooled_labels, threshold)
    overall_row = {
        "transformation": "overall_non_identity",
        "query_count": len(pooled_query_ids),
        "gallery_count": len(test_gallery),
        "recall_at_1": pooled_recall[1],
        "recall_at_3": pooled_recall[3],
        "recall_at_5": pooled_recall[5],
        "positive_pairs": int(pooled_labels.sum()),
        "negative_pairs": int(len(pooled_labels) - pooled_labels.sum()),
        "roc_auc": pooled_verification["roc_auc"],
        "f1": pooled_verification["f1"],
        "threshold": threshold,
    }
    summary_rows = [
        {
            **retrieval,
            **next(
                row for row in verification_rows
                if row["transformation"] == retrieval["transformation"]
            ),
        }
        for retrieval in retrieval_rows
    ] + [overall_row]

    pure_inference = measure_inference_latency(model, timing_batch, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    config = {
        "architecture": "torchvision.models.resnet18",
        "weights": f"ResNet18_Weights.{weights.name}",
        "weights_url": weights.url,
        "classifier": "replaced with Identity; use pooled feature representation before classification",
        "projection": "none; native 512-dimensional feature retained",
        "embedding_dimension": model.embedding_dim,
        "normalization": "L2",
        "frozen": True,
        "total_parameters": feature_parameters,
        "feature_extractor_parameters": feature_parameters,
        "original_resnet18_parameters": model.original_resnet18_parameters,
        "trainable_parameters": trainable_parameters,
        "preprocessing": _preprocessing_config(preprocess),
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "training_images_used": 0,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "pillow_version": PIL.__version__,
        "transformation_parameters": TRANSFORMATION_CONFIG,
        "threshold_selection": threshold_info,
    }
    timing = {
        "encoding_dataset_wait_and_preprocessing_seconds": all_encoding_times["dataset_wait_seconds"],
        "encoding_tensor_transfer_seconds": all_encoding_times["tensor_transfer_seconds"],
        "encoding_forward_pass_seconds": all_encoding_times["forward_pass_seconds"],
        "pure_inference": pure_inference,
        "pure_inference_excludes_file_loading_preprocessing_and_tensor_transfer": True,
    }

    retrieval_fields = (
        "transformation", "query_count", "gallery_count", "recall_at_1", "recall_at_3", "recall_at_5"
    )
    verification_fields = (
        "transformation", "positive_pairs", "negative_pairs", "roc_auc", "f1", "threshold"
    )
    summary_fields = retrieval_fields + ("positive_pairs", "negative_pairs", "roc_auc", "f1", "threshold")
    _write_csv(output_dir / "retrieval_per_transformation.csv", retrieval_rows, retrieval_fields)
    _write_csv(output_dir / "verification_per_transformation.csv", verification_rows, verification_fields)
    _write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    (output_dir / "configuration.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "selected_validation_threshold.json").write_text(
        json.dumps(threshold_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results = {
        "config": config,
        "threshold": threshold_info,
        "timing": timing,
        "retrieval": retrieval_rows,
        "verification": verification_rows,
        "summary": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    _write_baseline_report(
        output_dir / "baseline_report.md", summary_rows, threshold_info, config, timing
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-csv", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto (CPU fallback)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        results = run_baseline(
            args.canonical_csv,
            args.manifest_dir,
            args.dataset_root,
            args.output_dir,
            args.device,
            args.batch_size,
            args.num_workers,
            args.seed,
        )
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(f"Frozen baseline complete on {results['config']['device']}")
    print(f"Results: {args.output_dir / 'baseline_report.md'}")
    for row in results["summary"]:
        print(
            f"{row['transformation']}: R@1={row['recall_at_1']:.4f} "
            f"R@3={row['recall_at_3']:.4f} R@5={row['recall_at_5']:.4f} "
            f"AUC={row['roc_auc']:.4f} F1={row['f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())