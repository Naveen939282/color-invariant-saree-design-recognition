"""Evaluate an already validation-selected contrastive checkpoint on held-out test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torchvision.models import ResNet18_Weights

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from public_kaggle_experiment.scripts.create_color_invariance_manifest import (  # noqa: E402
    read_canonical_rows,
)
from public_kaggle_experiment.scripts.evaluate_resnet18_baseline import (  # noqa: E402
    EXPECTED_IDENTITIES,
    _load_csv,
    encode_records,
    measure_inference_latency,
    resolve_device,
    set_deterministic_seed,
    validate_test_gallery,
    validate_transform_manifest,
)
from public_kaggle_experiment.scripts.train_color_invariant import (  # noqa: E402
    _baseline_comparison,
    _score_split,
    _write_csv,
    _write_json,
    _write_report,
)
from public_kaggle_experiment.src.color_transforms import TRANSFORMATION_NAMES  # noqa: E402
from public_kaggle_experiment.src.contrastive_learning import validate_identity_splits  # noqa: E402
from public_kaggle_experiment.src.trained_model import ColorInvariantEmbeddingModel  # noqa: E402


def _read_history(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted.append(
            {
                "epoch": int(row["epoch"]),
                "mean_training_loss": float(row["mean_training_loss"]),
                "validation_overall_non_identity_recall_at_1": float(
                    row["validation_overall_non_identity_recall_at_1"]
                ),
                "checkpoint_selected": int(row["checkpoint_selected"]),
                "training_seconds": float(row["training_seconds"]),
                "validation_seconds": float(row["validation_seconds"]),
                "validation_identity_control_recall_at_1": float(
                    row["validation_identity_control_recall_at_1"]
                ),
            }
        )
    return converted


def evaluate_selected_checkpoint(
    dataset_root: Path,
    canonical_csv: Path,
    manifest_dir: Path,
    output_dir: Path,
    requested_device: str = "auto",
    num_workers: int = 0,
    seed: int = 42,
    baseline_summary: Path | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    output_dir = output_dir.resolve()
    required_files = (
        output_dir / "best_model.pt",
        output_dir / "config.json",
        output_dir / "training_history.csv",
        output_dir / "validation_metrics.csv",
        output_dir / "selected_validation_threshold.json",
    )
    missing = [path.name for path in required_files if not path.is_file()]
    if missing:
        raise ValueError(f"Selected checkpoint artifacts are missing: {missing}")

    with (output_dir / "config.json").open(encoding="utf-8") as source:
        config = json.load(source)
    with (output_dir / "selected_validation_threshold.json").open(encoding="utf-8") as source:
        threshold_info = json.load(source)
    history = _read_history(output_dir / "training_history.csv")
    if config.get("test_used_for_training_or_model_selection") is not False:
        raise ValueError("Training config does not certify test exclusion")
    if threshold_info.get("selection_split") != "valid" or threshold_info.get("test_labels_used") is not False:
        raise ValueError("Saved threshold was not selected exclusively from validation")
    best_epoch = int(threshold_info["checkpoint_epoch"])
    if best_epoch not in {row["epoch"] for row in history}:
        raise ValueError("Selected checkpoint epoch is not present in training history")
    best_validation = {
        "epoch": best_epoch,
        "validation_overall_non_identity_recall_at_1": float(
            history[best_epoch - 1]["validation_overall_non_identity_recall_at_1"]
        ),
    }

    canonical_rows = read_canonical_rows(canonical_csv)
    split_ids = validate_identity_splits(canonical_rows)
    if {split: len(identities) for split, identities in split_ids.items()} != EXPECTED_IDENTITIES:
        raise ValueError("Canonical candidate identity counts differ from the established split protocol")
    canonical_by_id = {row["identity_id"]: row for row in canonical_rows}

    device = resolve_device(requested_device)
    set_deterministic_seed(seed)
    model = ColorInvariantEmbeddingModel(
        weights=None, embedding_dim=int(config["embedding_dim"])
    ).to(device)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(
            output_dir / "best_model.pt", map_location=device, weights_only=True
        )
    if int(checkpoint["epoch"]) != best_epoch:
        raise ValueError("Checkpoint epoch differs from validation-selected threshold checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Test evaluation model unexpectedly has trainable parameters")

    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()

    # The selected checkpoint and validation threshold are verified before test manifests are read.
    test_manifest = _load_csv(manifest_dir / "test_manifest.csv")
    test_gallery_rows = validate_test_gallery(
        _load_csv(manifest_dir / "test_gallery.csv"), canonical_by_id
    )
    validate_transform_manifest(test_manifest, canonical_by_id, "test", 60)
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
    encoded, test_encoding_timing, timing_batch = encode_records(
        test_gallery_records + test_queries,
        dataset_root,
        preprocess,
        model,  # Same encoder interface; model is frozen for test inference.
        device,
        int(config["evaluation_batch_size"]),
        num_workers,
        seed + 3000,
    )
    test_gallery = {
        row["identity_id"]: encoded[(row["identity_id"], "canonical")]
        for row in test_gallery_rows
    }
    if len(test_gallery) != 60:
        raise ValueError(f"Expected 60 canonical test gallery identities, got {len(test_gallery)}")
    for transformation in TRANSFORMATION_NAMES:
        count = sum(row["transformation"] == transformation for row in test_queries)
        if count != 60:
            raise ValueError(f"Expected 60 test queries for {transformation}, got {count}")
    test_metrics, _, _, _ = _score_split(
        test_queries, encoded, test_gallery, threshold=float(threshold_info["threshold"])
    )

    default_baseline = manifest_dir.parent / "baseline_resnet18" / "summary.csv"
    baseline_path = baseline_summary or default_baseline
    if not baseline_path.is_file():
        raise ValueError(f"Frozen baseline summary not found: {baseline_path}")
    comparisons, comparison_outcomes = _baseline_comparison(baseline_path, test_metrics)
    inference_timing = measure_inference_latency(model, timing_batch, device)

    optimization_runtime = sum(row["training_seconds"] for row in history)
    validation_epoch_runtime = sum(row["validation_seconds"] for row in history)
    timing = {
        "optimization_runtime_seconds": optimization_runtime,
        "validation_epoch_runtime_seconds": validation_epoch_runtime,
        "training_and_validation_epoch_seconds": optimization_runtime + validation_epoch_runtime,
        "training_and_validation_epoch_seconds_excludes_initialization_threshold_and_final_test": True,
        "inference_encoding_dataset_wait_and_preprocessing_seconds": max(
            0.0,
            sum(test_encoding_timing.values())
            - test_encoding_timing.get("forward_pass_seconds", 0.0)
            - test_encoding_timing.get("tensor_transfer_seconds", 0.0),
        ),
        "inference_encoding_forward_seconds": test_encoding_timing.get("forward_pass_seconds", 0.0),
        "pure_inference": inference_timing,
        "pure_inference_excludes_loading_preprocessing_and_transfer": True,
    }
    config.update(
        {
            "device": str(device),
            "num_workers": num_workers,
            "best_checkpoint_epoch": best_epoch,
            "best_validation_overall_non_identity_recall_at_1": best_validation[
                "validation_overall_non_identity_recall_at_1"
            ],
            "validation_threshold": threshold_info,
            "baseline_summary_path": str(baseline_path),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "inference_trainable_parameters": 0,
            "training_trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    args = parser.parse_args()
    try:
        results = evaluate_selected_checkpoint(
            args.dataset_root,
            args.canonical_csv,
            args.manifest_dir,
            args.output_dir,
            args.device,
            args.num_workers,
            args.seed,
            args.baseline_summary,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    print(f"Evaluated selected epoch {results['best_validation']['epoch']}")
    print(f"Frozen validation threshold: {results['threshold']['threshold']:.8f}")
    print(f"Report: {args.output_dir / 'training_report.md'}")
    for row in results["test_metrics"]:
        print(
            f"{row['transformation']}: R@1={row['recall_at_1']:.4f} "
            f"R@3={row['recall_at_3']:.4f} R@5={row['recall_at_5']:.4f} "
            f"AUC={row['roc_auc']:.4f} F1={row['f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())