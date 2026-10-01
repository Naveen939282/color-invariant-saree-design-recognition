"""Evaluate grayscale baseline and/or trained embeddings on held-out splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .embedding_dataset import SareeImageDataset, build_transform
    from .embedding_model import GrayscaleResNet18Baseline, ResNet18Embedding
    from .evaluation_utils import (
        make_validation_pairs,
        retrieval_metrics,
        score_pairs,
        select_threshold,
        verification_metrics,
    )
except ImportError:
    from embedding_dataset import SareeImageDataset, build_transform
    from embedding_model import GrayscaleResNet18Baseline, ResNet18Embedding
    from evaluation_utils import (
        make_validation_pairs,
        retrieval_metrics,
        score_pairs,
        select_threshold,
        verification_metrics,
    )


def encode_frame(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    transform: Any,
    roots: dict[str, Path | None],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dataset = SareeImageDataset(frame, transform=transform, **roots)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings: dict[str, np.ndarray] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image_batch = batch["image"].to(device)
            vectors = model(image_batch).detach().cpu().numpy()
            for image_id, vector in zip(batch["image_id"], vectors):
                embeddings[image_id] = vector.astype(np.float32, copy=False)
    return embeddings


def load_trained_checkpoint(path: Path, device: torch.device) -> ResNet18Embedding:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = ResNet18Embedding(
        embedding_dim=int(config.get("embedding_dim", 128)),
        pretrained=False,
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeplure-root", type=Path)
    parser.add_argument("--kaggle-root", type=Path)
    parser.add_argument("--data-root", type=Path, help="Fallback for unknown/single-source rows")
    parser.add_argument("--validation-csv", type=Path, default=Path("metadata/validation.csv"))
    parser.add_argument("--gallery-csv", type=Path, default=Path("metadata/gallery.csv"))
    parser.add_argument("--query-csv", type=Path, default=Path("metadata/query.csv"))
    parser.add_argument("--pairs-csv", type=Path, default=Path("metadata/verification_pairs.csv"))
    parser.add_argument("--checkpoint", type=Path, help="Trained embedding checkpoint")
    parser.add_argument("--mode", choices=["baseline", "trained", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("local_outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def _evaluate_one(
    name: str,
    model: torch.nn.Module,
    transform: Any,
    validation: pd.DataFrame,
    gallery: pd.DataFrame,
    query: pd.DataFrame,
    evaluation_pairs: pd.DataFrame,
    roots: dict[str, Path | None],
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    all_rows = pd.concat([validation, gallery, query], ignore_index=True)
    if all_rows["image_id"].duplicated().any():
        raise ValueError("Validation/gallery/query splits contain duplicate image IDs.")
    embeddings = encode_frame(model, all_rows, transform, roots, device, batch_size)

    validation_pairs = make_validation_pairs(validation)
    validation_scored = score_pairs(validation_pairs, embeddings)
    threshold_selection = select_threshold(
        validation_scored["label"].astype(int).tolist(),
        validation_scored["similarity"].astype(float).tolist(),
    )
    threshold = threshold_selection["threshold"]
    scored_evaluation_pairs = score_pairs(evaluation_pairs, embeddings)
    verification = verification_metrics(scored_evaluation_pairs, threshold)
    retrieval, ranking = retrieval_metrics(query, gallery, embeddings)

    scored_evaluation_pairs.to_csv(output_dir / f"verification_{name}.csv", index=False)
    ranking.to_csv(output_dir / f"retrieval_{name}.csv", index=False)
    image_ids = list(embeddings)
    matrix = np.stack([embeddings[image_id] for image_id in image_ids])
    np.savez_compressed(
        output_dir / f"embeddings_{name}.npz",
        image_ids=np.asarray(image_ids),
        embeddings=matrix,
    )
    return {
        "model": name,
        "validation_threshold_selection": threshold_selection,
        "verification": verification,
        "retrieval": retrieval,
        "validation_pair_count": int(len(validation_pairs)),
        "evaluation_pair_count": int(len(evaluation_pairs)),
        "threshold_tuned_on_evaluation_pairs": False,
    }


def main(argv: list[str] | None = None, forced_mode: str | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = forced_mode or args.mode
    if args.batch_size < 1 or args.image_size < 32:
        parser.error("batch size must be positive and image size at least 32")
    if not any((args.data_root, args.deeplure_root, args.kaggle_root)):
        parser.error("Configure at least one external dataset root.")
    if mode in {"trained", "both"} and args.checkpoint is None:
        parser.error("--checkpoint is required for trained or both evaluation modes")

    roots = {
        "data_root": args.data_root,
        "deeplure_root": args.deeplure_root,
        "kaggle_root": args.kaggle_root,
    }
    output_dir = args.output_dir.resolve()
    for root in (args.data_root, args.deeplure_root, args.kaggle_root):
        if root and (root.resolve() == output_dir or root.resolve() in output_dir.parents):
            parser.error("Evaluation outputs must be outside all dataset roots.")
    device = _device(args.device)
    validation = pd.read_csv(args.validation_csv, dtype=str, keep_default_na=False)
    gallery = pd.read_csv(args.gallery_csv, dtype=str, keep_default_na=False)
    query = pd.read_csv(args.query_csv, dtype=str, keep_default_na=False)
    evaluation_pairs = pd.read_csv(args.pairs_csv, dtype=str, keep_default_na=False)
    required = {"image_id", "relative_path", "source_dataset", "design_id", "colorway"}
    for name, frame in (("validation", validation), ("gallery", gallery), ("query", query)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv missing required columns: {sorted(missing)}")
    pair_columns = {"image_id_1", "image_id_2", "label", "pair_type", "color_relationship"}
    if pair_columns - set(evaluation_pairs.columns):
        raise ValueError(f"Verification CSV missing columns: {sorted(pair_columns - set(evaluation_pairs.columns))}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, torch.nn.Module, Any]] = []
    if mode in {"baseline", "both"}:
        baseline = GrayscaleResNet18Baseline(pretrained=True)
        tasks.append(
            (
                "grayscale_resnet18",
                baseline.to(device).eval(),
                build_transform("grayscale", image_size=args.image_size),
            )
        )
    if mode in {"trained", "both"}:
        trained = load_trained_checkpoint(args.checkpoint, device)
        tasks.append(
            (
                "trained_color_aug_resnet18",
                trained,
                build_transform("color_aug", train=False, image_size=args.image_size),
            )
        )

    results: dict[str, Any] = {
        "device": str(device),
        "dataset_splits": {
            "validation_images": len(validation),
            "gallery_images": len(gallery),
            "query_images": len(query),
            "evaluation_pairs": len(evaluation_pairs),
        },
        "retrieval_metric_definition": "macro mean of per-query relevant-gallery Recall@K and reciprocal rank",
        "cross_color_definition": "same design_id with unequal nonblank manually labeled colorway strings",
        "evaluations": {},
    }
    comparison_rows = []
    for name, model, transform in tasks:
        evaluation = _evaluate_one(
            name,
            model,
            transform,
            validation,
            gallery,
            query,
            evaluation_pairs,
            roots,
            device,
            args.batch_size,
            output_dir,
        )
        results["evaluations"][name] = evaluation
        verification = evaluation["verification"]
        retrieval = evaluation["retrieval"]
        comparison_rows.append(
            {
                "model": name,
                "validation_threshold": evaluation["validation_threshold_selection"]["threshold"],
                "validation_f1": evaluation["validation_threshold_selection"]["validation_f1"],
                "recall_at_1": retrieval["recall@1"],
                "recall_at_3": retrieval["recall@3"],
                "recall_at_5": retrieval["recall@5"],
                "mrr": retrieval["mrr"],
                "cross_color_recall_at_1": retrieval["cross_color"]["cross_color_recall@1"],
                "roc_auc": verification["roc_auc"],
                "verification_f1": verification["f1"],
            }
        )
        print(
            f"{name}: threshold={evaluation['validation_threshold_selection']['threshold']} "
            f"ROC-AUC={verification['roc_auc']} Recall@1={retrieval['recall@1']} "
            f"cross-color Recall@1={retrieval['cross_color']['cross_color_recall@1']}"
        )
    results_path = output_dir / "evaluation_metrics.json"
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"Evaluation artifacts written to {output_dir}")
    print(f"Metrics: {results_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error