"""Train a ResNet18 metric-learning embedding from prepared splits."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader

try:
    from .embedding_dataset import (
        COLOR_AUGMENTATION,
        ContrastivePairDataset,
        SareeImageDataset,
        build_transform,
    )
    from .embedding_model import ResNet18Embedding, contrastive_loss
except ImportError:
    from embedding_dataset import (
        COLOR_AUGMENTATION,
        ContrastivePairDataset,
        SareeImageDataset,
        build_transform,
    )
    from embedding_model import ResNet18Embedding, contrastive_loss


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch random generators for repeatable runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return torch.device(requested)


def run_epoch(
    model: ResNet18Embedding,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    margin: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_pairs = 0
    for first, second, same_design in loader:
        first = first.to(device)
        second = second.to(device)
        same_design = same_design.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            embeddings = model(torch.cat((first, second), dim=0))
            first_embedding, second_embedding = embeddings.chunk(2, dim=0)
            loss = contrastive_loss(first_embedding, second_embedding, same_design, margin)
            if training:
                loss.backward()
                optimizer.step()
        pair_count = len(same_design)
        total_loss += float(loss.detach()) * pair_count
        total_pairs += pair_count
    if total_pairs == 0:
        raise ValueError("No image pairs were available for this epoch.")
    return total_loss / total_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("metadata/train.csv"))
    parser.add_argument("--validation-csv", type=Path, default=Path("metadata/validation.csv"))
    parser.add_argument("--data-root", type=Path, help="Fallback root for unknown/single-source metadata")
    parser.add_argument("--deeplure-root", type=Path)
    parser.add_argument("--kaggle-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("local_outputs/model"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of pairs per batch")
    parser.add_argument("--pairs-per-epoch", type=int, default=256)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from ImageNet weights")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers cannot be negative")
    if args.learning_rate <= 0 or args.embedding_dim < 1 or args.margin <= 0:
        parser.error("learning rate, embedding dimension, and margin must be positive")
    if args.pairs_per_epoch < 2 or args.validation_pairs < 2:
        parser.error("pair counts must be at least 2")
    if not any((args.data_root, args.deeplure_root, args.kaggle_root)):
        parser.error("Configure at least one external dataset root.")

    roots = [root.resolve() for root in (args.data_root, args.deeplure_root, args.kaggle_root) if root]
    output_dir = args.output_dir.resolve()
    if any(root == output_dir or root in output_dir.parents for root in roots):
        parser.error("Model output must be outside all dataset roots.")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    image_roots = {
        "data_root": args.data_root,
        "deeplure_root": args.deeplure_root,
        "kaggle_root": args.kaggle_root,
    }
    train_images = SareeImageDataset(
        args.train_csv,
        build_transform("color_aug", train=True, image_size=args.image_size, horizontal_flip=args.horizontal_flip),
        **image_roots,
    )
    validation_images = SareeImageDataset(
        args.validation_csv,
        build_transform("color_aug", train=False, image_size=args.image_size),
        **image_roots,
    )
    train_pairs = ContrastivePairDataset(
        train_images, pairs_per_epoch=args.pairs_per_epoch, seed=args.seed
    )
    validation_pairs = ContrastivePairDataset(
        validation_images, pairs_per_epoch=args.validation_pairs, seed=args.seed + 1
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ResNet18Embedding(
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "ResNet18Embedding",
        "pretrained": not args.no_pretrained,
        "pretrained_weights": "ResNet18_Weights.DEFAULT" if not args.no_pretrained else None,
        "embedding_dim": args.embedding_dim,
        "backbone_frozen": not args.unfreeze_backbone,
        "learning_rate": args.learning_rate,
        "batch_size_pairs": args.batch_size,
        "pairs_per_epoch": train_pairs.pairs_per_epoch,
        "validation_pairs": validation_pairs.pairs_per_epoch,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "deterministic_algorithms": "enabled with warn_only",
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "num_workers": args.num_workers,
        "optimizer": "AdamW",
        "loss": "contrastive_euclidean",
        "margin": args.margin,
        "training_transform": {
            "mode": "color_aug",
            "resize_crop_scale": [0.9, 1.0],
            "resize_crop_ratio": [0.95, 1.05],
            "color_jitter": COLOR_AUGMENTATION,
            "horizontal_flip": args.horizontal_flip,
            "image_size": args.image_size,
        },
        "validation_transform": "resize_short_side_then_center_crop_rgb",
        "train_split": str(args.train_csv),
        "validation_split": str(args.validation_csv),
        "evaluation_split_used_for_training": False,
    }
    history: list[dict] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    print(
        f"device={device} train_images={len(train_images)} "
        f"validation_images={len(validation_images)} "
        f"train_designs={train_images.frame['design_id'].nunique()} "
        f"validation_designs={validation_images.frame['design_id'].nunique()}"
    )

    for epoch in range(args.epochs):
        started = time.perf_counter()
        train_pairs.set_epoch(epoch)
        train_loss = run_epoch(model, train_loader, device, optimizer, args.margin)
        validation_loss = run_epoch(model, validation_loader, device, None, args.margin)
        elapsed = time.perf_counter() - started
        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": elapsed,
        }
        history.append(history_row)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} elapsed_seconds={elapsed:.1f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            checkpoint = {
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "model_config": {
                    "embedding_dim": args.embedding_dim,
                    "pretrained": not args.no_pretrained,
                    "freeze_backbone": not args.unfreeze_backbone,
                },
                "training_config": config,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
    config["best_epoch"] = best_epoch
    config["best_validation_loss"] = best_validation_loss
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"Best checkpoint: {output_dir / 'best_model.pt'} (epoch {best_epoch})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error