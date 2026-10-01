"""Create design-disjoint train, validation, gallery, query, and pair CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .split_utils import create_splits, load_labeled_dataset, validate_split_frames
except ImportError:
    from split_utils import create_splits, load_labeled_dataset, validate_split_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("metadata/image_metadata.csv"))
    parser.add_argument("--labels", type=Path, default=Path("metadata/design_labels.csv"))
    parser.add_argument("--data-root", type=Path, help="Fallback root for unknown or single-source data")
    parser.add_argument("--deeplure-root", type=Path)
    parser.add_argument("--kaggle-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-class", type=int, default=10000)
    parser.add_argument("--minimum-pairs-per-class", type=int, default=1)
    args = parser.parse_args(argv)
    if args.minimum_pairs_per_class < 1:
        parser.error("--minimum-pairs-per-class must be positive")

    dataset = load_labeled_dataset(
        args.metadata,
        args.labels,
        args.data_root,
        args.deeplure_root,
        args.kaggle_root,
    )
    splits = create_splits(
        dataset,
        args.seed,
        args.validation_fraction,
        args.evaluation_fraction,
        args.max_pairs_per_class,
    )
    warnings = validate_split_frames(
        splits,
        min_positive_pairs=args.minimum_pairs_per_class,
        min_negative_pairs=args.minimum_pairs_per_class,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(
        "Split complete: "
        f"train={len(splits['train'])}, validation={len(splits['validation'])}, "
        f"gallery={len(splits['gallery'])}, query={len(splits['query'])}, "
        f"positive_pairs={(splits['verification_pairs']['label'] == 1).sum()}, "
        f"negative_pairs={(splits['verification_pairs']['label'] == 0).sum()}"
    )
    print(f"Files written to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error