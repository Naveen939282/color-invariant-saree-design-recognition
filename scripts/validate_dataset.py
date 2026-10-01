"""Validate image metadata, manual labels, and optional generated splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .split_utils import (
        PAIR_COLUMNS,
        load_labeled_dataset,
        validate_metadata_only,
        validate_split_frames,
    )
except ImportError:
    from split_utils import (
        PAIR_COLUMNS,
        load_labeled_dataset,
        validate_metadata_only,
        validate_split_frames,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("metadata/image_metadata.csv"))
    parser.add_argument("--labels", type=Path, help="Completed human design label CSV")
    parser.add_argument("--data-root", type=Path, help="Fallback root for unknown/single-source data")
    parser.add_argument("--deeplure-root", type=Path)
    parser.add_argument("--kaggle-root", type=Path)
    parser.add_argument("--split-dir", type=Path, help="Validate previously generated split CSVs")
    args = parser.parse_args(argv)
    roots = {
        "deeplure": args.deeplure_root or args.data_root,
        "kaggle": args.kaggle_root or args.data_root,
        "unknown": args.data_root,
    }
    warnings = validate_metadata_only(args.metadata, roots)
    if args.labels:
        dataset = load_labeled_dataset(
            args.metadata,
            args.labels,
            args.data_root,
            args.deeplure_root,
            args.kaggle_root,
        )
        singleton_count = int((dataset.groupby("design_id").size() == 1).sum())
        if singleton_count:
            warnings.append(f"{singleton_count} designs have only one image and cannot produce positive pairs.")
        if dataset["colorway"].str.strip().eq("").any():
            warnings.append("Some colorway labels are blank; related verification pairs will be marked unknown.")
        normalized = dataset.assign(_color=dataset["colorway"].str.strip().str.casefold())
        variants = normalized.groupby(["design_id", "_color"])["colorway"].nunique()
        if (variants > 1).any():
            warnings.append("Colorway spelling differs only by case/whitespace for some designs; normalize manually.")

    if args.split_dir:
        names = ["train", "validation", "gallery", "query", "verification_pairs"]
        frames = {}
        for name in names:
            path = args.split_dir / f"{name}.csv"
            if not path.is_file():
                raise ValueError(f"Missing split file: {path}")
            frames[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing_pair_columns = set(PAIR_COLUMNS) - set(frames["verification_pairs"].columns)
        if missing_pair_columns:
            raise ValueError(f"Verification pairs are missing columns: {sorted(missing_pair_columns)}")
        warnings.extend(validate_split_frames(frames))

    for warning in warnings:
        print(f"WARNING: {warning}")
    print("Dataset validation passed." if not warnings else "Dataset validation passed with warnings.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error