"""Generate combined image metadata, duplicate reports, and a dataset summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .dataset_utils import build_metadata, build_summary, duplicate_groups, write_label_csv
except ImportError:
    from dataset_utils import build_metadata, build_summary, duplicate_groups, write_label_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="One dataset root; source is inferred conservatively")
    parser.add_argument("--source-dataset", choices=["auto", "deeplure", "kaggle", "unknown"], default="auto")
    parser.add_argument("--deeplure-root", type=Path, help="Optional external DeepLure dataset root")
    parser.add_argument("--kaggle-root", type=Path, help="Optional external Kaggle dataset root")
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--labels-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots: list[tuple[Path, str | None]] = []
    if args.data_root:
        roots.append((args.data_root, args.source_dataset))
    if args.deeplure_root:
        roots.append((args.deeplure_root, "deeplure"))
    if args.kaggle_root:
        roots.append((args.kaggle_root, "kaggle"))
    if not roots:
        parser.error("Provide --data-root, --deeplure-root, or --kaggle-root.")

    for root, _ in roots:
        if not root.is_dir():
            parser.error(f"Dataset root does not exist: {root}")
    labels_file = args.labels_file or args.metadata_dir / "design_labels.csv"
    output_paths = [args.metadata_dir.resolve(), args.reports_dir.resolve(), labels_file.resolve()]
    for root, _ in roots:
        resolved_root = root.resolve()
        if any(path == resolved_root or resolved_root in path.parents for path in output_paths):
            parser.error("Metadata/report output paths must be outside each dataset root.")

    metadata, non_images, empty_dirs = build_metadata(roots)
    duplicates = duplicate_groups(metadata)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_dir / "image_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    duplicates.to_csv(args.reports_dir / "duplicate_report.csv", index=False)
    summary = build_summary(metadata, non_images, empty_dirs, duplicates)
    (args.reports_dir / "dataset_summary.md").write_text(summary, encoding="utf-8")
    template_path = args.metadata_dir / "design_labels_template.csv"
    write_label_csv(template_path, metadata)
    write_label_csv(labels_file, metadata)
    write_label_csv(args.metadata_dir / "design_review.csv", metadata)

    print(f"Total images: {len(metadata)}")
    print(f"Readable images: {int((metadata['readable'] == True).sum())}")  # noqa: E712
    print(f"Unreadable images: {int((metadata['readable'] == False).sum())}")  # noqa: E712
    print("Images by source:")
    print(metadata["source_dataset"].value_counts(dropna=False).sort_index().to_string())
    print("Images by category:")
    print(metadata.groupby(["source_dataset", "source_category"], dropna=False).size().to_string())
    print("Images by extension:")
    print(metadata["extension"].value_counts().sort_index().to_string())
    if not metadata.empty:
        width = pd.to_numeric(metadata.loc[metadata["readable"], "width"])
        height = pd.to_numeric(metadata.loc[metadata["readable"], "height"])
        print(
            "Dimensions: "
            f"min={width.min()}x{height.min()}, max={width.max()}x{height.max()}, "
            f"average={width.mean():.1f}x{height.mean():.1f}"
        )
    print(f"Exact duplicate groups: {len(duplicates)}")
    if not duplicates.empty:
        print("WARNING: exact SHA256 duplicates found; see reports/duplicate_report.csv. No files were removed.")
    print(f"Metadata written to: {metadata_path}")
    print(f"Reports written to: {args.reports_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())