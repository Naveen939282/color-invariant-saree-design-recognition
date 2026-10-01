"""Audit an image directory and create local metadata and review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
METADATA_COLUMNS = [
    "image_id",
    "relative_path",
    "filename",
    "source_dataset",
    "source_category",
    "extension",
    "file_size_bytes",
    "width",
    "height",
    "channels",
    "aspect_ratio",
    "sha256",
    "readable",
    "notes",
]
LABEL_COLUMNS = [
    "image_id",
    "relative_path",
    "design_id",
    "colorway",
    "label_confidence",
    "label_source",
    "notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_image_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"image_{digest}"


def source_details(
    relative_path: Path, data_root: Path, source_option: str
) -> tuple[str, str]:
    parts = relative_path.parts
    source_markers = {"deeplure", "kaggle"}
    marker_index = next(
        (index for index, part in enumerate(parts) if part.casefold() in source_markers),
        None,
    )
    root_marker = data_root.name.casefold()

    if source_option != "auto":
        source = source_option
    elif marker_index is not None:
        source = parts[marker_index].casefold()
    elif root_marker in source_markers:
        source = root_marker
    else:
        source = "unknown"

    if marker_index is not None and marker_index + 1 < len(parts) - 1:
        category = parts[marker_index + 1]
    elif marker_index is not None:
        category = ""
    else:
        category = relative_path.parent.name if len(parts) > 1 else ""
    return source, category


def scan_data_root(
    data_root: Path, output_dir: Path, source_option: str
) -> tuple[list[dict], list[str], list[str]]:
    image_paths: list[Path] = []
    non_image_paths: list[str] = []
    directories: list[Path] = []
    output_resolved = output_dir.resolve()
    output_is_inside_root = output_resolved != data_root and data_root in output_resolved.parents

    for current, child_dirs, filenames in os.walk(data_root):
        current_path = Path(current)
        child_dirs[:] = sorted(
            name
            for name in child_dirs
            if not output_is_inside_root
            or (
                (current_path / name).resolve() != output_resolved
                and output_resolved not in (current_path / name).resolve().parents
            )
        )
        if output_is_inside_root and (
            current_path.resolve() == output_resolved
            or output_resolved in current_path.resolve().parents
        ):
            continue
        directories.append(current_path)
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix.casefold() in IMAGE_EXTENSIONS:
                image_paths.append(path)
            else:
                non_image_paths.append(path.relative_to(data_root).as_posix())

    image_paths.sort(key=lambda path: path.relative_to(data_root).as_posix().casefold())
    rows = []
    for path in image_paths:
        relative_path = path.relative_to(data_root)
        relative_text = relative_path.as_posix()
        source, category = source_details(relative_path, data_root, source_option)
        row = {
            "image_id": stable_image_id(relative_text),
            "relative_path": relative_text,
            "filename": path.name,
            "source_dataset": source,
            "source_category": category,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "width": "",
            "height": "",
            "channels": "",
            "aspect_ratio": "",
            "sha256": sha256_file(path),
            "readable": False,
            "notes": "",
        }
        try:
            with Image.open(path) as image:
                image.load()
                row.update(
                    width=image.width,
                    height=image.height,
                    channels=len(image.getbands()),
                    aspect_ratio=round(image.width / image.height, 6),
                    readable=True,
                )
        except (OSError, ValueError, UnidentifiedImageError) as error:
            row["notes"] = f"Unreadable image: {error}"
        rows.append(row)

    has_files: dict[Path, bool] = {}
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        has_files[directory] = any(
            (directory / name).is_file()
            for name in os.listdir(directory)
            if not output_is_inside_root
            or (
                (directory / name).resolve() != output_resolved
                and output_resolved not in (directory / name).resolve().parents
            )
        ) or any(has_files.get(child, False) for child in directory.iterdir() if child.is_dir())
    empty_dirs = [
        path.relative_to(data_root).as_posix() or "."
        for path in directories
        if not has_files.get(path, False)
    ]
    return rows, sorted(non_image_paths), sorted(empty_dirs)


def write_label_file(path: Path, metadata: pd.DataFrame) -> None:
    current = pd.DataFrame(columns=LABEL_COLUMNS)
    if path.exists():
        prior = pd.read_csv(path, dtype=str, keep_default_na=False)
        if {"image_id", "relative_path"}.issubset(prior.columns):
            current = prior
    labels = metadata[["image_id", "relative_path"]].copy()
    labels = labels.merge(
        current.drop_duplicates("image_id").drop(
            columns=[column for column in current.columns if column not in LABEL_COLUMNS],
            errors="ignore",
        ),
        on=["image_id", "relative_path"],
        how="left",
        suffixes=("", "_prior"),
    )
    for column in LABEL_COLUMNS:
        if column not in labels:
            labels[column] = ""
    labels[LABEL_COLUMNS].fillna("").to_csv(path, index=False)


def write_contact_sheet(data_root: Path, output_dir: Path, metadata: pd.DataFrame, page_size: int) -> None:
    page_dir = output_dir / "design_review_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    pages = [metadata.iloc[start : start + page_size] for start in range(0, len(metadata), page_size)]
    page_links = [
        f'<a href="design_review_pages/page_{page_number:03d}.html">Page {page_number}</a>'
        for page_number in range(1, len(pages) + 1)
    ]
    for page_number, page in enumerate(pages, start=1):
        filename = f"page_{page_number:03d}.html"
        cards = []
        for row in page.to_dict("records"):
            image_path = data_root / Path(row["relative_path"])
            relative_image = os.path.relpath(image_path, page_dir).replace(os.sep, "/")
            image_url = quote(relative_image, safe="/:._-!")
            cards.append(
                "<figure>"
                f'<img loading="lazy" src="{html.escape(image_url, quote=True)}" '
                f'alt="{html.escape(row["filename"], quote=True)}">'
                f'<figcaption><strong>{html.escape(row["image_id"])}</strong><br>'
                f'{html.escape(row["filename"])}</figcaption></figure>'
            )
        (page_dir / filename).write_text(
            "<!doctype html><html><meta charset=\"utf-8\"><title>Design review</title>"
            "<style>body{font:14px sans-serif;margin:24px;color:#222}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px}"
            "figure{margin:0;border:1px solid #ccc;padding:8px}img{width:100%;height:220px;object-fit:contain}"
            "figcaption{overflow-wrap:anywhere;margin-top:8px}</style>"
            f"<h1>Design review, page {page_number}</h1><nav>{' | '.join(page_links)}</nav>"
            f"<main class=\"grid\">{''.join(cards)}</main></html>",
            encoding="utf-8",
        )
    (output_dir / "design_review.html").write_text(
        "<!doctype html><html><meta charset=\"utf-8\"><title>Design review pages</title>"
        "<h1>Design review</h1><p>Images are suggestions for human inspection only; no design labels are assigned.</p>"
        f"<nav>{' | '.join(page_links) if page_links else 'No images found.'}</nav></html>",
        encoding="utf-8",
    )


def format_counts(frame: pd.DataFrame, column: str) -> str:
    counts = frame[column].replace("", "(blank)").value_counts(dropna=False).sort_index()
    return "\n".join(f"  {name}: {count}" for name, count in counts.items()) or "  (none)"


def build_summary(metadata: pd.DataFrame, non_images: list[str], empty_dirs: list[str]) -> str:
    readable = metadata[metadata["readable"] == True]  # noqa: E712
    corrupt = metadata[metadata["readable"] == False]  # noqa: E712
    dimension_stats = readable[["width", "height"]].describe().to_string() if not readable.empty else "  (no readable images)"
    aspect_stats = readable["aspect_ratio"].describe().to_string() if not readable.empty else "  (no readable images)"
    duplicate_groups = [group for _, group in metadata.groupby("sha256") if len(group) > 1]
    duplicate_text = "\n".join(
        "  " + ", ".join(group["relative_path"].tolist()) for group in duplicate_groups
    ) or "  (none)"
    all_extensions = pd.Series(
        [*metadata["extension"].tolist(), *(Path(path).suffix.lower() or "(no extension)" for path in non_images)]
    )
    extension_counts = all_extensions.value_counts().sort_index()
    extensions_text = "\n".join(
        f"  {extension}: {count}" for extension, count in extension_counts.items()
    ) or "  (none)"
    return "\n".join(
        [
            "Dataset audit summary",
            f"Total images: {len(metadata)}",
            f"Readable images: {len(readable)}",
            "Images by source:",
            format_counts(metadata, "source_dataset"),
            "Images by category:",
            format_counts(metadata, "source_category"),
            "Image dimensions (width and height):",
            dimension_stats,
            "Aspect-ratio statistics:",
            aspect_stats,
            f"Corrupted/unreadable images ({len(corrupt)}):",
            "\n".join(f"  {row.relative_path}: {row.notes}" for row in corrupt.itertuples()) or "  (none)",
            f"Duplicate groups by SHA256 ({len(duplicate_groups)}):",
            duplicate_text,
            f"Empty folders ({len(empty_dirs)}):",
            "\n".join(f"  {path}" for path in empty_dirs) or "  (none)",
            "File extensions:",
            extensions_text,
            f"Non-image files ({len(non_images)}):",
            "\n".join(f"  {path}" for path in non_images) or "  (none)",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Extracted dataset directory to scan")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_audit_output"))
    parser.add_argument(
        "--source-dataset",
        choices=["auto", "deeplure", "kaggle", "unknown"],
        default="auto",
        help="Explicitly identify a single-source root; auto only trusts path components named deeplure/kaggle",
    )
    parser.add_argument("--page-size", type=int, choices=[25, 50], default=25)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        parser.error(f"--data-root must be an extracted directory: {data_root}")
    output_dir = args.output_dir.resolve()
    if output_dir == data_root:
        parser.error("--output-dir must differ from --data-root")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, non_images, empty_dirs = scan_data_root(data_root, output_dir, args.source_dataset)
    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    if metadata["image_id"].duplicated().any():
        raise RuntimeError("Stable image_id collision detected; no output was written.")

    metadata.to_csv(output_dir / "image_metadata.csv", index=False)
    write_label_file(output_dir / "design_labels_template.csv", metadata)
    write_label_file(output_dir / "design_review.csv", metadata)
    summary = build_summary(metadata, non_images, empty_dirs)
    (output_dir / "dataset_summary.txt").write_text(summary, encoding="utf-8")
    write_contact_sheet(data_root, output_dir, metadata, args.page_size)
    print(summary)
    print(f"\nArtifacts written to: {output_dir}")


if __name__ == "__main__":
    main()