"""Shared image scanning, path resolution, and report helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCES = {"deeplure", "kaggle", "unknown"}
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


def stable_image_id(source_dataset: str, relative_path: str) -> str:
    identity = f"{source_dataset}\0{relative_path.replace(os.sep, '/')}"
    return "image_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def infer_source(root: Path, relative_path: Path, source_option: str | None) -> str:
    if source_option and source_option != "auto":
        return source_option
    path_parts = [part.casefold() for part in (*root.parts, *relative_path.parts)]
    for source in ("deeplure", "kaggle"):
        if any(source in part for part in path_parts):
            return source
    return "unknown"


def infer_category(relative_path: Path) -> str:
    parts = relative_path.parts
    if len(parts) < 2:
        return ""
    first = parts[0].casefold()
    if "deeplure" in first or "kaggle" in first:
        return parts[1] if len(parts) > 2 else ""
    return parts[0]


def scan_root(
    data_root: Path, source_option: str | None = "auto"
) -> tuple[list[dict], list[str], list[str]]:
    """Read supported image metadata without changing source files."""
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {data_root}")

    image_paths: list[Path] = []
    non_image_paths: list[str] = []
    directories: list[Path] = []
    for current, child_dirs, filenames in os.walk(data_root):
        current_path = Path(current)
        child_dirs[:] = sorted(child_dirs)
        directories.append(current_path)
        for filename in sorted(filenames, key=str.casefold):
            path = current_path / filename
            if path.suffix.casefold() in IMAGE_EXTENSIONS:
                image_paths.append(path)
            else:
                non_image_paths.append(path.relative_to(data_root).as_posix())

    image_paths.sort(key=lambda path: path.relative_to(data_root).as_posix().casefold())
    records = []
    for path in image_paths:
        relative_path = path.relative_to(data_root)
        relative_text = relative_path.as_posix()
        source = infer_source(data_root, relative_path, source_option)
        row = {
            "image_id": stable_image_id(source, relative_text),
            "relative_path": relative_text,
            "filename": path.name,
            "source_dataset": source,
            "source_category": infer_category(relative_path),
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
        except Exception as error:  # Pillow raises several decoder-specific exceptions.
            row["notes"] = f"Unreadable image: {error}"
        records.append(row)

    contains_files: dict[Path, bool] = {}
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        contains_files[directory] = any(
            (directory / name).is_file() for name in os.listdir(directory)
        ) or any(
            contains_files.get(child, False)
            for child in directory.iterdir()
            if child.is_dir()
        )
    empty_dirs = [
        path.relative_to(data_root).as_posix() or "."
        for path in directories
        if not contains_files[path]
    ]
    return records, sorted(non_image_paths), sorted(empty_dirs)


def build_metadata(
    roots: Iterable[tuple[Path, str | None]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    all_records: list[dict] = []
    all_non_images: list[str] = []
    all_empty_dirs: list[str] = []
    for root, source_option in roots:
        records, non_images, empty_dirs = scan_root(root, source_option)
        all_records.extend(records)
        prefix = source_option if source_option and source_option != "auto" else root.name
        all_non_images.extend(f"{prefix}/{path}" for path in non_images)
        all_empty_dirs.extend(f"{prefix}/{path}" for path in empty_dirs)

    metadata = pd.DataFrame(all_records, columns=METADATA_COLUMNS)
    if metadata["image_id"].duplicated().any():
        raise ValueError("Stable image_id collision detected across input roots.")
    return metadata, all_non_images, all_empty_dirs


def write_label_csv(path: Path, metadata: pd.DataFrame) -> None:
    """Create missing label rows while preserving prior human annotations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prior_by_id: dict[str, dict] = {}
    if path.exists():
        prior = pd.read_csv(path, dtype=str, keep_default_na=False)
        if {"image_id", "relative_path"}.issubset(prior.columns):
            prior_by_id = {
                row["image_id"]: row
                for row in prior.to_dict("records")
                if row.get("relative_path")
            }

    rows = []
    for item in metadata[["image_id", "relative_path"]].to_dict("records"):
        old = prior_by_id.get(item["image_id"], {})
        row = {column: "" for column in LABEL_COLUMNS}
        row.update(item)
        if old.get("relative_path") == item["relative_path"]:
            row.update({column: old.get(column, "") for column in LABEL_COLUMNS})
        rows.append(row)
    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(path, index=False)


def duplicate_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["sha256", "duplicate_count", "image_ids", "relative_paths"]
    if metadata.empty:
        return pd.DataFrame(columns=columns)
    duplicates = metadata[metadata["sha256"].ne("")].groupby("sha256", sort=True)
    rows = []
    for digest, group in duplicates:
        if len(group) > 1:
            rows.append(
                {
                    "sha256": digest,
                    "duplicate_count": len(group),
                    "image_ids": json.dumps(group["image_id"].tolist()),
                    "relative_paths": json.dumps(
                        [
                            f"{source}:{path}"
                            for source, path in zip(
                                group["source_dataset"], group["relative_path"]
                            )
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    headers = [str(column) for column in frame.columns]
    rows = [headers, ["---"] * len(headers)]
    for values in frame.itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|").replace("\n", " ") for value in values])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def build_summary(
    metadata: pd.DataFrame,
    non_images: list[str],
    empty_dirs: list[str],
    duplicates: pd.DataFrame,
) -> str:
    readable = metadata[metadata["readable"] == True]  # noqa: E712
    unreadable = metadata[metadata["readable"] == False]  # noqa: E712
    numeric_dimensions = readable[["width", "height"]].apply(pd.to_numeric, errors="coerce")
    dimensions = numeric_dimensions.describe().round(3) if not readable.empty else pd.DataFrame()
    numeric_aspect = pd.to_numeric(readable["aspect_ratio"], errors="coerce")
    aspect = (
        numeric_aspect.describe().round(4)
        if not readable.empty
        else pd.Series(dtype=float)
    )
    source_counts = metadata["source_dataset"].value_counts(dropna=False).sort_index()
    category_counts = (
        metadata.groupby(["source_dataset", "source_category"], dropna=False)
        .size()
        .rename("images")
        .reset_index()
    )
    extension_counts = metadata["extension"].value_counts().sort_index()
    all_extensions = [*metadata["extension"].tolist(), *(Path(path).suffix.lower() or "(no extension)" for path in non_images)]
    file_extensions = pd.Series(all_extensions).value_counts().sort_index()

    lines = [
        "# Dataset Summary",
        "",
        "## Dataset Sources",
        "",
        _markdown_table(source_counts.rename("images").rename_axis("source_dataset").reset_index()),
        "",
        "Kaggle remains a separate source. No source or category is reinterpreted as a saree type.",
        "",
        "## Image Counts",
        "",
        f"- Total images: {len(metadata)}",
        f"- Readable images: {len(readable)}",
        f"- Unreadable images: {len(unreadable)}",
        "",
        "## Dimensions and Aspect Ratios",
        "",
        "Dimensions (width and height):",
        "",
        _markdown_table(dimensions.reset_index().rename(columns={"index": "statistic"}))
        if not dimensions.empty
        else "(no readable images)",
        "",
        "Aspect ratio (width / height):",
        "",
        _markdown_table(aspect.rename_axis("statistic").reset_index())
        if not aspect.empty
        else "(no readable images)",
        "",
        "## Categories",
        "",
        _markdown_table(category_counts.rename(columns={"source_category": "category"})),
        "",
        "## Duplicates",
        "",
        f"Exact SHA256 duplicate groups: {len(duplicates)}",
        "",
        _markdown_table(duplicates),
        "",
        "Duplicate files are reported only; nothing is deleted automatically. Exact byte duplicates are not proof of matching designs.",
        "",
        "## Corrupted Images",
        "",
        _markdown_table(unreadable[["source_dataset", "relative_path", "notes"]]),
        "",
        "## Empty Directories",
        "",
        "\n".join(f"- `{path}`" for path in empty_dirs) or "(none)",
        "",
        "## File Types",
        "",
        _markdown_table(extension_counts.rename("images").rename_axis("extension").reset_index()),
        "",
        "All file extensions, including non-images:",
        "",
        _markdown_table(file_extensions.rename("files").rename_axis("extension").reset_index()),
        "",
        f"Non-image files detected: {len(non_images)}",
        "",
        "## Important Limitations",
        "",
        "Image metadata, filenames, folders, colors, and embedding similarity do **not** establish design identity. Design IDs require human confirmation of the underlying textile motif.",
        "",
    ]
    return "\n".join(lines)


def resolve_image_path(
    row: dict, data_root: Path | None, deeplure_root: Path | None, kaggle_root: Path | None
) -> Path:
    source = row.get("source_dataset", "unknown")
    root = {
        "deeplure": deeplure_root,
        "kaggle": kaggle_root,
    }.get(source)
    if root is None:
        root = data_root
    if root is None:
        raise ValueError(f"No local data root configured for source {source!r}.")
    relative_path = Path(row["relative_path"])
    candidate = (root / relative_path).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise ValueError(f"Image path escapes configured data root: {relative_path}")
    return candidate