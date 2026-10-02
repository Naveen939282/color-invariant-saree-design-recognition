"""Audit the public Roboflow saree dataset without modifying source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


EXPECTED_CLASSES = ("Banarasi", "Bandhani", "Ikat", "Pichwai")
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}(?=\.[^.]+$)", re.IGNORECASE)
SOURCE_EXTENSION_SUFFIX = re.compile(r"_(jpg|jpeg|png)(\.[^.]+)$", re.IGNORECASE)
GENERIC_FILENAME = re.compile(r"^(?:image|images)\d*\.[^.]+$", re.IGNORECASE)
METADATA_FIELDS = (
    "split",
    "class_name",
    "relative_path",
    "filename",
    "width",
    "height",
    "mode",
    "file_size",
    "md5",
    "source_id",
    "normalized_source_filename",
    "source_variant_index",
    "source_variant_count",
)


def normalize_source_filename(filename: str) -> str:
    """Remove only Roboflow's hash, then normalize its encoded source extension."""
    normalized = RF_SUFFIX.sub("", PurePosixPath(filename).name)
    normalized = SOURCE_EXTENSION_SUFFIX.sub(r".\1", normalized)
    return normalized.casefold()


def make_source_id(class_name: str, filename: str) -> str:
    """Return a class-scoped candidate ID, not a verified design identity."""
    return f"{class_name.casefold()}::{normalize_source_filename(filename)}"


def _dataset_paths(dataset_root: Path) -> tuple[list[tuple[str, str, str, str]], set[str]]:
    """Return (split, class, relative path, filename) entries and discovered classes."""
    paths: list[tuple[str, str, str, str]] = []
    classes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}

    if dataset_root.is_file() and dataset_root.suffix.casefold() == ".zip":
        try:
            with zipfile.ZipFile(dataset_root) as archive:
                names = archive.namelist()
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError(f"Cannot read dataset ZIP: {dataset_root}: {error}") from error
        for member in names:
            parts = PurePosixPath(member).parts
            if len(parts) < 3 or parts[0] not in SPLITS:
                continue
            split, class_name = parts[0], parts[1]
            classes_by_split[split].add(class_name)
            if len(parts) == 3 and Path(parts[2]).suffix.casefold() in IMAGE_SUFFIXES:
                paths.append((split, class_name, member, parts[2]))
    elif dataset_root.is_dir():
        for split in SPLITS:
            split_root = dataset_root / split
            if not split_root.is_dir():
                continue
            for class_dir in split_root.iterdir():
                if not class_dir.is_dir():
                    continue
                classes_by_split[split].add(class_dir.name)
                for item in class_dir.iterdir():
                    if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES:
                        relative_path = item.relative_to(dataset_root).as_posix()
                        paths.append((split, class_dir.name, relative_path, item.name))
    else:
        raise ValueError(
            f"Dataset root must be an extracted directory or .zip file: {dataset_root}"
        )

    missing_splits = [split for split in SPLITS if not classes_by_split[split]]
    if missing_splits:
        raise ValueError(
            "Dataset is missing split folders or class directories: "
            + ", ".join(missing_splits)
        )
    for split, found in classes_by_split.items():
        if found != set(EXPECTED_CLASSES):
            missing = sorted(set(EXPECTED_CLASSES) - found)
            unexpected = sorted(found - set(EXPECTED_CLASSES))
            raise ValueError(
                f"Unexpected classes in {split}: missing={missing}, unexpected={unexpected}; "
                f"expected exactly {list(EXPECTED_CLASSES)}"
            )
    return sorted(paths), set.union(*classes_by_split.values())


def discover_classes(dataset_root: Path) -> set[str]:
    """Discover class directories across the required split folders."""
    _, classes = _dataset_paths(Path(dataset_root))
    return classes


def create_metadata(dataset_root: Path) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    """Read image metadata and bytes from an extracted dataset or ZIP archive."""
    dataset_root = Path(dataset_root)
    entries, _ = _dataset_paths(dataset_root)
    if dataset_root.is_file():
        with zipfile.ZipFile(dataset_root) as archive:
            payloads = {relative: archive.read(relative) for _, _, relative, _ in entries}
    else:
        payloads = {
            relative: (dataset_root / Path(*PurePosixPath(relative).parts)).read_bytes()
            for _, _, relative, _ in entries
        }

    rows: list[dict[str, object]] = []
    for split, class_name, relative_path, filename in entries:
        data = payloads[relative_path]
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height = image.size
                mode = image.mode
        except (OSError, ValueError) as error:
            raise ValueError(f"Unreadable image {relative_path}: {error}") from error
        normalized_filename = normalize_source_filename(filename)
        rows.append(
            {
                "split": split,
                "class_name": class_name,
                "relative_path": relative_path,
                "filename": filename,
                "width": width,
                "height": height,
                "mode": mode,
                "file_size": len(data),
                "md5": hashlib.md5(data).hexdigest(),
                "source_id": make_source_id(class_name, filename),
                "normalized_source_filename": normalized_filename,
                "source_variant_index": 0,
                "source_variant_count": 0,
            }
        )

    source_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        source_groups[str(row["source_id"])].append(row)
    for group_rows in source_groups.values():
        group_rows.sort(key=lambda row: (str(row["split"]), str(row["relative_path"]).casefold()))
        for index, row in enumerate(group_rows, start=1):
            row["source_variant_index"] = index
            row["source_variant_count"] = len(group_rows)
    return rows, payloads


def group_by_field(
    rows: Iterable[dict[str, object]], field: str
) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return dict(groups)


def find_exact_duplicate_groups(
    rows: Iterable[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    return {
        md5: group
        for md5, group in group_by_field(rows, "md5").items()
        if len(group) > 1
    }


def find_split_overlaps(
    rows: Iterable[dict[str, object]], identity_field: str
) -> dict[str, list[str]]:
    splits_by_identity: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_identity[str(row[identity_field])].add(str(row["split"]))
    overlaps: dict[str, list[str]] = {}
    for first, second in combinations(SPLITS, 2):
        overlaps[f"{first}/{second}"] = sorted(
            identity
            for identity, splits in splits_by_identity.items()
            if first in splits and second in splits
        )
    return overlaps


def _pixel_similarity(
    group_rows: list[dict[str, object]], payloads: dict[str, bytes]
) -> tuple[int, int, float | None, float | None]:
    if len(group_rows) < 2:
        return 0, 0, None, None
    reference_row = group_rows[0]
    with Image.open(io.BytesIO(payloads[str(reference_row["relative_path"])])) as image:
        reference = np.asarray(image.convert("RGB"), dtype=np.uint8)
    changed_fractions: list[float] = []
    mean_absolute_differences: list[float] = []
    identical_pixels = 0
    comparisons = 0
    for row in group_rows[1:]:
        with Image.open(io.BytesIO(payloads[str(row["relative_path"])])) as image:
            current = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if current.shape != reference.shape:
            continue
        comparisons += 1
        delta = np.abs(current.astype(np.int16) - reference.astype(np.int16))
        changed_fractions.append(float(np.any(delta > 16, axis=2).mean()))
        mean_absolute_differences.append(float(delta.mean()))
        identical_pixels += int(np.array_equal(current, reference))
    return (
        comparisons,
        identical_pixels,
        statistics.median(changed_fractions) if changed_fractions else None,
        statistics.median(mean_absolute_differences) if mean_absolute_differences else None,
    )


def analyze_source_groups(
    rows: list[dict[str, object]], payloads: dict[str, bytes]
) -> list[dict[str, object]]:
    source_groups = group_by_field(rows, "source_id")
    exact_by_md5 = group_by_field(rows, "md5")
    results: list[dict[str, object]] = []
    for source_id, group_rows in sorted(source_groups.items()):
        group_rows.sort(key=lambda row: (str(row["split"]), str(row["relative_path"]).casefold()))
        comparisons, identical, changed_fraction, mean_difference = _pixel_similarity(
            group_rows, payloads
        )
        md5_groups = sum(
            1
            for duplicate_rows in exact_by_md5.values()
            if len(duplicate_rows) > 1
            and any(row["source_id"] == source_id for row in duplicate_rows)
        )
        normalized_name = str(group_rows[0]["normalized_source_filename"])
        results.append(
            {
                "source_id": source_id,
                "class_name": group_rows[0]["class_name"],
                "normalized_source_filename": normalized_name,
                "image_count": len(group_rows),
                "splits": ",".join(sorted({str(row["split"]) for row in group_rows})),
                "relative_paths": " | ".join(str(row["relative_path"]) for row in group_rows),
                "reference_comparisons": comparisons,
                "identical_decoded_comparisons": identical,
                "median_fraction_pixels_rgb_delta_gt_16": changed_fraction,
                "median_mean_absolute_rgb_difference": mean_difference,
                "generic_basename": bool(GENERIC_FILENAME.fullmatch(normalized_name)),
                "exact_duplicate_hash_groups": md5_groups,
            }
        )
    return results


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _thumbnail(data: bytes, size: tuple[int, int]) -> Image.Image:
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
        image.thumbnail(size)
        return image


def _ascii_caption(value: str) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")


def write_contact_sheets(
    rows: list[dict[str, object]],
    payloads: dict[str, bytes],
    output_dir: Path,
    seed: int,
    samples_per_class: int,
    variant_group_count: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cell_width, cell_height, margin = 210, 250, 12
    class_sheet = Image.new(
        "RGB",
        (margin + samples_per_class * cell_width, margin + len(EXPECTED_CLASSES) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(class_sheet)
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        candidates = sorted(
            (row for row in rows if row["class_name"] == class_name),
            key=lambda row: str(row["relative_path"]).casefold(),
        )
        chosen = rng.sample(candidates, min(samples_per_class, len(candidates)))
        row_top = margin + class_index * cell_height
        draw.text((margin, row_top), class_name, fill="black")
        for sample_index, row in enumerate(chosen):
            x = margin + sample_index * cell_width
            y = row_top + 22
            thumbnail = _thumbnail(payloads[str(row["relative_path"])], (190, 190))
            class_sheet.paste(thumbnail, (x, y))
            draw.text((x, y + 194), _ascii_caption(str(row["filename"]))[:28], fill="black")
            draw.text((x, y + 211), str(row["split"]), fill="gray")
    class_path = output_dir / "class_contact_sheet.png"
    class_sheet.save(class_path)

    source_groups = group_by_field(rows, "source_id")
    eligible = sorted(
        (group for group in source_groups.values() if len(group) > 1),
        key=lambda group: str(group[0]["source_id"]),
    )
    chosen_groups = rng.sample(eligible, min(variant_group_count, len(eligible)))
    columns = min(4, max((len(group) for group in chosen_groups), default=1))
    variant_sheet = Image.new(
        "RGB",
        (margin + columns * cell_width, margin + max(1, len(chosen_groups)) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(variant_sheet)
    for group_index, group in enumerate(chosen_groups):
        group.sort(key=lambda row: (str(row["split"]), str(row["relative_path"]).casefold()))
        row_top = margin + group_index * cell_height
        label = f"{group[0]['source_id']} ({len(group)} candidates)"
        draw.text((margin, row_top), _ascii_caption(label)[:110], fill="black")
        for variant_index, row in enumerate(group[:columns]):
            x = margin + variant_index * cell_width
            y = row_top + 22
            thumbnail = _thumbnail(payloads[str(row["relative_path"])], (190, 190))
            variant_sheet.paste(thumbnail, (x, y))
            draw.text((x, y + 194), f"variant {row['source_variant_index']} / {row['split']}", fill="black")
    variant_path = output_dir / "variant_contact_sheet.png"
    variant_sheet.save(variant_path)
    return class_path, variant_path


def _format_counter(counter: Counter[object]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}: {counter[key]}" for key in sorted(counter, key=str))


def write_summary_report(
    output_path: Path,
    rows: list[dict[str, object]],
    source_groups: list[dict[str, object]],
    exact_duplicates: dict[str, list[dict[str, object]]],
    source_overlaps: dict[str, list[str]],
    exact_overlaps: dict[str, list[str]],
    seed: int,
) -> None:
    by_split = Counter(str(row["split"]) for row in rows)
    by_class = Counter(str(row["class_name"]) for row in rows)
    by_split_class = Counter((str(row["split"]), str(row["class_name"])) for row in rows)
    dimensions = Counter(f"{row['width']}x{row['height']}" for row in rows)
    modes = Counter(str(row["mode"]) for row in rows)
    group_sizes = Counter(int(group["image_count"]) for group in source_groups)
    group_size_report = {
        size: group_sizes.get(size, 0) for size in (1, 2, 3)
    }
    group_size_report[">3"] = sum(count for size, count in group_sizes.items() if size > 3)
    generic_groups = sum(bool(group["generic_basename"]) for group in source_groups)
    normalized_to_classes: dict[str, set[str]] = defaultdict(set)
    for group in source_groups:
        normalized_to_classes[str(group["normalized_source_filename"])].add(
            str(group["class_name"])
        )
    cross_class_filename_keys = sum(len(classes) > 1 for classes in normalized_to_classes.values())
    similarity = [
        float(group["median_fraction_pixels_rgb_delta_gt_16"])
        for group in source_groups
        if group["median_fraction_pixels_rgb_delta_gt_16"] is not None
    ]
    mean_rgb_differences = [
        float(group["median_mean_absolute_rgb_difference"])
        for group in source_groups
        if group["median_mean_absolute_rgb_difference"] is not None
    ]
    warnings = [
        "Banarasi, Bandhani, Ikat, and Pichwai are textile/pattern categories, not individual design IDs.",
        "Filename-derived source IDs are candidate grouping keys only; filenames do not prove real-world source identity.",
        "Generic basenames are common. Class scoping prevents cross-category merges, but same-class filename collisions cannot be ruled out from filenames alone.",
        "This dataset cannot directly provide real same-design/different-color pairs. A later controlled transformation study must be described as synthetic/controlled color-invariance evaluation, not real cross-color evaluation.",
    ]
    if source_overlaps and any(source_overlaps.values()):
        warnings.append("Candidate source groups span multiple splits; investigate as possible leakage.")
    if exact_duplicates and any(exact_overlaps.values()):
        warnings.append("Exact duplicate image content occurs across splits.")
    if group_size_report[1] or group_size_report[2] or group_size_report[">3"]:
        warnings.append("Observed filename-group sizes do not all match the README's stated three versions.")

    lines = [
        "# Public Kaggle Dataset Audit",
        "",
        f"- Fixed random seed: {seed}",
        f"- Total images: {len(rows)}",
        f"- Images by split: {_format_counter(by_split)}",
        f"- Images by class: {_format_counter(by_class)}",
        f"- Dimensions: {_format_counter(dimensions)}",
        f"- Color modes: {_format_counter(modes)}",
        "",
        "## Split/Class Counts",
        "",
        "| Split | " + " | ".join(EXPECTED_CLASSES) + " | Total |",
        "|---|" + "---:|" * (len(EXPECTED_CLASSES) + 1),
    ]
    for split in SPLITS:
        counts = [by_split_class[(split, class_name)] for class_name in EXPECTED_CLASSES]
        lines.append(f"| {split} | " + " | ".join(map(str, counts)) + f" | {by_split[split]} |")
    lines.extend(
        [
            "",
            "## Source-Group Audit",
            "",
            f"Candidate source groups (class + normalized filename): **{len(source_groups)}**.",
            "",
            f"- Groups with 1 image: {group_size_report[1]}",
            f"- Groups with 2 images: {group_size_report[2]}",
            f"- Groups with 3 images: {group_size_report[3]}",
            f"- Groups with more than 3 images: {group_size_report['>3']}",
            f"- Generic `image`/`images` filename groups: {generic_groups}",
            f"- Normalized filename keys appearing under multiple classes (safely class-scoped): {cross_class_filename_keys}",
            f"- Median fraction of pixels with RGB delta > 16 vs. the group's first variant: {statistics.median(similarity):.6f}" if similarity else "- Pixel comparison: no multi-image source groups available",
            f"- Median mean absolute RGB difference (0-255 scale): {statistics.median(mean_rgb_differences):.3f}" if mean_rgb_differences else "",
            "",
            "Variant indices are deterministic ordering labels within candidate groups; they are not augmentation lineage or design IDs.",
            "",
            "## Exact Duplicates",
            "",
            f"Exact duplicate MD5 groups: **{len(exact_duplicates)}**.",
            "",
            "| Split | Duplicate MD5 groups within split |",
            "|---|---:|",
        ]
    )
    for split in SPLITS:
        duplicate_group_count = sum(
            sum(str(row["split"]) == split for row in group) > 1
            for group in exact_duplicates.values()
        )
        lines.append(f"| {split} | {duplicate_group_count} |")
    lines.extend(["", "Exact duplicate groups across split pairs (shared MD5 keys):", ""])
    for pair, hashes in exact_overlaps.items():
        lines.append(f"- {pair}: {len(hashes)}")
    if exact_duplicates:
        lines.extend(["", "| MD5 | Splits | Paths |", "|---|---|---|"])
        for md5, group in sorted(exact_duplicates.items()):
            splits = ", ".join(sorted({str(row["split"]) for row in group}))
            paths = "<br>".join(str(row["relative_path"]) for row in group)
            lines.append(f"| `{md5}` | {splits} | {paths} |")
    lines.extend(["", "## Candidate Source Overlap Across Splits", ""])
    for pair, overlaps in source_overlaps.items():
        lines.append(f"- {pair}: {len(overlaps)} candidate source IDs")
    lines.extend(["", "## Limitations and Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "- Roboflow README states that three salt-and-pepper-noise versions were created per source image; measured group sizes above are filename-derived and must not be treated as verified identities.",
            "- The contact sheets are for human visual inspection; pixel similarity is supporting evidence, not proof of same-design identity.",
            "",
            "## Outputs",
            "",
            "- `metadata.csv`: one row per image, including dimensions, mode, MD5, and candidate source metadata.",
            "- `source_groups.csv`: candidate group membership and decoded-pixel comparison summaries.",
            "- `class_contact_sheet.png`: deterministic class examples.",
            "- `variant_contact_sheet.png`: candidate variants shown side by side.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(dataset_root: Path, output_dir: Path, seed: int = 42) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, payloads = create_metadata(dataset_root)
    if not rows:
        raise ValueError(f"No supported images found under dataset root: {dataset_root}")
    source_groups = analyze_source_groups(rows, payloads)
    exact_duplicates = find_exact_duplicate_groups(rows)
    source_overlaps = find_split_overlaps(rows, "source_id")
    exact_overlaps = find_split_overlaps(rows, "md5")
    _write_csv(output_dir / "metadata.csv", METADATA_FIELDS, rows)
    _write_csv(
        output_dir / "source_groups.csv",
        tuple(source_groups[0].keys()) if source_groups else (),
        source_groups,
    )
    write_contact_sheets(rows, payloads, output_dir, seed, 5, 6)
    write_summary_report(
        output_dir / "summary_report.md",
        rows,
        source_groups,
        exact_duplicates,
        source_overlaps,
        exact_overlaps,
        seed,
    )
    return {
        "total_images": len(rows),
        "split_counts": dict(Counter(str(row["split"]) for row in rows)),
        "class_counts": dict(Counter(str(row["class_name"]) for row in rows)),
        "source_group_counts": dict(Counter(int(group["image_count"]) for group in source_groups)),
        "exact_duplicate_groups": exact_duplicates,
        "source_overlaps": source_overlaps,
        "exact_overlaps": exact_overlaps,
        "output_dir": output_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Extracted dataset root or archive.zip")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for audit CSV, report, and contact sheets")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed (default: 42)")
    args = parser.parse_args()
    try:
        results = run_audit(args.dataset_root, args.output_dir, args.seed)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(f"Audited {results['total_images']} images; outputs: {results['output_dir']}")
    print(f"By split: {results['split_counts']}")
    print(f"By class: {results['class_counts']}")
    print(f"Candidate source group sizes: {results['source_group_counts']}")
    print(f"Exact duplicate MD5 groups: {len(results['exact_duplicate_groups'])}")
    print(f"Candidate source overlaps: {results['source_overlaps']}")
    print(f"Exact duplicate overlaps: {results['exact_overlaps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())