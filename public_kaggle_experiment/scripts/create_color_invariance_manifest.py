"""Create deterministic transform, gallery, and verification-pair specifications."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from public_kaggle_experiment.src.color_transforms import (  # noqa: E402
    TRANSFORMATION_CONFIG,
    TRANSFORMATION_NAMES,
)


MANIFEST_FIELDS = (
    "split",
    "identity_id",
    "source_id",
    "identity_scope",
    "class_name",
    "canonical_path",
    "role",
    "transformation",
    "transformation_config_json",
    "transformed_path",
    "random_seed",
    "original_group_size",
)
GALLERY_FIELDS = ("identity_id", "source_id", "class_name", "split", "canonical_path")
PAIR_FIELDS = (
    "query_identity_id",
    "query_source_id",
    "query_class_name",
    "query_canonical_path",
    "query_transformation",
    "query_transformation_config_json",
    "gallery_identity_id",
    "gallery_source_id",
    "gallery_class_name",
    "gallery_canonical_path",
    "same_identity",
    "pair_type",
)
IDENTITY_SCOPE = "candidate filename group; not verified design identity"


def read_canonical_rows(canonical_csv: Path) -> list[dict[str, str]]:
    with canonical_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {
        "identity_id",
        "source_id",
        "class_name",
        "split",
        "canonical_path",
        "original_group_size",
    }
    if not rows:
        raise ValueError(f"Canonical CSV contains no identities: {canonical_csv}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Canonical CSV is missing required columns: {sorted(missing)}")
    if any(row["identity_id"] != row["source_id"] for row in rows):
        raise ValueError("identity_id must preserve the candidate source_id; do not infer design IDs")
    identities: dict[str, dict[str, str]] = {}
    for row in rows:
        identity_id = row["identity_id"]
        prior = identities.get(identity_id)
        if prior is not None:
            if prior != row:
                raise ValueError(f"Identity appears more than once or crosses splits: {identity_id}")
            raise ValueError(f"Canonical CSV contains more than one row for identity: {identity_id}")
        if row["split"] not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split for {identity_id}: {row['split']}")
        identities[identity_id] = row
    return sorted(rows, key=lambda row: row["identity_id"])


def build_transform_manifest(canonical_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for identity in sorted(canonical_rows, key=lambda row: row["identity_id"]):
        for transformation in TRANSFORMATION_NAMES:
            manifest.append(
                {
                    "split": identity["split"],
                    "identity_id": identity["identity_id"],
                    "source_id": identity["source_id"],
                    "identity_scope": IDENTITY_SCOPE,
                    "class_name": identity["class_name"],
                    "canonical_path": identity["canonical_path"],
                    "role": "query_view_specification",
                    "transformation": transformation,
                    "transformation_config_json": json.dumps(
                        TRANSFORMATION_CONFIG[transformation],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "transformed_path": "",
                    "random_seed": "",
                    "original_group_size": identity["original_group_size"],
                }
            )
    return manifest


def build_test_gallery(canonical_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "identity_id": row["identity_id"],
            "source_id": row["source_id"],
            "class_name": row["class_name"],
            "split": row["split"],
            "canonical_path": row["canonical_path"],
        }
        for row in sorted(canonical_rows, key=lambda item: item["identity_id"])
        if row["split"] == "test"
    ]


def build_test_verification_pairs(
    test_manifest: Iterable[dict[str, str]], gallery: Iterable[dict[str, str]]
) -> list[dict[str, str]]:
    gallery_rows = list(gallery)
    pairs: list[dict[str, str]] = []
    for query in test_manifest:
        for candidate in gallery_rows:
            same_identity = query["identity_id"] == candidate["identity_id"]
            pairs.append(
                {
                    "query_identity_id": query["identity_id"],
                    "query_source_id": query["source_id"],
                    "query_class_name": query["class_name"],
                    "query_canonical_path": query["canonical_path"],
                    "query_transformation": query["transformation"],
                    "query_transformation_config_json": query["transformation_config_json"],
                    "gallery_identity_id": candidate["identity_id"],
                    "gallery_source_id": candidate["source_id"],
                    "gallery_class_name": candidate["class_name"],
                    "gallery_canonical_path": candidate["canonical_path"],
                    "same_identity": "1" if same_identity else "0",
                    "pair_type": "positive" if same_identity else "negative",
                }
            )
    return pairs


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_manifests(canonical_csv: Path, output_dir: Path) -> dict[str, object]:
    canonical_rows = read_canonical_rows(canonical_csv)
    manifest = build_transform_manifest(canonical_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "color_invariance_manifest.csv", MANIFEST_FIELDS, manifest)

    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        split_rows = [row for row in manifest if row["split"] == split]
        _write_csv(output_dir / f"{split}_manifest.csv", MANIFEST_FIELDS, split_rows)
        split_counts[split] = sum(row["split"] == split for row in canonical_rows)

    gallery = build_test_gallery(canonical_rows)
    _write_csv(output_dir / "test_gallery.csv", GALLERY_FIELDS, gallery)
    test_manifest = [row for row in manifest if row["split"] == "test"]
    pairs = build_test_verification_pairs(test_manifest, gallery)
    _write_csv(output_dir / "test_verification_pairs.csv", PAIR_FIELDS, pairs)

    summary = {
        "identity_scope": IDENTITY_SCOPE,
        "canonical_identities_by_split": split_counts,
        "transformations": list(TRANSFORMATION_NAMES),
        "transformation_count": len(TRANSFORMATION_NAMES),
        "manifest_rows_by_split": {
            split: sum(row["split"] == split for row in manifest)
            for split in ("train", "valid", "test")
        },
        "total_manifest_rows": len(manifest),
        "test_gallery_rows": len(gallery),
        "test_query_rows": len(test_manifest),
        "test_verification_pair_rows": len(pairs),
        "test_positive_pairs": sum(row["same_identity"] == "1" for row in pairs),
        "test_negative_pairs": sum(row["same_identity"] == "0" for row in pairs),
        "transformed_images_written": 0,
        "random_seed": None,
        "transform_config": TRANSFORMATION_CONFIG,
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = create_manifests(args.canonical_csv, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Manifest identities by split: {summary['canonical_identities_by_split']}")
    print(f"Transformations: {summary['transformation_count']}")
    print(f"Manifest rows: {summary['total_manifest_rows']}")
    print(
        "Test gallery/query/pairs: "
        f"{summary['test_gallery_rows']}/{summary['test_query_rows']}/"
        f"{summary['test_verification_pair_rows']} "
        f"({summary['test_positive_pairs']} positive, {summary['test_negative_pairs']} negative)"
    )
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())