"""Select one deterministic canonical image per audited candidate source group."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CANONICAL_FIELDS = (
    "identity_id",
    "source_id",
    "class_name",
    "split",
    "canonical_path",
    "canonical_filename",
    "canonical_md5",
    "original_group_size",
    "selection_rule",
)
SELECTION_RULE = "lexicographically_smallest_relative_path"


def select_canonical_rows(metadata_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        groups[row["source_id"]].append(row)

    canonical_rows: list[dict[str, str]] = []
    for source_id, rows in sorted(groups.items()):
        splits = {row["split"] for row in rows}
        classes = {row["class_name"] for row in rows}
        declared_sizes = {int(row["source_variant_count"]) for row in rows}
        if len(splits) != 1:
            raise ValueError(f"Candidate source group crosses splits: {source_id}: {sorted(splits)}")
        if len(classes) != 1:
            raise ValueError(f"Candidate source group crosses classes: {source_id}: {sorted(classes)}")
        if len(declared_sizes) != 1 or declared_sizes != {len(rows)}:
            raise ValueError(f"Inconsistent variant count for candidate source group: {source_id}")
        if len(rows) not in {1, 3}:
            raise ValueError(
                f"Expected audited candidate groups of size 1 or 3, got {len(rows)}: {source_id}"
            )

        selected = min(rows, key=lambda row: row["relative_path"])
        canonical_rows.append(
            {
                "identity_id": source_id,
                "source_id": source_id,
                "class_name": selected["class_name"],
                "split": selected["split"],
                "canonical_path": selected["relative_path"],
                "canonical_filename": selected["filename"],
                "canonical_md5": selected["md5"],
                "original_group_size": str(len(rows)),
                "selection_rule": SELECTION_RULE,
            }
        )
    return canonical_rows


def create_canonical_csv(metadata_csv: Path, output_csv: Path) -> list[dict[str, str]]:
    with metadata_csv.open(newline="", encoding="utf-8") as source:
        rows = select_canonical_rows(csv.DictReader(source))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = create_canonical_csv(args.metadata_csv, args.output_csv)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["split"]] += 1
    print(f"Selected {len(rows)} canonical candidate identities: {dict(sorted(counts.items()))}")
    print(f"Canonical CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())