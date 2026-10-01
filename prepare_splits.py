"""Validate manual design labels and create design-disjoint retrieval splits."""

from __future__ import annotations

import argparse
import itertools
import random
import re
from pathlib import Path

import pandas as pd


LABEL_COLUMNS = [
    "image_id",
    "relative_path",
    "design_id",
    "colorway",
    "label_confidence",
    "label_source",
    "notes",
]
PAIR_COLUMNS = ["image_id_1", "image_id_2", "label", "pair_type", "color_relationship"]


def fail(message: str) -> None:
    raise ValueError(message)


def validate_inputs(data_root: Path, metadata_path: Path, labels_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    missing_metadata_columns = {"image_id", "relative_path", "sha256", "readable"} - set(metadata.columns)
    missing_label_columns = set(LABEL_COLUMNS) - set(labels.columns)
    if missing_metadata_columns:
        fail(f"Metadata is missing required columns: {sorted(missing_metadata_columns)}")
    if missing_label_columns:
        fail(f"Labels are missing required columns: {sorted(missing_label_columns)}")
    if metadata["image_id"].duplicated().any():
        fail("Duplicate image_id values in image_metadata.csv.")
    if labels["image_id"].duplicated().any():
        fail("Duplicate image_id values in design_labels.csv.")
    if metadata["relative_path"].duplicated().any() or labels["relative_path"].duplicated().any():
        fail("Duplicate relative_path values found.")
    extra_label_ids = set(labels["image_id"]) - set(metadata["image_id"])
    if extra_label_ids:
        fail(f"Labels reference unknown image IDs: {sorted(extra_label_ids)[:10]}")

    joined = metadata.merge(labels[LABEL_COLUMNS], on="image_id", how="left", suffixes=("_metadata", ""), indicator=True)
    if (joined["_merge"] != "both").any():
        missing = joined.loc[joined["_merge"] != "both", "image_id"].tolist()
        fail(f"Missing design labels for image IDs: {missing[:10]}")
    path_mismatches = joined["relative_path_metadata"] != joined["relative_path"]
    if path_mismatches.any():
        fail("Label paths do not match metadata for image IDs: " + ", ".join(joined.loc[path_mismatches, "image_id"].head(10)))
    if (joined["design_id"].str.strip() == "").any():
        missing = joined.loc[joined["design_id"].str.strip() == "", "image_id"].tolist()
        fail(f"Blank design_id values must be manually reviewed: {missing[:10]}")
    invalid = ~joined["design_id"].str.fullmatch(r"D\d+", na=False)
    if invalid.any():
        fail("Invalid design_id values (expected D followed by digits): " + ", ".join(joined.loc[invalid, "design_id"].unique()[:10]))

    missing_files = [
        relative_path
        for relative_path in joined["relative_path"]
        if not (data_root / Path(relative_path)).is_file()
    ]
    if missing_files:
        fail("Missing image files: " + ", ".join(missing_files[:10]))
    unreadable = joined[~joined["readable"].str.casefold().eq("true")]
    if not unreadable.empty:
        fail("Unreadable images cannot be split: " + ", ".join(unreadable["image_id"].head(10)))
    duplicates = metadata[metadata["sha256"].ne("") & metadata["sha256"].duplicated(keep=False)]
    if not duplicates.empty:
        groups = duplicates.groupby("sha256")["relative_path"].apply(list).tolist()
        fail(f"Duplicate image contents must be resolved before splitting: {groups[:5]}")

    joined = joined.drop(columns="_merge")
    joined["relative_path"] = joined["relative_path"].astype(str)
    return joined


def color_relation(first: str, second: str) -> str:
    first_normalized = first.strip().casefold()
    second_normalized = second.strip().casefold()
    if not first_normalized or not second_normalized:
        return "unknown"
    return "same" if first_normalized == second_normalized else "different"


def make_pairs(gallery: pd.DataFrame, query: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    candidates: dict[int, list[tuple]] = {1: [], 0: []}
    for query_row, gallery_row in itertools.product(
        query.to_dict("records"), gallery.to_dict("records")
    ):
        if query_row["image_id"] == gallery_row["image_id"]:
            fail("The same image appeared in both gallery and query.")
        same_design = query_row["design_id"] == gallery_row["design_id"]
        label = int(same_design)
        relation = color_relation(query_row.get("colorway", ""), gallery_row.get("colorway", ""))
        if same_design:
            pair_type = f"positive_{'cross' if relation == 'different' else relation}_colorway"
            priority = 0 if relation == "different" else 1
        else:
            pair_type = f"negative_{relation}_colorway"
            priority = 0 if relation == "same" else 1
        candidates[label].append(
            (priority, query_row["image_id"], gallery_row["image_id"], label, pair_type, relation)
        )

    rng = random.Random(seed)
    selected = []
    for label in (1, 0):
        label_candidates = candidates[label]
        rng.shuffle(label_candidates)
        label_candidates.sort(key=lambda pair: pair[0])
        selected.extend(label_candidates[:limit])
    records = [
        {
            "image_id_1": image_id_1,
            "image_id_2": image_id_2,
            "label": label,
            "pair_type": pair_type,
            "color_relationship": relation,
        }
        for _, image_id_1, image_id_2, label, pair_type, relation in selected
    ]
    return pd.DataFrame(records, columns=PAIR_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Directory scanned by audit_dataset.py")
    parser.add_argument("--metadata", type=Path, default=Path("dataset_audit_output/image_metadata.csv"))
    parser.add_argument("--labels", type=Path, required=True, help="Manually completed design_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-class", type=int, default=10000)
    parser.add_argument("--minimum-pairs-per-class", type=int, default=1)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1 or not 0 < args.evaluation_fraction < 1:
        parser.error("validation and evaluation fractions must be between 0 and 1")
    if args.max_pairs_per_class < 1 or args.minimum_pairs_per_class < 1:
        parser.error("pair limits must be positive")

    data_root = args.data_root.resolve()
    dataset = validate_inputs(data_root, args.metadata.resolve(), args.labels.resolve())
    counts = dataset.groupby("design_id").size().sort_index()
    singleton_designs = counts[counts == 1].index.tolist()
    eligible = counts[counts >= 2].index.tolist()
    if singleton_designs:
        print(f"Warning: {len(singleton_designs)} single-image designs cannot produce positive evaluation pairs; retained in design-level splits.")
    if len(counts) < 3 or not eligible:
        fail("At least three distinct designs and one design with two or more images are required for train/validation/retrieval evaluation.")

    seed_rng = random.Random(args.seed)
    eligible_shuffled = eligible.copy()
    seed_rng.shuffle(eligible_shuffled)
    evaluation_count = max(1, round(len(eligible) * args.evaluation_fraction))
    evaluation_count = min(evaluation_count, len(counts) - 2, len(eligible))
    if evaluation_count < 1:
        fail("Cannot reserve evaluation designs while leaving separate train and validation designs.")
    evaluation_designs = set(eligible_shuffled[:evaluation_count])
    remaining_designs = [design for design in counts.index if design not in evaluation_designs]
    seed_rng.shuffle(remaining_designs)
    validation_count = max(1, round(len(remaining_designs) * args.validation_fraction))
    validation_count = min(validation_count, len(remaining_designs) - 1)
    if validation_count < 1:
        fail("Cannot create non-empty, design-disjoint train and validation sets.")
    validation_designs = set(remaining_designs[:validation_count])
    train_designs = set(remaining_designs[validation_count:])
    if not train_designs:
        fail("Training split would be empty.")

    train = dataset[dataset["design_id"].isin(train_designs)].copy()
    validation = dataset[dataset["design_id"].isin(validation_designs)].copy()
    evaluation = dataset[dataset["design_id"].isin(evaluation_designs)].copy()
    gallery_rows = []
    query_rows = []
    for design_id, group in evaluation.groupby("design_id", sort=True):
        records = group.to_dict("records")
        seed_rng.shuffle(records)
        query_record = max(
            records,
            key=lambda candidate: sum(
                color_relation(candidate.get("colorway", ""), other.get("colorway", "")) == "different"
                for other in records
                if other["image_id"] != candidate["image_id"]
            ),
        )
        query_rows.append(query_record)
        gallery_rows.extend(record for record in records if record["image_id"] != query_record["image_id"])
    gallery = pd.DataFrame(gallery_rows, columns=dataset.columns)
    query = pd.DataFrame(query_rows, columns=dataset.columns)

    if set(train["design_id"]) & (set(validation["design_id"]) | set(evaluation["design_id"])):
        fail("Train/evaluation design leakage detected.")
    if set(validation["design_id"]) & set(evaluation["design_id"]):
        fail("Validation/evaluation design leakage detected.")
    if set(gallery["image_id"]) & set(query["image_id"]):
        fail("Gallery/query image overlap detected.")
    pairs = make_pairs(gallery, query, args.max_pairs_per_class, args.seed)
    for label in (0, 1):
        count = int((pairs["label"] == label).sum())
        if count < args.minimum_pairs_per_class:
            fail(f"Insufficient {'positive' if label else 'negative'} verification pairs: {count}; minimum is {args.minimum_pairs_per_class}.")
    if ((pairs["label"] == 1) & (pairs["image_id_1"] == pairs["image_id_2"])).any():
        fail("Positive pair contains identical image IDs.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    gallery.to_csv(output_dir / "gallery.csv", index=False)
    query.to_csv(output_dir / "query.csv", index=False)
    pairs.to_csv(output_dir / "verification_pairs.csv", index=False)

    normalized_colorways = dataset.assign(_normalized=dataset["colorway"].str.strip().str.casefold())
    variants = normalized_colorways.groupby(["design_id", "_normalized"])["colorway"].nunique()
    inconsistent = variants[variants > 1]
    blank_colorways = int(dataset["colorway"].str.strip().eq("").sum())
    if not inconsistent.empty:
        print(f"Warning: {len(inconsistent)} design/colorway spelling groups differ only by case or whitespace; normalize manually.")
    if blank_colorways:
        print(f"Warning: {blank_colorways} images have blank colorway values; these pairs are marked unknown.")
    print(
        "Split complete: "
        f"train={len(train)} images/{len(train_designs)} designs, "
        f"validation={len(validation)} images/{len(validation_designs)} designs, "
        f"gallery={len(gallery)}, query={len(query)}, "
        f"positive_pairs={(pairs['label'] == 1).sum()}, negative_pairs={(pairs['label'] == 0).sum()}."
    )
    print(f"Files written to: {output_dir}")


if __name__ == "__main__":
    main()