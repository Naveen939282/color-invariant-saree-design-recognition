"""Validation and design-aware splitting helpers."""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    from .dataset_utils import LABEL_COLUMNS, resolve_image_path, sha256_file
except ImportError:
    from dataset_utils import LABEL_COLUMNS, resolve_image_path, sha256_file


PAIR_COLUMNS = ["image_id_1", "image_id_2", "label", "pair_type", "color_relationship"]
METADATA_REQUIRED = {"image_id", "relative_path", "source_dataset", "sha256", "readable"}


def fail(message: str) -> None:
    raise ValueError(message)


def load_labeled_dataset(
    metadata_path: Path,
    labels_path: Path,
    data_root: Path | None,
    deeplure_root: Path | None,
    kaggle_root: Path | None,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    missing_metadata = METADATA_REQUIRED - set(metadata.columns)
    missing_labels = set(LABEL_COLUMNS) - set(labels.columns)
    if missing_metadata:
        fail(f"Metadata is missing columns: {sorted(missing_metadata)}")
    if missing_labels:
        fail(f"Labels are missing columns: {sorted(missing_labels)}")
    if metadata["image_id"].eq("").any() or metadata["relative_path"].eq("").any():
        fail("Metadata contains blank image IDs or paths.")
    if metadata["image_id"].duplicated().any() or labels["image_id"].duplicated().any():
        fail("Duplicate image_id values found.")
    if metadata.duplicated(["source_dataset", "relative_path"]).any():
        fail("Duplicate relative paths found within a source dataset.")
    extra_ids = set(labels["image_id"]) - set(metadata["image_id"])
    if extra_ids:
        fail(f"Labels reference unknown image IDs: {sorted(extra_ids)[:10]}")

    joined = metadata.merge(
        labels[LABEL_COLUMNS],
        on="image_id",
        how="left",
        suffixes=("_metadata", ""),
        indicator=True,
    )
    missing_ids = joined.loc[joined["_merge"] != "both", "image_id"].tolist()
    if missing_ids:
        fail(f"Missing label rows for image IDs: {missing_ids[:10]}")
    path_mismatch = joined["relative_path_metadata"] != joined["relative_path"]
    if path_mismatch.any():
        fail("Label paths do not match metadata for: " + ", ".join(joined.loc[path_mismatch, "image_id"].head(10)))
    blank_ids = joined["design_id"].str.strip().eq("")
    if blank_ids.any():
        fail("Blank design IDs require human review: " + ", ".join(joined.loc[blank_ids, "image_id"].head(10)))
    invalid_ids = ~joined["design_id"].str.fullmatch(r"D\d+", na=False)
    if invalid_ids.any():
        values = joined.loc[invalid_ids, "design_id"].unique().tolist()
        fail(f"Invalid design IDs (expected D followed by digits): {values[:10]}")
    unreadable = ~joined["readable"].str.casefold().eq("true")
    if unreadable.any():
        fail("Unreadable images cannot be split: " + ", ".join(joined.loc[unreadable, "image_id"].head(10)))

    for row in joined.to_dict("records"):
        image_path = resolve_image_path(row, data_root, deeplure_root, kaggle_root)
        if not image_path.is_file():
            fail(f"Missing image file for {row['image_id']}: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            fail(f"Image became unreadable for {row['image_id']}: {error}")
        recorded_hash = row.get("sha256", "")
        actual_hash = sha256_file(image_path)
        if recorded_hash and actual_hash != recorded_hash:
            fail(f"Image content changed since metadata generation: {row['image_id']}")

    duplicate_hashes = joined[joined["sha256"].ne("") & joined["sha256"].duplicated(keep=False)]
    if not duplicate_hashes.empty:
        duplicate_paths = duplicate_hashes.groupby("sha256")["relative_path"].apply(list).tolist()
        fail(f"Exact duplicate image contents must be resolved before splitting: {duplicate_paths[:5]}")
    return joined.drop(columns="_merge")


def color_relationship(first: str, second: str) -> str:
    first_value = first.strip().casefold()
    second_value = second.strip().casefold()
    if not first_value or not second_value:
        return "unknown"
    return "same" if first_value == second_value else "different"


def generate_pairs(
    gallery: pd.DataFrame,
    query: pd.DataFrame,
    max_pairs_per_class: int,
    seed: int,
) -> pd.DataFrame:
    candidates: dict[int, list[tuple]] = {0: [], 1: []}
    for query_row, gallery_row in itertools.product(
        query.to_dict("records"), gallery.to_dict("records")
    ):
        if query_row["image_id"] == gallery_row["image_id"]:
            fail("The same image cannot be paired with itself.")
        relation = color_relationship(
            query_row.get("colorway", ""), gallery_row.get("colorway", "")
        )
        same_design = query_row["design_id"] == gallery_row["design_id"]
        label = int(same_design)
        if same_design:
            pair_type = f"positive_{'cross' if relation == 'different' else relation}_colorway"
            priority = 0 if relation == "different" else 1
        else:
            pair_type = f"negative_{relation}_colorway"
            priority = 0 if relation == "same" else 1
        candidates[label].append(
            (
                priority,
                query_row["image_id"],
                gallery_row["image_id"],
                label,
                pair_type,
                relation,
            )
        )

    rng = random.Random(seed)
    rows = []
    for label in (1, 0):
        values = candidates[label]
        rng.shuffle(values)
        values.sort(key=lambda item: item[0])
        for _, first, second, pair_label, pair_type, relation in values[:max_pairs_per_class]:
            rows.append(
                {
                    "image_id_1": first,
                    "image_id_2": second,
                    "label": pair_label,
                    "pair_type": pair_type,
                    "color_relationship": relation,
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def create_splits(
    dataset: pd.DataFrame,
    seed: int = 42,
    validation_fraction: float = 0.15,
    evaluation_fraction: float = 0.2,
    max_pairs_per_class: int = 10000,
) -> dict[str, pd.DataFrame]:
    if not 0 < validation_fraction < 1 or not 0 < evaluation_fraction < 1:
        fail("Validation/evaluation fractions must be between 0 and 1.")
    if max_pairs_per_class < 1:
        fail("Maximum pairs per class must be positive.")
    design_counts = dataset.groupby("design_id").size().sort_index()
    singletons = design_counts[design_counts == 1].index.tolist()
    eligible = design_counts[design_counts >= 2].index.tolist()
    if singletons:
        print(
            f"WARNING: {len(singletons)} designs have one image and cannot contribute positive pairs."
        )
    if len(design_counts) < 3 or len(eligible) < 2:
        fail("At least three designs and two multi-image designs are needed for train/validation/retrieval evaluation.")

    rng = random.Random(seed)
    shuffled_eligible = eligible.copy()
    rng.shuffle(shuffled_eligible)
    evaluation_count = max(2, round(len(eligible) * evaluation_fraction))
    evaluation_count = min(evaluation_count, len(eligible), len(design_counts) - 2)
    if evaluation_count < 2:
        fail("Cannot reserve at least two evaluation designs while leaving train and validation designs.")
    evaluation_designs = set(shuffled_eligible[:evaluation_count])
    remaining = [design for design in design_counts.index if design not in evaluation_designs]
    rng.shuffle(remaining)
    validation_count = max(1, round(len(remaining) * validation_fraction))
    validation_count = min(validation_count, len(remaining) - 1)
    if validation_count < 1:
        fail("Cannot create separate non-empty train and validation design groups.")
    validation_designs = set(remaining[:validation_count])
    train_designs = set(remaining[validation_count:])

    train = dataset[dataset["design_id"].isin(train_designs)].copy()
    validation = dataset[dataset["design_id"].isin(validation_designs)].copy()
    evaluation = dataset[dataset["design_id"].isin(evaluation_designs)].copy()
    query_rows: list[dict] = []
    gallery_rows: list[dict] = []
    for _, group in evaluation.groupby("design_id", sort=True):
        records = group.to_dict("records")
        rng.shuffle(records)
        selected_query = max(
            records,
            key=lambda candidate: sum(
                color_relationship(candidate.get("colorway", ""), gallery.get("colorway", ""))
                == "different"
                for gallery in records
                if candidate["image_id"] != gallery["image_id"]
            ),
        )
        query_rows.append(selected_query)
        gallery_rows.extend(
            row for row in records if row["image_id"] != selected_query["image_id"]
        )
    query = pd.DataFrame(query_rows, columns=dataset.columns)
    gallery = pd.DataFrame(gallery_rows, columns=dataset.columns)
    pairs = generate_pairs(gallery, query, max_pairs_per_class, seed)
    return {
        "train": train,
        "validation": validation,
        "gallery": gallery,
        "query": query,
        "verification_pairs": pairs,
    }


def validate_split_frames(
    splits: dict[str, pd.DataFrame], min_positive_pairs: int = 1, min_negative_pairs: int = 1
) -> list[str]:
    train = splits["train"]
    validation = splits["validation"]
    gallery = splits["gallery"]
    query = splits["query"]
    pairs = splits["verification_pairs"]
    warnings: list[str] = []

    for name, frame in splits.items():
        if "image_id" in frame and frame["image_id"].duplicated().any():
            fail(f"Duplicate image IDs in {name} split.")
    train_designs = set(train["design_id"])
    validation_designs = set(validation["design_id"])
    gallery_designs = set(gallery["design_id"])
    query_designs = set(query["design_id"])
    if train_designs & (validation_designs | gallery_designs | query_designs):
        fail("Train/evaluation design leakage detected.")
    if validation_designs & (gallery_designs | query_designs):
        fail("Validation/evaluation design leakage detected.")
    if set(gallery["image_id"]) & set(query["image_id"]):
        fail("The same image appears in both gallery and query.")
    if set(train["image_id"]) & set(query["image_id"]):
        fail("The same image appears in both train and query.")
    if set(train["image_id"]) & set(gallery["image_id"]):
        fail("The same image appears in both train and gallery.")

    all_ids = set().union(*(set(frame["image_id"]) for frame in splits.values() if "image_id" in frame))
    required_pair_columns = set(PAIR_COLUMNS)
    if required_pair_columns - set(pairs.columns):
        fail(f"Verification pairs are missing columns: {sorted(required_pair_columns - set(pairs.columns))}")
    for pair in pairs.to_dict("records"):
        first, second = pair["image_id_1"], pair["image_id_2"]
        if first == second:
            fail("Verification pair contains identical image IDs.")
        if first not in all_ids or second not in all_ids:
            fail(f"Verification pair references unknown IDs: {first}, {second}")
        if first not in set(query["image_id"]) or second not in set(gallery["image_id"]):
            fail("Verification pairs must use query as image_id_1 and gallery as image_id_2.")
        if pair["label"] not in (0, 1, "0", "1"):
            fail(f"Invalid verification label: {pair['label']}")
        actual_label = int(
            query.loc[query["image_id"] == first, "design_id"].iloc[0]
            == gallery.loc[gallery["image_id"] == second, "design_id"].iloc[0]
        )
        if int(pair["label"]) != actual_label:
            fail(f"Verification label does not match design IDs for pair {first}, {second}.")
        query_row = query.loc[query["image_id"] == first].iloc[0]
        gallery_row = gallery.loc[gallery["image_id"] == second].iloc[0]
        expected_color = color_relationship(
            query_row.get("colorway", ""), gallery_row.get("colorway", "")
        )
        if pair["color_relationship"] != expected_color:
            fail(f"Color relationship is incorrect for pair {first}, {second}.")
        relation_name = (
            "cross"
            if actual_label and expected_color == "different"
            else expected_color
        )
        expected_type = f"{'positive' if actual_label else 'negative'}_{relation_name}_colorway"
        if pair["pair_type"] != expected_type:
            fail(f"Pair type is incorrect for pair {first}, {second}.")
    positive_count = int((pairs["label"].astype(str) == "1").sum())
    negative_count = int((pairs["label"].astype(str) == "0").sum())
    if positive_count < min_positive_pairs:
        fail(f"Insufficient positive verification pairs: {positive_count} (minimum {min_positive_pairs}).")
    if negative_count < min_negative_pairs:
        fail(f"Insufficient negative verification pairs: {negative_count} (minimum {min_negative_pairs}).")

    cross_color = pairs[
        (pairs["label"].astype(str) == "1")
        & (pairs["color_relationship"] == "different")
    ]
    if cross_color.empty:
        warnings.append("No positive verification pair has different manually labeled colorways.")
    if (pairs["color_relationship"] == "unknown").any():
        warnings.append("Some verification pairs have unknown color relationship because a colorway is blank.")
    return warnings


def validate_metadata_only(metadata_path: Path, roots: dict[str, Path | None]) -> list[str]:
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing = METADATA_REQUIRED - set(metadata.columns)
    if missing:
        fail(f"Metadata is missing columns: {sorted(missing)}")
    if metadata["image_id"].duplicated().any():
        fail("Duplicate image IDs in metadata.")
    if metadata.duplicated(["source_dataset", "relative_path"]).any():
        fail("Duplicate paths within a source in metadata.")
    unreadable = metadata[~metadata["readable"].str.casefold().eq("true")]
    if not unreadable.empty:
        fail("Unreadable images are recorded in metadata: " + ", ".join(unreadable["image_id"].head(10)))
    hashes: dict[str, list[str]] = {}
    for row in metadata.to_dict("records"):
        root = roots.get(row["source_dataset"]) or roots.get("unknown")
        if root is None:
            fail(f"No root configured for source {row['source_dataset']!r}.")
        path = (root / row["relative_path"]).resolve()
        if root.resolve() not in path.parents and path != root.resolve():
            fail(f"Metadata path escapes configured root: {row['relative_path']}")
        if not path.is_file():
            fail(f"Missing image file: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            fail(f"Unreadable image {row['image_id']}: {error}")
        actual_hash = sha256_file(path)
        if row["sha256"] and row["sha256"] != actual_hash:
            fail(f"Image content changed since metadata generation: {row['image_id']}")
        hashes.setdefault(actual_hash, []).append(row["image_id"])
    duplicate_groups = [image_ids for image_ids in hashes.values() if len(image_ids) > 1]
    if duplicate_groups:
        fail(f"Exact duplicate image contents found: {duplicate_groups[:5]}")
    return []