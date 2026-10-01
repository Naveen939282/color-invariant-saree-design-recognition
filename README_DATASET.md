# AIE-CASE Dataset Preparation

This repository contains dataset preparation code and documentation only. Keep proprietary DeepLure images, public dataset image copies, generated contact sheets, and data-specific CSV outputs on your local machine; do not commit or redistribute them.

## Data Sources

The inspected workspace currently contains `drive-download-20261001T112443Z-1-001.zip`. Its archive listing has 165 JPG files directly at the archive root and no category directories. The filenames do not establish design identity, category, or palette. Extract the archive locally before running the directory auditor. The empty `Normal_sarees` folder and any local Kaggle dataset can be included if present, but Kaggle images are not automatically called "normal sarees".

`source_dataset` is inferred only from a path component exactly named `deeplure` or `kaggle`, or can be set explicitly when auditing a single-source root. Otherwise it is `unknown`. `source_category` is an observed folder name; root-level images have a blank category. No filename, color, or visual similarity creates a design label.

## Install and Audit

From the workspace, install the small required dependency set:

```powershell
python -m pip install -r requirements.txt
```

Extract images into local folders and run the audit. For the inspected DeepLure archive, explicitly identify its source because extraction produces a flat directory:

```powershell
python audit_dataset.py --data-root "D:\path\to\extracted-images" --source-dataset deeplure --output-dir dataset_audit_output
```

For a mixed root with folders named `deeplure` and/or `kaggle`, use `--source-dataset auto` (the default). The auditor reads images without modifying them and writes:

- `image_metadata.csv`: one row per supported JPG/JPEG/PNG/WEBP file, including stable path-derived `image_id`, SHA256, dimensions, channels, source/category, readability, and notes.
- `dataset_summary.txt`: counts, statistics, unreadable images, duplicate groups, empty folders, extensions, and non-image files.
- `design_labels_template.csv` and `design_review.csv`: blank human annotation sheets. Existing annotations are preserved on a rerun when the same image ID and path remain.
- `design_review.html` and `design_review_pages/`: local contact-sheet pages, 25 images per page by default (or `--page-size 50`).

The audit reports SHA256 duplicates but does not remove or alter files. Resolve exact duplicate content before making splits because duplicate copies can leak across evaluation examples.

## Manual Design Labels

Copy `design_labels_template.csv` to `design_labels.csv`, then review the local contact sheet and fill in annotations. `design_id` must identify the underlying saree design/motif, not its color; use a consistent format such as `D001`. Do not assign the same design ID based only on filenames, palette, or model similarity. `colorway` is a human-entered palette name; use consistent spelling. Record the evidence and confidence in `notes`, `label_confidence`, and `label_source`. Leave uncertain matches unassigned until reviewed; the split builder intentionally rejects blank or malformed design IDs.

## Design-Aware Splits and Verification

After completing every row in `design_labels.csv`, generate the split files:

```powershell
python prepare_splits.py --data-root "D:\path\to\extracted-images" --metadata dataset_audit_output\image_metadata.csv --labels design_labels.csv --output-dir dataset_splits --seed 42
```

`train.csv` and `validation.csv` contain whole designs only; evaluation designs are excluded from both. `gallery.csv` and `query.csv` contain separate images from evaluation-only designs, which is necessary to measure retrieval for a known design without training on it. The query image is chosen to favor a different labeled colorway in the gallery when available. No image occurs in both gallery and query.

`verification_pairs.csv` is formed across query/gallery images. Positive pairs share a human-confirmed design ID, with cross-colorway positives prioritized. Negative pairs have different design IDs, with same-colorway negatives prioritized when those color labels exist. A shared colorway is only a palette-level negative condition, not proof of visual similarity. The pair file includes `image_id_1`, `image_id_2`, `label` (1 same design, 0 different design), `pair_type`, and `color_relationship` (`same`, `different`, or `unknown`).

Before writing outputs, the split builder fails on missing files/labels, duplicate image IDs or paths, duplicate file hashes, blank/invalid design IDs, design leakage, gallery/query overlap, and insufficient positive or negative pairs. Single-image designs are reported and retained in design-disjoint train/validation splits, but cannot supply positive retrieval pairs. Blank colorways and spelling variants that differ only by case/whitespace are reported for human cleanup. Set `--minimum-pairs-per-class` to raise the required count. Split assignment and pair sampling use a fixed seed.

## Optional Similarity Suggestions

Install the larger optional dependencies only if you want embedding suggestions:

```powershell
python -m pip install torch torchvision
python suggest_similar_pairs.py --data-root "D:\path\to\extracted-images" --metadata dataset_audit_output\image_metadata.csv --output candidate_pairs.csv
```

The first run may download pretrained ResNet-18 weights. `candidate_pairs.csv` contains `image_id_1`, `image_id_2`, `similarity_score`, and `review_status=NEEDS_HUMAN_REVIEW`. Similarity is a machine-generated suggestion only and is never converted into `design_id` values.

## Reproducibility and Git Hygiene

Use the same source tree, manually reviewed labels, tool versions, split fractions, and `--seed` to reproduce the outputs. The repository `.gitignore` excludes image/archive files and generated data artifacts. Commit code and documentation only; keep all dataset content and generated reports local.