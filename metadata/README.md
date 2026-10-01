# Metadata and Labeling Workflow

## Sources and Local Paths

Pass local roots at runtime; never put a personal machine path in source code. Use `--deeplure-root` for the proprietary DeepLure corpus and `--kaggle-root` for the public Indian Saree Patterns dataset. The two roots are scanned separately and joined only in metadata, with `source_dataset` retained on every row. Kaggle is not automatically mapped to `normal_sarees`. Actual child folder names are retained in `source_category`; flat root-level images have a blank category.

DeepLure images and archives must stay outside the Git repository and must not be uploaded, redistributed, or linked publicly. Local contact sheets and candidate-pair outputs are ignored by Git. Metadata CSVs contain paths and measurements, not image bytes; review them before publication for any sensitive filenames or local path information.

## Metadata Generation

Run `python scripts/generate_metadata.py --deeplure-root <local-root> --kaggle-root <local-root>`. It writes:

- `image_metadata.csv`: stable ID, source-relative path, filename, source/category, extension, byte size, dimensions, channels, aspect ratio, SHA256, readability, and notes.
- `design_labels_template.csv`: blank manual-label template.
- `design_labels.csv`: working label file. Existing human entries are preserved on reruns for matching ID/path rows; new images receive blank labels.
- `design_review.csv`: review copy with the same columns.
- `../reports/duplicate_report.csv`: exact SHA256 duplicate groups, with paths grouped as JSON strings.
- `../reports/dataset_summary.md`: source counts, dimensions, aspect ratios, categories, duplicate groups, corrupt images, empty directories, and file types.

An exact duplicate means identical file bytes (SHA256), not the same design. Nothing is removed automatically. Stable IDs hash both source name and source-relative path, so identically named files in different datasets remain distinct.

## Human Design Labels

Columns are `image_id`, `relative_path`, `design_id`, `colorway`, `label_confidence`, `label_source`, and `notes`. Use a design ID only after reviewing the textile motif. The same design in two color palettes gets the same ID; different motifs get different IDs even when their colors are equal. Record uncertain cases for later review rather than guessing. The split script accepts IDs matching `D` followed by digits and rejects blank IDs.

Suggested review distinctions:

- Same design, different color: same design ID; distinct standardized colorway values.
- Same design, similar color: same design ID; document the visual judgment.
- Different design, same/similar color: different design IDs; a useful negative example.
- Uncertain: leave design ID blank until human verification.

Never use filenames, source folders, dominant colors, pixel averages, or model embeddings as design ground truth.

## Color-Invariance Splits

`python scripts/create_splits.py` creates `train.csv`, `validation.csv`, `gallery.csv`, `query.csv`, and `verification_pairs.csv` under `metadata/`. It assigns whole design IDs to training, validation, or evaluation. Evaluation query images are separate from gallery images, while designs remain represented in both query and gallery so retrieval can be evaluated. Query selection prioritizes a colorway different from gallery variants for that same design.

Positive pairs have one design ID and prioritize different colorways. Negative pairs have different design IDs and prioritize equal colorway text. Equal text is only a labeled-color proxy for a hard negative, not a claim of perceptual similarity. Machine similarity outputs are never used to create labels or pairs.

## Validation

Run `python scripts/validate_dataset.py --metadata metadata/image_metadata.csv --labels metadata/design_labels.csv --deeplure-root <local-root> --kaggle-root <local-root> --split-dir metadata`. Critical file, label, duplicate, leakage, or pair errors fail. Single-image designs, blank/inconsistent colorways, and insufficient cross-color positives are reported so a person can decide whether the data supports the intended evaluation. All random operations use the supplied fixed seed (default 42).