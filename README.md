# AIE-CASE: Color-Invariant Saree Design Recognition

Dataset inspection and preparation tools for a PyTorch image retrieval and verification task. This repository prepares metadata, human-confirmed design labels, leakage-resistant splits, and verification pairs; it does not train the retrieval model.

## Proprietary Data

The DeepLure Drive corpus is proprietary third-party data. Keep the original archive and all extracted DeepLure images outside this Git repository. Never commit or redistribute them, and do not create public image URLs. `.gitignore` blocks common image/archive/model files, and `scripts/check_git_safety.py` checks tracked and local workspace files. Metadata and documentation may be reviewed and committed; image files, archives, contact sheets, and model weights must not be.

The original workspace ZIP was already present when this project work began and remains ignored, not tracked. Prefer moving it and all extracted files outside the repository. The previously extracted working copy was moved to `%USERPROFILE%\DeepLureData\deeplure` on this machine.

## Install

The audit and split workflows need only pandas and Pillow:

```shell
python -m pip install -r requirements.txt
```

The optional ResNet embedding helper has separate dependencies and may download pretrained weights on its first run:

```shell
python -m pip install -r requirements-similarity.txt
```

## Inspect One or Both Sources

Every input root is read-only. Metadata outputs are kept outside the dataset roots. Source inference looks for `deeplure` or `kaggle` in the path, unless a source is explicitly specified. Otherwise the source is `unknown`. A source folder/category is copied only from an actual child directory; no category or design is guessed from image appearance or filenames.

Windows PowerShell, DeepLure only:

```powershell
python scripts/inspect_dataset.py --deeplure-root "$env:USERPROFILE\DeepLureData\deeplure"
```

Windows PowerShell, two separate roots:

```powershell
python scripts/inspect_dataset.py --deeplure-root "D:\Datasets\deeplure" --kaggle-root "D:\Datasets\indian-saree-patterns"
```

Linux or Kaggle:

```shell
python scripts/inspect_dataset.py --deeplure-root /data/local/deeplure --kaggle-root /kaggle/input/indian-saree-patterns
```

The same `--deeplure-root` and `--kaggle-root` flags can be passed to `scripts/generate_metadata.py`. Kaggle remains a separate `source_dataset=kaggle`; it is not automatically called “normal sarees.”

The inspection creates `metadata/image_metadata.csv`, `metadata/design_labels_template.csv`, `metadata/design_labels.csv`, `metadata/design_review.csv`, `reports/duplicate_report.csv`, and `reports/dataset_summary.md`. It supports JPG/JPEG/PNG/WEBP/BMP, reports non-image files and empty directories, records SHA256 exact duplicates, and never changes source images. Rerunning metadata generation preserves existing manual labels when image ID and relative path still match.

## Contact Sheets and Manual Labels

Create local paginated review pages (25 images per page by default):

```shell
python scripts/generate_contact_sheet.py --metadata metadata/image_metadata.csv --deeplure-root /data/local/deeplure --kaggle-root /data/local/kaggle
```

Open `local_outputs/contact_sheets/index.html` in a browser. The generated pages reference the local images; do not publish or commit them.

Review the contact sheets and fill `metadata/design_labels.csv`. `design_id` identifies the underlying textile design/motif, not a color palette. Use consistent IDs such as `D001`; record confidence, source, and evidence. Distinguish same design/different color, same design/similar color, different design/same or similar color, and uncertain. Leave uncertain matches blank until a person verifies them. A blank design ID deliberately blocks split generation. Do not infer identity from filename, folder, dominant color, pixel similarity, or an embedding score.

## Splits and Verification

After manually completing labels:

```shell
python scripts/create_splits.py --metadata metadata/image_metadata.csv --labels metadata/design_labels.csv --deeplure-root /data/local/deeplure --kaggle-root /data/local/kaggle --output-dir metadata --seed 42
```

The generator writes `train.csv`, `validation.csv`, `gallery.csv`, `query.csv`, and `verification_pairs.csv` under `metadata/`. Whole designs are assigned to train, validation, or evaluation, so a design in train is never also in the retrieval evaluation. Evaluation designs have disjoint query and gallery images. The query selection prefers a palette different from its own design's gallery examples when available.

Positive pairs use the same human-confirmed design ID and prioritize different labeled colorways. Negative pairs use different design IDs and prioritize the same manually labeled colorway. This same-colorway negative is only a useful hard-negative proxy; different color names are not evidence that palettes are perceptually similar. `color_relationship` is `same`, `different`, or `unknown`. Split randomness uses a fixed seed.

Run validation separately or as a post-split check:

```shell
python scripts/validate_dataset.py --metadata metadata/image_metadata.csv --labels metadata/design_labels.csv --deeplure-root /data/local/deeplure --kaggle-root /data/local/kaggle --split-dir metadata
```

The validator checks missing/unreadable/changed files, duplicate IDs/paths/content, blank and invalid labels, singleton designs, source-root escapes, design and image leakage, gallery/query and train/query overlap, pair correctness, and pair counts. Colorway blanks or a lack of cross-color positives are reported as warnings; missing design labels, duplicates that can leak, and invalid pairs fail validation.

## Optional Similarity Suggestions

```shell
python scripts/find_similar_candidates.py --metadata metadata/image_metadata.csv --deeplure-root /data/local/deeplure --kaggle-root /data/local/kaggle --output local_outputs/candidate_pairs.csv
```

Every result is marked `NEEDS_HUMAN_REVIEW`. Similarity is a candidate-generation hint only and is never ground truth or a design label.

## Tests and Git Safety

Synthetic images are created in a temporary directory during tests; no proprietary images are used or checked in:

```shell
python -m unittest discover -s tests -v
```

Before publishing, run:

```shell
python scripts/check_git_safety.py
git status --short
git ls-files
```

The safety checker does not delete anything. It fails if sensitive image/archive/model files are tracked or absolute Windows paths are embedded in Python source, and warns about sensitive files found locally but ignored/untracked.