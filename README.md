# AIE-CASE: Color-Invariant Saree Design Recognition

Dataset preparation, PyTorch embedding training, and evaluation for a color-invariant saree design retrieval and verification task. The identity target is the underlying textile design (`design_id`), not its colorway.

## Proprietary Data

The DeepLure Drive corpus is proprietary third-party data. Keep the original archive and all extracted DeepLure images outside this Git repository. Never commit or redistribute them, and do not create public image URLs. `.gitignore` blocks common image/archive/model files, and `scripts/check_git_safety.py` checks tracked and local workspace files. Metadata and documentation may be reviewed and committed; image files, archives, contact sheets, and model weights must not be.

The original workspace ZIP was already present when this project work began and remains ignored, not tracked. Prefer moving it and all extracted files outside the repository. The previously extracted working copy was moved to `%USERPROFILE%\DeepLureData\deeplure` on this machine.

## Install

The basic audit and split workflow uses pandas and Pillow. Install its requirements with:

```shell
python -m pip install -r requirements.txt
```

For model training and evaluation, use the model requirements command:

```shell
python -m pip install -r requirements-similarity.txt
```

In the current checkout, `requirements.txt` also lists NumPy, scikit-learn, matplotlib, PyTorch, and torchvision; `requirements-similarity.txt` currently includes `requirements.txt` and adds no further packages. The commands are shown separately for the basic-data and model workflows, but the files presently resolve to overlapping dependencies.

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

## Model Training

The embedding model is for retrieval and verification of the same underlying design across palettes; it does not classify a fixed set of design IDs. It uses an ImageNet-pretrained ResNet18 backbone and a learned projection head to produce 128-dimensional L2-normalized embeddings. Contrastive learning trains on balanced positive and negative image pairs: positive pairs share a human-confirmed `design_id`, while negative pairs have different design IDs. Cross-color positive pairs are preferred when available. The backbone is frozen initially.

Two input strategies are implemented:

- **Grayscale baseline:** RGB is converted to grayscale, replicated into three channels, and passed through pretrained ResNet18 preprocessing.
- **Trained color-augmentation model:** RGB is retained while brightness, contrast, saturation, and hue vary during training. Geometry is conservative: a near-full-frame crop is used, and horizontal flipping is opt-in to avoid changing directional textile layouts.

The actual CPU training configuration was:

- Epochs: 10
- Batch size: 8 pairs
- Pairs per epoch: 64
- Validation pairs: 64
- Embedding dimension: 128
- Random seed: 42
- Training split: `metadata/train.csv`
- Validation split: `metadata/validation.csv`
- Output directory: `local_outputs/model`
- Device: CPU

Run the same training command with an external DeepLure root:

```powershell
python scripts/train_embedding.py `
  --deeplure-root "$env:USERPROFILE\DeepLureData\deeplure" `
  --train-csv metadata\train.csv `
  --validation-csv metadata\validation.csv `
  --output-dir local_outputs\model `
  --epochs 10 `
  --batch-size 8 `
  --pairs-per-epoch 64 `
  --validation-pairs 64 `
  --embedding-dim 128 `
  --seed 42 `
  --device cpu
```

The best checkpoint is written to `local_outputs/model/best_model.pt`; the training configuration and loss history are saved beside it. `local_outputs/` is intentionally ignored by Git because it contains local model and evaluation artifacts. Do not commit those artifacts.

The workflow is: install basic dependencies, install model dependencies, prepare and validate metadata/splits using the sections above, train from the train and validation CSVs, then evaluate against the held-out gallery, query, and verification pairs. Training does not load the evaluation images or labels.

Evaluation compares the grayscale ResNet18 baseline with the trained color-augmentation model. Retrieval metrics include Recall@1/3/5 and MRR; verification metrics include ROC-AUC, EER, and threshold-based precision/recall/F1. The cosine threshold is selected on validation pairs, not evaluation pairs.

In the measured task experiment, both models had Recall@1 0.3611 and cross-color Recall@1 0.3958. The trained model had Recall@3 0.9167 versus 0.8750 for the grayscale baseline, and ROC-AUC 0.9951 versus 0.9914. These are results for this split, not a general performance claim: evaluation contained only 6 queries and 108 verification pairs. See [reports/experiment_results.md](reports/experiment_results.md) for the comparison and caveats.

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