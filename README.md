# AIE-CASE: Color-Invariant Saree Design Recognition

Dataset preparation, PyTorch embedding training, and evaluation for a color-invariant saree design retrieval and verification task. The identity target is the underlying textile design (`design_id`), not its colorway.

## 500-Character Approach Note

ImageNet-pretrained ResNet18 maps each saree to a 128-D L2-normalized embedding. Contrastive learning uses human-confirmed same-design positives and different-design negatives, prioritizing positives across colorways. The trainable RGB model uses brightness, contrast, saturation, hue, and conservative geometry augmentation; a grayscale, three-channel ResNet18 baseline is compared. Retrieval ranks gallery embeddings for each query, and verification scores labeled image pairs.

## Proprietary Data

The DeepLure Drive corpus is proprietary third-party data. Keep the original archive and all extracted DeepLure images outside this Git repository. Never commit or redistribute them, and do not create public image URLs. `.gitignore` blocks common image/archive/model files, and `scripts/check_git_safety.py` checks tracked and local workspace files. Metadata and documentation may be reviewed and committed; image files, archives, contact sheets, and model weights must not be.

The original DeepLure ZIP and all extracted images must remain outside this repository. The configured local data root is `%USERPROFILE%\DeepLureData\deeplure`.

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

The exact training command is included in [Reproduce the Reported Experiment](#reproduce-the-reported-experiment).

The best checkpoint is written to `local_outputs/model/best_model.pt`; the training configuration and loss history are saved beside it. `local_outputs/` is intentionally ignored by Git because it contains local model and evaluation artifacts. Do not commit those artifacts.

Training reads only `metadata/train.csv`; checkpoint selection uses `metadata/validation.csv`. Gallery, query, and verification evaluation data are not used for training or checkpoint selection.

## Evaluation Protocol

The current saved split has 117 training images from 34 designs and 24 validation images from 6 designs. The held-out evaluation contains 6 query images and 18 gallery images, each covering the same 6 evaluation designs. One image per evaluation design is the query; the remaining images for that design form its gallery. Train, validation, and evaluation design IDs are disjoint, and no image is shared between gallery and query.

Identification ranks every gallery embedding by cosine similarity for each query. A result is relevant when query and gallery have the same human-confirmed `design_id`, regardless of colorway. Reported identification metrics are Recall@1, Recall@3, Recall@5, and MRR. Cross-color Recall@1 restricts relevant gallery items to those with a different nonblank manually labeled colorway.

Verification positives have the same `design_id`; negatives have different `design_id`. The current evaluation has 18 positive pairs and 90 negative pairs, with 11 positives crossing labeled colorways. Fifteen designs have only one image in the full dataset and cannot produce positive image pairs.

Each model's cosine threshold is selected by maximum F1 over all unordered validation-image pairs (276 pairs in this split). That threshold is then fixed before scoring the held-out verification pairs. Evaluation pairs are not used for threshold selection. Verification reports ROC-AUC and F1 (plus EER and other threshold metrics).

## Results

Measured results from the saved evaluation artifacts:

| Model | Recall@1 | Recall@3 | Recall@5 | Cross-color Recall@1 | Verification ROC-AUC | Verification F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grayscale ResNet18 baseline | 0.3611 | 0.8750 | 1.0000 | 0.3958 | 0.9914 | 0.3186 |
| Trained color-augmentation model | 0.3611 | 0.9167 | 1.0000 | 0.3958 | 0.9951 | 0.5455 |

The evaluation set is small (6 queries and 108 verification pairs, including 18 positives), so these results should be interpreted as descriptive rather than as a statistically strong estimate of generalization. They do not establish statistical significance or general performance superiority. See [reports/experiment_results.md](reports/experiment_results.md) for additional metrics and threshold details.

## Efficiency

Measured from the trained checkpoint with `scripts/measure_efficiency.py`, using a deterministic 1x3x224x224 tensor, 20 warmups, 100 timed runs, and one CPU thread:

| Metric | Value |
| --- | --- |
| Backbone | ResNet18Embedding |
| Embedding dimension | 128 |
| Total parameters | 11,340,736 |
| Trainable parameters | 164,224 |
| CPU forward latency | 116.56 ms median (117.85 ms mean) |
| FLOPs/MACs | Not reported; no profiler dependency added |

Latency is a local measurement on Windows 11 with PyTorch 2.14.1+cpu and an Intel64 CPU; it covers model forward only and excludes image decoding/preprocessing. It is not a universal benchmark. Re-measure on the target machine with the command in the reproduction section.

## Reproduce the Reported Experiment

Keep the proprietary DeepLure dataset outside this repository at `$env:USERPROFILE\DeepLureData\deeplure`. Do not copy it into Git, commit it, upload it, or redistribute it. Install the basic dataset dependencies and model/evaluation dependencies with:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-similarity.txt
```

The current `requirements.txt` also contains the model/evaluation packages, while `requirements-similarity.txt` includes it; the commands are separated by workflow, but currently resolve to overlapping dependencies. Validate the prepared metadata first:

```powershell
python scripts/validate_dataset.py `
  --metadata metadata\image_metadata.csv `
  --deeplure-root "$env:USERPROFILE\DeepLureData\deeplure"
```

Train from the saved train/validation splits:

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

Evaluate both the grayscale baseline and trained checkpoint:

```powershell
python scripts/evaluate_retrieval.py `
  --deeplure-root "$env:USERPROFILE\DeepLureData\deeplure" `
  --checkpoint local_outputs\model\best_model.pt `
  --mode both `
  --output-dir local_outputs\evaluation `
  --device cpu `
  --batch-size 16
```

Measure checkpoint efficiency locally:

```powershell
python scripts/measure_efficiency.py --checkpoint local_outputs\model\best_model.pt
```

The test and Git-safety commands are in [Tests and Git Safety](#tests-and-git-safety). `local_outputs/` is ignored by Git; keep checkpoints and generated evaluation artifacts there and do not commit them.

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