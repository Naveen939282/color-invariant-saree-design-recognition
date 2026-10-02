# Public Kaggle Experiment: Dataset Audit

This is an isolated experiment using only the public Roboflow/Kaggle `archive.zip`. It does not change the submitted pipeline or add external data. The audit/protocol stages do not train or download weights. The frozen baseline stage uses torchvision's pretrained ImageNet ResNet18 weights for inference only; it does not fine-tune or train a projection. Keep the archive and generated output local; the repository ignores `.zip`, image files, and `local_outputs/`.

The dataset contains four textile/pattern categories: Banarasi, Bandhani, Ikat, and Pichwai. These are **not individual design identities**. The dataset therefore cannot directly supply real-world same-design/different-color pairs. A later experiment may apply controlled color transformations to source images, but that evaluation must be described as **synthetic/controlled color-invariance evaluation**, not real cross-color evaluation.

## Run the Audit

The script accepts an extracted dataset root or the uploaded ZIP directly. The expected layout has `train/`, `valid/`, and `test/`, each with exactly the four category folders.

The completed archive audit found 1,470 members (1,468 images plus two README files), all images 640x640 RGB. Class totals are Banarasi 489, Bandhani 316, Ikat 342, and Pichwai 321; split totals are train 1,293, valid 115, and test 60. It found 606 class-scoped candidate filename groups (431 groups of three and 175 singletons), five exact duplicate MD5 groups all within train, and no candidate-group overlap across splits. Filename groups remain candidates, not verified designs.

Windows PowerShell from the submitted repository root:

```powershell
python public_kaggle_experiment/scripts/audit_kaggle_dataset.py `
  --dataset-root "..\DEEPLURE-Kaggle\archive.zip" `
  --output-dir "public_kaggle_experiment\local_outputs\kaggle_audit" `
  --seed 42
```

Kaggle/Linux:

```shell
python public_kaggle_experiment/scripts/audit_kaggle_dataset.py \
  --dataset-root /kaggle/input/indian-fabric-patterns/archive.zip \
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit \
  --seed 42
```

Outputs are `metadata.csv`, `source_groups.csv`, `summary_report.md`, `class_contact_sheet.png`, and `variant_contact_sheet.png`. Filename-derived source IDs are class-scoped candidates: the Roboflow `.rf.<hexhash>` is removed, the encoded source extension is normalized, and no filename is treated as a verified design ID. Pixel comparisons and contact sheets provide evidence for review, not identity ground truth.

## Tests

From this directory:

```shell
python -m unittest discover -s tests -v
```

## Controlled Color-Invariance Protocol

This dataset's four labels are textile/pattern categories, not individual design IDs. Its three-image filename groups are near-identical noise-augmented variants, not verified real colorways. The experiment treats one class-scoped filename group as one **candidate experiment identity**; it does not claim that the candidate is a verified real-world design identity.

For each candidate group, `create_canonical_images.py` selects exactly one image: the lexicographically smallest audited relative path. The canonical CSV records the source ID, candidate identity ID, split, class, selected path and MD5, original group size, and selection rule. The existing noisy copies are not separate identities.

The manifest specifies, but does not save, one query transformation per canonical image for each of the 11 named transformations. Parameters are fixed in `src/color_transforms.py` before evaluation. The transforms operate on RGB values only and preserve image geometry. No stochastic operation is used, so transform seeds are not applicable. `transformed_path` remains empty: variants are generated on demand from the canonical image and its recorded configuration.

Fixed configuration:

| Transformation | Parameters |
|---|---|
| Identity | No change |
| Brightness decrease / increase | 0.75 / 1.25 |
| Saturation decrease / increase | 0.65 / 1.35 |
| Hue shift negative / positive | -0.10 / +0.10 of a full hue cycle |
| Contrast decrease / increase | 0.80 / 1.20 |
| Grayscale | Pillow luminance conversion, replicated to RGB |
| Combined moderate | Brightness 1.10, saturation 1.15, hue +0.05 cycle, contrast 1.10; in that order |

For test retrieval, `test_gallery.csv` contains one canonical image per test candidate identity; each query specification is matched against all test gallery identities, with its own `identity_id` as the correct answer. The identity transformation is an unchanged control and should be reported separately from color-transformed queries because it is pixel-identical to its gallery image. Retrieval metrics should include Recall@1, Recall@3, and Recall@5. Verification pairs label a query/gallery pair positive only when their candidate identity IDs match; all other test-gallery identities are negatives. Class labels are descriptive metadata only and never define identity or pair labels.

These results must be described as **controlled synthetic color-invariance evaluation**. This protocol is **not a claim of real-world cross-color generalization**: no verified same-design/different-color examples are present in this dataset.

Create local canonical and manifest files from the completed audit outputs (run from the repository root):

```powershell
python public_kaggle_experiment/scripts/create_canonical_images.py `
  --metadata-csv public_kaggle_experiment/local_outputs/kaggle_audit/metadata.csv `
  --output-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv

python public_kaggle_experiment/scripts/create_color_invariance_manifest.py `
  --canonical-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv `
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit/color_invariance
```

The manifest stage creates per-split transform specifications, a test gallery, test verification pairs, and a JSON summary. It writes no transformed images. The separate existing repository test suite includes its pre-existing synthetic one-epoch training smoke test; that test is unrelated to this public dataset experiment.

## Frozen ResNet18 Baseline

The baseline uses `torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`, all parameters frozen, eval mode, inference mode, and no gradients. The classifier is replaced with `Identity`; the native 512-dimensional pooled feature is L2-normalized. No learned projection or training split tuning is used. Preprocessing comes from the selected weight enum: resize to 256 with bilinear interpolation, center crop to 224, convert to RGB tensor, and normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

For verification, one global threshold maximizes F1 across pooled validation pairs from the ten non-identity transformations; ties choose the largest threshold. The identity control is excluded from threshold selection. That threshold is applied unchanged to every test transformation, and no test label participates in threshold selection.

Run from the repository root (pretrained weights are downloaded by torchvision only if not already cached):

```powershell
python public_kaggle_experiment/scripts/evaluate_resnet18_baseline.py `
  --canonical-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv `
  --manifest-dir public_kaggle_experiment/local_outputs/kaggle_audit/color_invariance `
  --dataset-root "..\DEEPLURE-Kaggle\archive.zip" `
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit/baseline_resnet18 `
  --device auto --batch-size 32 --num-workers 0 --seed 42
```

The evaluator writes per-transformation retrieval and verification CSVs, configuration, threshold, timing, summary JSON/CSV, and `baseline_report.md` under the ignored output directory. It checks split isolation, 60 gallery identities, 60 queries per transformation, finite unit-normalized embeddings, and query-to-gallery identity matches before reporting metrics. Inference timing includes a 10-run warm-up and 50 timed forward-only runs, separate from dataset loading, preprocessing, and tensor transfer.

The completed run and exact results are in the local ignored `local_outputs/kaggle_audit/baseline_resnet18/baseline_report.md`. Its validation-selected threshold was `0.84651804`; this is run-specific, not universally optimal. Treat all metrics as controlled synthetic color-invariance results over candidate filename groups, not as verified real-world design or colorway recognition.

## Trained Contrastive Model

The trained model starts from the same `ResNet18_Weights.IMAGENET1K_V1` ResNet18, removes its classifier, retains the 512-D pooled feature vector, and adds only `Linear(512, 128)` followed by L2 normalization. The backbone and projection are trained jointly. Candidate `source_id` values are used as **candidate experiment identities**, not verified design IDs. In each batch, the canonical view and one fixed transformed view of an identity are positives; views from other in-batch candidate identities are negatives. The transform assignment rotates deterministically through the ten non-identity fixed transforms across epochs, with no extra random augmentation.

The executed configuration was seed 42, 10 epochs, batch size 8 image views (four candidate identities per optimizer step), AdamW learning rate `1e-4`, weight decay `1e-4`, embedding dimension 128, contrastive temperature `0.07`, and CPU. Only the 431 training candidate identities were used for gradient updates. Each epoch, the best checkpoint was selected by overall non-identity Recall@1 pooled over validation transforms from the 115 validation identities; strict ties retain the earlier epoch. The selected checkpoint was epoch 1 with validation Recall@1 `0.9878`.

After checkpoint selection, one global cosine threshold was chosen on pooled non-identity validation pairs by maximizing F1, with ties resolved toward the largest threshold. The selected threshold was `0.77639711` (validation F1 `0.9256`) and was frozen for all test transformations. Test identities were not used for training, checkpoint selection, or threshold selection. Final evaluation used the unchanged test set of 60 canonical candidates and 660 transform queries.

Measured results and baseline comparison:

| Transformation | Baseline R@1 | Trained R@1 | Baseline R@3 | Trained R@3 | Baseline R@5 | Trained R@5 | Baseline AUC | Trained AUC | Baseline F1 | Trained F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9231 | 0.9524 |
| brightness decrease | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9449 | 0.9600 |
| brightness increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 0.9302 | 0.9524 |
| saturation decrease | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9449 | 0.9524 |
| saturation increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9231 | 0.9449 |
| hue shift negative | 0.9833 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9994 | 0.9989 | 0.9194 | 0.9355 |
| hue shift positive | 0.9833 | 0.9833 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9996 | 0.9996 | 0.9027 | 0.9244 |
| contrast decrease | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9302 | 0.9524 |
| contrast increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9302 | 0.9524 |
| grayscale | 0.9667 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9962 | 0.9973 | 0.4416 | 0.8000 |
| combined moderate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9998 | 0.9449 | 0.9524 |
| overall non-identity | 0.9933 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9985 | 0.9994 | 0.8993 | 0.9344 |

Across 60 transformation/metric comparisons, 14 improved, 35 were unchanged, and 11 regressed. Overall non-identity Recall@1 decreased by `0.0033`; grayscale Recall@1 decreased by `0.0167` while grayscale F1 increased by `0.3584`. Thus the result is mixed, with a substantial grayscale verification F1 gain but retrieval regressions; no blanket improvement is claimed. All Recall@3 and Recall@5 values were unchanged at 1.0.

The training loop took 1,205.9 seconds summed across epoch measurements; validation took 789.3 seconds, for 1,995.2 seconds combined recorded epoch time. These sums exclude model initialization and final test evaluation, so they are not full process wall-clock time. The model has 11,242,176 parameters, all trainable during training; inference is frozen. On CPU with inference batch size 32, median forward latency was 1,024.9 ms/batch or 32.03 ms/image after 10 warm-ups and 50 timed runs.

Detailed outputs, including `best_model.pt`, `config.json`, training history, validation metrics, test results, comparison CSV, timing, and `training_report.md`, are in the ignored local `local_outputs/kaggle_audit/trained_contrastive/` directory. The concise reviewed report is also checked in at `public_kaggle_experiment/training_report.md`. These findings remain a **controlled synthetic color-invariance evaluation**, not evidence of real-world cross-color generalization: transformations are applied to the same source image and candidate filename groups are not verified real-world design identities.

### Reproduce the Trained Experiment

Run from the repository root in PowerShell. Set `$datasetRoot` to the supplied external archive or an extracted dataset directory; the dataset must remain outside Git.

```powershell
$datasetRoot = Read-Host "Path to external archive.zip or extracted dataset root"

python public_kaggle_experiment/scripts/audit_kaggle_dataset.py `
  --dataset-root $datasetRoot `
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit `
  --seed 42

python public_kaggle_experiment/scripts/create_canonical_images.py `
  --metadata-csv public_kaggle_experiment/local_outputs/kaggle_audit/metadata.csv `
  --output-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv

python public_kaggle_experiment/scripts/create_color_invariance_manifest.py `
  --canonical-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv `
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit/color_invariance

python public_kaggle_experiment/scripts/evaluate_resnet18_baseline.py `
  --canonical-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv `
  --manifest-dir public_kaggle_experiment/local_outputs/kaggle_audit/color_invariance `
  --dataset-root $datasetRoot `
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit/baseline_resnet18 `
  --device auto --batch-size 32 --num-workers 0 --seed 42

python public_kaggle_experiment/scripts/train_color_invariant.py `
  --dataset-root $datasetRoot `
  --canonical-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv `
  --manifest-dir public_kaggle_experiment/local_outputs/kaggle_audit/color_invariance `
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit/trained_contrastive `
  --epochs 10 --batch-size 8 --learning-rate 1e-4 --weight-decay 1e-4 `
  --embedding-dim 128 --temperature 0.07 --seed 42 --device auto --num-workers 0

python public_kaggle_experiment/scripts/evaluate_trained_checkpoint.py `
  --dataset-root $datasetRoot `
  --canonical-csv public_kaggle_experiment/local_outputs/kaggle_audit/canonical_images.csv `
  --manifest-dir public_kaggle_experiment/local_outputs/kaggle_audit/color_invariance `
  --output-dir public_kaggle_experiment/local_outputs/kaggle_audit/trained_contrastive `
  --device auto --num-workers 0 --seed 42
```

The training command selects a checkpoint using validation data and creates the validation threshold before the separate checkpoint evaluator reads test manifests. For a fresh complete run, the training script also performs final test evaluation; the separate evaluator supports reproducible evaluation from an already selected checkpoint without retraining.

Run both requested test suites from the repository root:

```powershell
python -m unittest discover -s public_kaggle_experiment/tests -v
python -m unittest discover -s tests -v
python scripts/check_git_safety.py
git diff --check
```