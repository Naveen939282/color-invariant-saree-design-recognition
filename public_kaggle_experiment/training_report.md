# Trained Color-Invariant Embedding Experiment

This is a **controlled synthetic color-invariance evaluation** over filename-derived candidate experiment identities. It is not a real-world cross-color saree design-recognition benchmark: the four dataset labels are textile categories, and the candidate groups are not verified real-world design IDs or genuine colorways.

## Model and Training

- Backbone: `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`; classifier removed.
- Projection: trainable `Linear(512, 128)` followed by L2 normalization; backbone and projection trained jointly.
- Objective: supervised contrastive loss. Canonical and fixed-transform views of one candidate `source_id` are positives; other candidate identities in the batch are negatives.
- Configuration: 10 epochs, batch size 8 image views (four candidate identities per update), AdamW, learning rate `1e-4`, weight decay `1e-4`, temperature `0.07`, seed 42, CPU.
- Only 431 train candidate identities were used for gradient updates. The 115 validation identities selected checkpoints; 60 test identities were held out.
- A deterministic schedule rotates one of the ten fixed non-identity transformations for each training candidate each epoch. No extra random transformations were added.
- Selected checkpoint: epoch 1, validation overall non-identity Recall@1 `0.9878`; exact ties retain the earlier epoch.
- Global cosine threshold: `0.77639711`, selected by maximizing pooled validation F1 over 132,250 non-identity validation pairs. Ties select the largest threshold. The same threshold is used for all test transforms; test labels were not used to choose it.

## Held-Out Test Results

| Transformation | Recall@1 | Recall@3 | Recall@5 | ROC-AUC | F1 |
|---|---:|---:|---:|---:|---:|
| identity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9524 |
| brightness decrease | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9600 |
| brightness increase | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9524 |
| saturation decrease | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9524 |
| saturation increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9449 |
| hue negative | 0.9667 | 1.0000 | 1.0000 | 0.9989 | 0.9355 |
| hue positive | 0.9833 | 1.0000 | 1.0000 | 0.9996 | 0.9244 |
| contrast decrease | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9524 |
| contrast increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9524 |
| grayscale | 0.9500 | 1.0000 | 1.0000 | 0.9973 | 0.8000 |
| combined moderate | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9524 |
| overall non-identity | 0.9900 | 1.0000 | 1.0000 | 0.9994 | 0.9344 |

## Baseline Comparison

Frozen baseline uses `ResNet18_Weights.IMAGENET1K_V1`, a frozen/eval ResNet18, native 512-D pooled features, and L2 normalization. Its validation-selected threshold was `0.84651804`.

| Transformation | Baseline R@1 | Trained R@1 | Baseline R@3 | Trained R@3 | Baseline R@5 | Trained R@5 | Baseline AUC | Trained AUC | Baseline F1 | Trained F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9231 | 0.9524 |
| brightness decrease | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9449 | 0.9600 |
| brightness increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 0.9302 | 0.9524 |
| saturation decrease | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9449 | 0.9524 |
| saturation increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9231 | 0.9449 |
| hue negative | 0.9833 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9994 | 0.9989 | 0.9194 | 0.9355 |
| hue positive | 0.9833 | 0.9833 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9996 | 0.9996 | 0.9027 | 0.9244 |
| contrast decrease | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9302 | 0.9524 |
| contrast increase | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9302 | 0.9524 |
| grayscale | 0.9667 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9962 | 0.9973 | 0.4416 | 0.8000 |
| combined moderate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9998 | 0.9449 | 0.9524 |
| overall non-identity | 0.9933 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9985 | 0.9994 | 0.8993 | 0.9344 |

Across the 60 transformation/metric comparisons, 14 improved, 35 were unchanged, and 11 regressed. Overall R@1 decreased by `0.0033`; R@3/R@5 were unchanged; ROC-AUC increased by `0.0009` and F1 by `0.0351`. Grayscale verification F1 rose from `0.4416` to `0.8000`, while grayscale R@1 declined from `0.9667` to `0.9500`. Results are mixed; no universal improvement is claimed.

## Compute and Limitations

- Trainable parameters: `11,242,176`; embedding dimension: 128.
- Frozen baseline feature extractor parameters: `11,176,512`.
- CPU training-loop time summed across epochs: `1,205.9 s`; validation time: `789.3 s`; combined recorded epoch time: `1,995.2 s`. Initialization and final checkpoint/test evaluation are excluded; this is not total process wall-clock time.
- CPU inference, batch size 32: baseline median `1,013.44 ms/batch` (`31.67 ms/image`); trained checkpoint median `1,022.2 ms/batch` (`31.94 ms/image`). Both use 10 warmups and 50 timed batches; preprocessing is excluded. Timing is hardware-dependent.
- Test protocol: 60 canonical test gallery images and 660 fixed transformed queries. Test identities were not used for training, checkpoint selection, or threshold selection.
- The labels are categories, not design IDs. Candidate groups are filename-derived and do not prove real design identity. Every query is a controlled transformation of its own canonical source image, not a genuine different-color product image. Do not interpret these measurements as real-world cross-color generalization.

The machine-generated detailed report, CSVs, selected checkpoint, configuration, and timing remain under ignored `local_outputs/kaggle_audit/trained_contrastive/`; no checkpoint or generated output is included in Git.