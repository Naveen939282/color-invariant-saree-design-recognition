# Model Evaluation

## System

The project produces image embeddings for retrieval and verification. It treats human-confirmed `design_id` as the identity target because colorway is an appearance variation, not the design itself. It does not train a classifier over the known design IDs, so an unseen design can still be represented and compared to a gallery.

## Baseline and Trained Model

The baseline uses torchvision's ImageNet-pretrained ResNet18 with its classification layer removed. RGB images are converted to grayscale, replicated to three channels, normalized with the pretrained weights' ImageNet statistics, and mapped to L2-normalized 512-D features. Similarity is cosine similarity.

The trainable model uses the same pretrained backbone and a 128-D projection head. It is initially frozen and optimized with balanced contrastive pairs based only on the training split's human design labels. Positive pairs preferentially cross colorways; negative pairs preferentially share a manually labeled colorway. RGB color jitter varies brightness, contrast, saturation, and hue while preserving most image geometry. Validation contrastive loss selects the checkpoint. Evaluation images and labels are not used for training or checkpoint selection.

## Data Splits and Leakage

Train, validation, and retrieval evaluation are design-disjoint. Gallery and query contain separate images from evaluation designs. Training does not read gallery, query, or verification-pair files. The evaluation threshold is selected by maximizing validation-pair F1; ties choose the higher cosine threshold. Evaluation verification pairs are scored only after that threshold is fixed.

## Results

The actual CPU run is recorded in [experiment_results.md](experiment_results.md). The frozen-backbone model trained for 10 epochs with 64 sampled pairs per epoch; epoch 4 had the lowest validation contrastive loss (0.1712). On the held-out evaluation set, grayscale baseline and trained model both achieved Recall@1 0.3611, cross-color Recall@1 0.3958, Recall@5 1.0, and MRR 1.0. The trained model measured Recall@3 0.9167 versus 0.8750, ROC-AUC 0.9951 versus 0.9914, and verification F1 0.5455 versus 0.3186. This is a mixed, small-sample comparison, not evidence of a general performance improvement.

There are only six queries (four with cross-color relevant gallery items) and 108 verification pairs (18 positive, 90 negative). Verification F1 is sensitive to that class imbalance. The thresholds were selected on validation pairs: 0.5877 for the baseline and 0.8098 for the trained model. Evaluation F1 was 0.3186 and 0.5455, respectively; both thresholds produced recall 1.0, so many negatives were accepted. EER was approximately 0.0611 for the baseline and 0.0500 for the trained model. MRR is 1.0 for both because each query's first relevant result is ranked first, even though additional same-design images can rank lower.

The evaluator writes full JSON metrics, ranked retrieval results, scored verification pairs, and embeddings to ignored `local_outputs/evaluation/`.

Reported retrieval Recall@K is the macro average over eligible queries of the fraction of all relevant same-design gallery images retrieved in the top K. MRR uses the first relevant rank. Cross-color metrics count only same-design gallery items with a different nonblank colorway. Queries with no relevant gallery item are counted separately and excluded from recall/MRR means. Verification includes ROC-AUC, approximate EER, accuracy, precision, recall, F1, and class-conditional similarity distributions.

## Limitations and Failure Cases

The corpus is small and has singleton designs that cannot create positive pairs. The grayscale baseline can discard useful texture contrast. Color jitter only encourages robustness; it cannot guarantee color invariance. Training and evaluation distributions may differ, and manually entered colorway names do not measure perceptual color distance. A query without a relevant gallery item is explicitly reported rather than counted as a fabricated hit. Small pair counts make threshold and ROC/EER estimates uncertain.

## Future Improvements

Add human-confirmed cross-palette examples, compare LAB/L-channel inputs, run several fixed seeds, report confidence intervals, and tune augmentation only on validation data. Any added source should retain its own provenance and use independently verified motif labels.