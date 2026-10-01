# Experiment Results

Measurements below come from the actual commands documented in `README.md` using the current validated split files. No values were inferred from training loss.

| Experiment | Model | Input representation | Loss | Embedding dimension | Epochs | Validation result | Retrieval Recall@1 | Recall@3 | Recall@5 | MRR | Verification ROC-AUC | Verification F1 | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-10-01 | Pretrained ResNet18 baseline | Grayscale replicated to RGB | None (ImageNet pretrained) | 512 | 0 | Validation F1 0.6799; threshold 0.5877 | 0.3611 | 0.8750 | 1.0000 | 1.0000 | 0.9914 | 0.3186 | 6 queries; 108 verification pairs; threshold selected on 276 validation pairs |
| 2026-10-01 | ResNet18 metric-learning model | RGB with strong color jitter | Contrastive | 128 | 10 | Best validation loss 0.1712 at epoch 4; validation F1 0.8178; threshold 0.8098 | 0.3611 | 0.9167 | 1.0000 | 1.0000 | 0.9951 | 0.5455 | Frozen pretrained backbone; 64 balanced pairs/epoch; same 6 queries and 108 verification pairs |

Cross-color retrieval metrics are reported in `local_outputs/evaluation/evaluation_metrics.json`. Both models had cross-color Recall@1 0.3958; cross-color Recall@3 was 0.8750 for the baseline and 0.9375 for the trained model. Thresholds are calibrated on validation pairs, never on the evaluation verification pairs. With only six evaluation queries, these results are descriptive and may be unstable.