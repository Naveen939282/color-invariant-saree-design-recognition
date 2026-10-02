"""Measure checkpoint parameter counts and local CPU embedding latency.

Latency covers model forward only for a deterministic 1x3xHxW tensor; it does
not include image decoding or preprocessing and is machine-dependent.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .evaluate_retrieval import load_trained_checkpoint
except ImportError:
    from evaluate_retrieval import load_trained_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.runs < 2 or args.image_size < 32 or args.threads < 1:
        parser.error("warmup must be nonnegative; runs >= 2; image-size >= 32; threads >= 1")

    torch.set_num_threads(args.threads)
    model = load_trained_checkpoint(args.checkpoint, torch.device("cpu")).eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sample = torch.randn(
        (1, 3, args.image_size, args.image_size), generator=generator, device="cpu"
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
        timings_ms = []
        for _ in range(args.runs):
            started = time.perf_counter()
            model(sample)
            timings_ms.append((time.perf_counter() - started) * 1000.0)
        embedding = model(sample)

    report = {
        "model": "ResNet18Embedding",
        "checkpoint": str(args.checkpoint),
        "embedding_dimension": int(embedding.shape[1]),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "device": "cpu",
        "input_shape": list(sample.shape),
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "threads": args.threads,
        "latency_scope": "model forward only; excludes image decode and preprocessing",
        "latency_ms_median": statistics.median(timings_ms),
        "latency_ms_mean": statistics.mean(timings_ms),
        "latency_ms_p90": float(np.percentile(timings_ms, 90)),
        "measurement_environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported by platform",
            "torch": torch.__version__,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error