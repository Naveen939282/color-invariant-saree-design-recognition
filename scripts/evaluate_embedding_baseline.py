"""Evaluate the pretrained grayscale ResNet18 baseline only."""

from __future__ import annotations

import sys

try:
    from .evaluate_retrieval import main as evaluate_main
except ImportError:
    from evaluate_retrieval import main as evaluate_main


if __name__ == "__main__":
    try:
        raise SystemExit(evaluate_main(sys.argv[1:], forced_mode="baseline"))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error