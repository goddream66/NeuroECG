"""Entry point for the ECGFounder + statement71 training pipeline."""

import sys

from pipeline.train import run_training_pipeline


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
    run_training_pipeline()
