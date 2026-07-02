"""
Feature Engineering modules for auxiliary (non-deep) features.

This sub-package contains feature extraction logic that is intentionally
isolated from the main deep ECG backbone training pipeline:

  - statement71.py : PTB-XL 71-class semantic label feature extractor
                     (xresnet1d101 inference on I-CARE single-lead segments)
  - hrv.py         : Time-domain HRV (Heart Rate Variability) feature computation
                     (R-peak detection + 10 HRV statistics)

Both modules are imported by the main run_*.py scripts via the backward-
compatible shims in feature/statement71.py and feature/hrv.py, so existing
code does not need to change its import paths.
"""
