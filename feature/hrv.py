# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible re-export shim.
#
# The canonical implementation has been moved to:
#   evaluation/feature_engineering/hrv.py
#
# This file re-exports everything from there so that all existing imports of
# the form:
#   from feature.hrv import ...
# continue to work without modification.
# ─────────────────────────────────────────────────────────────────────────────
from evaluation.feature_engineering.hrv import *  # noqa: F401, F403
