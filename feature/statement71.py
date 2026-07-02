# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible re-export shim.
#
# The canonical implementation has been moved to:
#   evaluation/feature_engineering/statement71.py
#
# This file re-exports everything from there so that all existing imports of
# the form:
#   from feature.statement71 import ...
# continue to work without modification.
# ─────────────────────────────────────────────────────────────────────────────
from evaluation.feature_engineering.statement71 import *  # noqa: F401, F403
