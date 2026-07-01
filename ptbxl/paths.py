"""Central paths for the extracted PTB-XL statement71 artifacts.

Environment variables can override the defaults:
    PTBXL_PTH_PATH
    PTBXL_MLB_PATH
    PTBXL_STANDARD_SCALER_PATH
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
PTBXL_RESOURCE_DIR = PROJECT_ROOT / "ptbxl" / "resources"


def _path_from_env(name, default_path):
    value = os.environ.get(name)
    if value:
        return value
    return str(Path(default_path))


PTBXL_REPO_ROOT = str(PROJECT_ROOT)
PTBXL_CODE_DIR = str(PROJECT_ROOT / "model")
PTBXL_OUTPUT_DIR = str(CHECKPOINT_DIR)
PTBXL_PTH_PATH = _path_from_env("PTBXL_PTH_PATH", CHECKPOINT_DIR / "fastai_xresnet1d101.pth")
PTBXL_MLB_PATH = _path_from_env("PTBXL_MLB_PATH", PTBXL_RESOURCE_DIR / "mlb.pkl")
PTBXL_STANDARD_SCALER_PATH = _path_from_env(
    "PTBXL_STANDARD_SCALER_PATH",
    PTBXL_RESOURCE_DIR / "standard_scaler.pkl",
)
