"""Helpers for importing the extracted PTB-XL statement71 model code."""

import os

from .paths import (
    PTBXL_CODE_DIR,
    PTBXL_MLB_PATH,
    PTBXL_OUTPUT_DIR,
    PTBXL_PTH_PATH,
    PTBXL_REPO_ROOT,
)


def ensure_ptbxl_code_on_path():
    if not os.path.isdir(PTBXL_CODE_DIR):
        raise FileNotFoundError(f"PTB-XL local model directory not found: {PTBXL_CODE_DIR}")
    return PTBXL_CODE_DIR


def describe_ptbxl_paths():
    return {
        "repo_root": PTBXL_REPO_ROOT,
        "code_dir": PTBXL_CODE_DIR,
        "output_dir": PTBXL_OUTPUT_DIR,
        "pth_path": PTBXL_PTH_PATH,
        "mlb_path": PTBXL_MLB_PATH,
        "repo_exists": os.path.isdir(PTBXL_REPO_ROOT),
        "code_exists": os.path.isdir(PTBXL_CODE_DIR),
        "pth_exists": os.path.exists(PTBXL_PTH_PATH),
        "mlb_exists": os.path.exists(PTBXL_MLB_PATH),
    }


def import_xresnet1d101():
    ensure_ptbxl_code_on_path()
    from model.ptbxl_xresnet1d import xresnet1d101

    return xresnet1d101


__all__ = ["ensure_ptbxl_code_on_path", "describe_ptbxl_paths", "import_xresnet1d101"]
