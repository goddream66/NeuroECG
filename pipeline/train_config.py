# Shared configuration and imports for the training pipeline.
import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.signal import resample
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

from ptbxl.paths import PTBXL_MLB_PATH, PTBXL_PTH_PATH, PTBXL_REPO_ROOT
from utils.runtime import CPU_CORE_LIMIT, CPU_WORKERS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(PROJECT_ROOT)
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoint")
CACHE_DIR = SCRIPT_DIR
PARENT_EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)

torch.set_num_threads(CPU_WORKERS)
try:
    torch.set_num_interop_threads(CPU_WORKERS)
except RuntimeError:
    pass

MODEL_TIME_BUCKETS = ("0-12h", "12-24h", "24-48h", "48-72h", ">72h")
TEMPORAL_AGG_QUANTILE = 0.24
PATIENT_POOLING_TAG = f"q{int(round(TEMPORAL_AGG_QUANTILE * 100)):02d}"
STATIC_FEATURE_NAMES = (
    "age_years",
    "sex_male",
    "rosc_minutes",
    "ohca",
    "shockable_rhythm",
    "ttm_celsius",
)
HRV_FEATURE_NAMES = (
    "nn50_count",
    "pnn50",
    "rmssd",
    "sd1",
    "sd2",
    "sd1_sd2_ratio",
    "sd2_sd1_ratio",
    "csi",
    "cvi",
    "rr_diff_std_mean_ratio",
)
TARGET_FS = 500.0

ECGFOUNDER_ROOT = os.environ.get("ECGFOUNDER_ROOT", str(PROJECT_ROOT / "model"))
ECGFOUNDER_1LEAD_CKPT = os.environ.get(
    "ECGFOUNDER_1LEAD_CKPT",
    str(PROJECT_ROOT / "checkpoint" / "1_lead_ECGFounder.pth"),
)
ECGFOUNDER_FEATURE_DIM = 1024
ECGFOUNDER_FREEZE_BACKBONE = False
ECGFOUNDER_TRAIN_PROJECTOR = False
ECGFOUNDER_PROJECTOR_DIM = None
ECGFOUNDER_PROJECTOR_DROPOUT = 0.0
ECGFOUNDER_HEAD_LR = 1e-4
ECGFOUNDER_LAST_TWO_STAGES_LR = 1e-5
ECGFOUNDER_LAST_FOUR_STAGES_LR = 5e-6
ECGFOUNDER_FULL_BACKBONE_LR = 1e-6
ECGFOUNDER_FREEZE_HEAD_ONLY_EPOCHS = 7
ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH = 7
ECGFOUNDER_UNFREEZE_LAST_FOUR_STAGES_EPOCH = 15
ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH = 20
ECGFOUNDER_INPUT_ZSCORE = False
STAGE1_BACKBONE = "ecgfounder"
ECG_JEPA_ROOT = os.environ.get("ECG_JEPA_ROOT", str(PROJECT_ROOT / "ECG_JEPA"))
ECG_JEPA_CKPT = os.environ.get(
    "ECG_JEPA_CKPT",
    str(PROJECT_ROOT / "checkpoint" / "ecg_jepa_multiblock_epoch100.pth"),
)
ECG_JEPA_FEATURE_DIM = 768
ECG_JEPA_TARGET_LEN = 2500
ECG_JEPA_LEADS = (1,)
ECGFM_ROOT = os.environ.get("ECGFM_ROOT", str(PROJECT_ROOT / "ecg-fm"))
ECGFM_CKPT = os.environ.get(
    "ECGFM_CKPT",
    str(PROJECT_ROOT / "checkpoint" / "ecgfm_mimic_iv_ecg_physionet_pretrained.pt"),
)
ECGFM_FEATURE_DIM = 768
ECGFM_N_LEADS = 12
ECGFM_TARGET_LEN = 2500
ECGFM_SINGLE_LEAD_INDEX = 1
ST_MEM_ROOT = os.environ.get("ST_MEM_ROOT", str(PROJECT_ROOT / "ST-MEM"))
ST_MEM_CKPT = os.environ.get(
    "ST_MEM_CKPT",
    str(PROJECT_ROOT / "checkpoint" / "st_mem_vit_base_encoder.pth"),
)
ST_MEM_FEATURE_DIM = 768
ST_MEM_N_LEADS = 12
ST_MEM_TARGET_LEN = 2250
ST_MEM_PATCH_SIZE = 75
ST_MEM_SINGLE_LEAD_INDEX = 1
SERESNET_FEATURE_DIM = 1024
SERESNET_DROPOUT = 0.5
SERESNET_LR_LOW = 1e-5
SERESNET_LR_MID = 3e-5
SERESNET_LR_HIGH = 1e-4
SERESNET_LR_LOW_EPOCHS = 10
SERESNET_LR_MID_EPOCHS = 20
SERESNET_BACKBONE_LR = SERESNET_LR_LOW
SERESNET_HEAD_LR = SERESNET_LR_LOW
CONVNEXT1D_FEATURE_DIM = 1024
CONVNEXT1D_DIMS = (64, 128, 256, 1024)
CONVNEXT1D_DEPTHS = (2, 2, 6, 2)
CONVNEXT1D_KERNEL_SIZE = 7
CONVNEXT1D_LAYER_SCALE_INIT = 1e-6
CONVNEXT1D_DROP_PATH_RATE = 0.1
CONVNEXT1D_DROPOUT = 0.2
CONVNEXT1D_LR_LOW = 1e-5
CONVNEXT1D_LR_MID = 3e-5
CONVNEXT1D_LR_HIGH = 1e-4
CONVNEXT1D_LR_LOW_EPOCHS = 10
CONVNEXT1D_LR_MID_EPOCHS = 20
GLOBAL_AGG_INCLUDE_COUNT = False
DEEP_PCA_DIM = 64
AUX_FEATURE_PROJECTOR_DIM = 1024
AUX_FEATURE_PROJECTOR_SEED = 42

META_DIR = "/data/xcy_group/gjj/cinc2023/cinc2023_dataset/training/"
PROCESSED_DIR = "/data/xcy_group/gjj/cinc2023/cinc2023_dataset/processed_npy_500hz_qc_mask_flatline_only/"
PROCESSED_INDEX_PKL = "processed_index.pkl"
RECORD_CACHE = os.path.join(
    CACHE_DIR,
    "processed_records_cache_first72h_500hz_singlelead_samples_w5000_s10000_qcmask_flatline_only_v1.pkl",
)
REQUIRE_RECORD_CACHE = False
RESUME_CHECKPOINT_PATH = os.path.join(
    PARENT_EXPERIMENT_DIR,
    "latest_resume_checkpoint_bce_ecg409_e30_mil30_topk5_projln_valauroc_"
    "qcmask_segpol_qrs_v1_robustmil_qgate_ptbxl_random12_slowtopk_q88_times.pth",
)

PTBXL_FEED_MODE = "leadII_zero"
PTBXL_TARGET_LEN = 1000
PTBXL_N_LEADS = 12
PTBXL_CROP_LEN = 250
PTBXL_CROP_STRIDE = 125
PTBXL_CROP_AGG = "max"
WVNUM = 5
ENTRY_STRIDE_SEGMENTS = 5
DROP_INCOMPLETE_ENTRY = True
ENTRY_STATEMENT_AGG_MODE = "mean"
PATIENT_ENTRY_AGG_MODE = "mean"

WINDOW_SIZE = 5000
STRIDE = 10000
INPUT_CLIP_VALUE = 5.0
SCAN_PROCESSES = 3
MAX_OBSERVATION_HOUR = 72.0
FIRST72_TIME_BUCKETS = ("0-12h", "12-24h", "24-48h", "48-72h")
SELECTED_TIME_BUCKETS = FIRST72_TIME_BUCKETS
MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT = 409

BATCH_SIZE = 512
DATALOADER_WORKERS = min(2, CPU_WORKERS)
DATALOADER_PIN_MEMORY = False
NUM_WORKERS = DATALOADER_WORKERS
STAGE1_MAX_SEGMENTS_PER_PATIENT_PER_EPOCH = 256
STAGE1_EPOCHS = 25
STAGE1_SEGMENT_TRAIN_BATCH_SIZE = 512

CATBOOST_ITERATIONS = 100
CATBOOST_DEPTH = 4
CATBOOST_LR = 0.03
CLASS_WEIGHTS = [1.7, 1.0]
NUM_RUNS = 5
BASE_SEED = 42

SPLIT_TRAIN_SIZE = 0.70
SPLIT_VAL_SIZE = 0.15
SPLIT_REUSE_STAGE1_RESUME = False
SPLIT_REUSE_EXTERNAL_RESUME = False

USE_DEEP_STATIC_GATED_FUSION = False
USE_CLINICAL_RESIDUAL_DEEP_FUSION = False
FUSION_ALIGN_DIM = 128
FUSION_HIDDEN_DIM = 128
FUSION_DROPOUT = 0.2
FUSION_BATCH_SIZE = 32
FUSION_EPOCHS = 120
FUSION_PATIENCE = 20
FUSION_LR = 1e-3
FUSION_WEIGHT_DECAY = 1e-4
FUSION_CPC_LOSS_WEIGHT = 0.05
FIXED_GATE_DEEP_WEIGHT = 0.485
FIXED_GATE_STATIC_WEIGHT = 0.515

STAGE2_ONLY_LOAD_STAGE1_CKPT = False
SKIP_STAGE1_DIRECT_EVAL_IN_STAGE2_ONLY = True

STATEMENT71_OUTPUT_DIM = 71
SEMANTIC_FEATURE_NAME = "statement71"
SEMANTIC_OUTPUT_DIM = STATEMENT71_OUTPUT_DIM
SEMANTIC_FEED_MODE = PTBXL_FEED_MODE
TAG = f"first72h_500hz_{SEMANTIC_FEATURE_NAME}_wvnum{WVNUM}_{SEMANTIC_FEED_MODE}_{ENTRY_STATEMENT_AGG_MODE}_entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
RAW_CSV = os.path.join(SCRIPT_DIR, f"{SEMANTIC_FEATURE_NAME}_wvnum5_ablation_v1_{TAG}_raw.csv")
SUMMARY_CSV = os.path.join(SCRIPT_DIR, f"{SEMANTIC_FEATURE_NAME}_wvnum5_ablation_v1_{TAG}_summary.csv")
CONFIG_JSON = os.path.join(SCRIPT_DIR, f"{SEMANTIC_FEATURE_NAME}_wvnum5_ablation_v1_{TAG}_config.json")
if STAGE1_BACKBONE.lower() == "seresnet":
    TRAIN_STAGE1_CKPT_TAG = (
        f"first72h_500hz_seresnet1lead_e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_"
        f"valloss_pat6_singlelead_samples_v1_{PTBXL_FEED_MODE}_wvnum{WVNUM}_"
        f"entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
    TRAIN_TAG = (
        f"first72h_500hz_seresnet1lead_e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_"
        f"valloss_pat6_singlelead_samples_v1_{SEMANTIC_FEATURE_NAME}_{SEMANTIC_FEED_MODE}_"
        f"wvnum{WVNUM}_entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
elif STAGE1_BACKBONE.lower() == "convnext1d":
    TRAIN_STAGE1_CKPT_TAG = (
        f"first72h_500hz_convnext1d_e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_"
        f"d{CONVNEXT1D_DEPTHS[0]}{CONVNEXT1D_DEPTHS[1]}{CONVNEXT1D_DEPTHS[2]}{CONVNEXT1D_DEPTHS[3]}_"
        f"dim{CONVNEXT1D_FEATURE_DIM}_valloss_pat6_singlelead_samples_v1_"
        f"{PTBXL_FEED_MODE}_wvnum{WVNUM}_entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
    TRAIN_TAG = (
        f"first72h_500hz_convnext1d_e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_"
        f"d{CONVNEXT1D_DEPTHS[0]}{CONVNEXT1D_DEPTHS[1]}{CONVNEXT1D_DEPTHS[2]}{CONVNEXT1D_DEPTHS[3]}_"
        f"dim{CONVNEXT1D_FEATURE_DIM}_valloss_pat6_singlelead_samples_v1_"
        f"{SEMANTIC_FEATURE_NAME}_{SEMANTIC_FEED_MODE}_wvnum{WVNUM}_"
        f"entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
elif STAGE1_BACKBONE.lower() == "ecg_jepa":
    TRAIN_STAGE1_CKPT_TAG = (
        f"first72h_500hz_ecgjepa1lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_"
        f"e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_valloss_pat6_"
        f"singlelead_samples_v1_{PTBXL_FEED_MODE}_wvnum{WVNUM}_"
        f"entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
    TRAIN_TAG = (
        f"first72h_500hz_ecgjepa1lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_"
        f"e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_valloss_pat6_"
        f"singlelead_samples_v1_{SEMANTIC_FEATURE_NAME}_{SEMANTIC_FEED_MODE}_"
        f"wvnum{WVNUM}_entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
elif STAGE1_BACKBONE.lower() in {"ecgfm", "ecg_fm"}:
    TRAIN_STAGE1_CKPT_TAG = (
        f"first72h_500hz_ecgfm1lead_to12lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_"
        f"e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_valloss_pat6_"
        f"singlelead_samples_v1_{PTBXL_FEED_MODE}_wvnum{WVNUM}_"
        f"entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
    TRAIN_TAG = (
        f"first72h_500hz_ecgfm1lead_to12lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_"
        f"e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_valloss_pat6_"
        f"singlelead_samples_v1_{SEMANTIC_FEATURE_NAME}_{SEMANTIC_FEED_MODE}_"
        f"wvnum{WVNUM}_entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
elif STAGE1_BACKBONE.lower() == "stmem":
    TRAIN_STAGE1_CKPT_TAG = (
        f"first72h_500hz_stmem1lead_to12lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_"
        f"e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_valloss_pat6_"
        f"singlelead_samples_v1_{PTBXL_FEED_MODE}_wvnum{WVNUM}_"
        f"entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
    TRAIN_TAG = (
        f"first72h_500hz_stmem1lead_to12lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_"
        f"e{STAGE1_EPOCHS}_b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_valloss_pat6_"
        f"singlelead_samples_v1_{SEMANTIC_FEATURE_NAME}_{SEMANTIC_FEED_MODE}_"
        f"wvnum{WVNUM}_entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
else:
    TRAIN_STAGE1_CKPT_TAG = "first72h_500hz_ecgfounder1lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_e25_b512_valloss_pat6_singlelead_samples_v1_leadII_zero_wvnum5_entrycap409"
    TRAIN_TAG = (
        f"first72h_500hz_ecgfounder1lead_gradual_unfreeze_h7_l2e8_l4e16_fulle21_e{STAGE1_EPOCHS}_"
        f"b{STAGE1_SEGMENT_TRAIN_BATCH_SIZE}_valloss_pat6_singlelead_samples_v1_"
        f"{SEMANTIC_FEATURE_NAME}_{SEMANTIC_FEED_MODE}_wvnum{WVNUM}_"
        f"entrycap{MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT}"
    )
TRAIN_OUTPUT_TAG = f"{TRAIN_TAG}_stage1train_deeppca{DEEP_PCA_DIM}_auxproj{AUX_FEATURE_PROJECTOR_DIM}"
TRAIN_RAW_CSV = os.path.join(SCRIPT_DIR, f"{TRAIN_OUTPUT_TAG}_raw.csv")
TRAIN_SUMMARY_CSV = os.path.join(SCRIPT_DIR, f"{TRAIN_OUTPUT_TAG}_summary.csv")
TRAIN_CONFIG_JSON = os.path.join(SCRIPT_DIR, f"{TRAIN_OUTPUT_TAG}_config.json")
TRAIN_BEST_CKPT = os.path.join(CHECKPOINT_DIR, f"{TRAIN_STAGE1_CKPT_TAG}_best.pth")
TRAIN_RESUME_CKPT = os.path.join(CHECKPOINT_DIR, f"{TRAIN_STAGE1_CKPT_TAG}_resume.pth")
