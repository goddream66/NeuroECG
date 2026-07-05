# ECG I-CARE Deep + Static Outcome Modeling

Patient-level outcome prediction for the I-CARE ECG cohort using deep ECG representations and static clinical covariates.

This repository is currently organized around a simplified experiment setting:

- ECG waveform backbone features are kept.
- Static clinical covariates are kept.

The main pipeline fine-tunes or loads an ECG backbone, aggregates segment-level ECG embeddings to patient-level features, compresses deep embeddings with PCA, and evaluates CatBoost patient-level models.

## Table of Contents

- [Current Scope](#current-scope)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Data and Checkpoints](#data-and-checkpoints)
- [Main Workflows](#main-workflows)
- [Project Structure](#project-structure)
- [Key Configuration](#key-configuration)
- [Outputs and Metrics](#outputs-and-metrics)
- [Reproducibility Notes](#reproducibility-notes)
- [Common Issues](#common-issues)
- [Citation and Acknowledgement](#citation-and-acknowledgement)
- [License](#license)

## Current Scope

The active experimental design is:

```text
10-second ECG segments
        |
ECG backbone feature extractor
        |
patient-level quantile pooling, q = 0.24
        |
raw deep feature or PCA-compressed deep feature
        |
concatenation with static clinical covariates for the fused model
        |
CatBoost outcome classifier and CPC regressor
```

Active feature sets:

| Variant | Description |
|---|---|
| `static_only` | Static clinical covariates only |
| `deep_only` | Patient-level pooled deep ECG representation |
| `deep_pca64_only` | PCA-compressed deep ECG representation |
| `deep_pca64_static` | PCA-compressed deep ECG representation plus static clinical covariates |

## Quick Start

Run the main pipeline:

```bash
cd D:/ECG_I_CARE
python main.py train
```

Equivalent:

```bash
python main.py
```

Before running, check the local paths in:

```text
pipeline/train_config.py
```

At minimum, confirm:

```python
META_DIR = "/path/to/cinc2023/training/"
PROCESSED_DIR = "/path/to/processed_npy_500hz_qc_mask_flatline_only/"
ECGFOUNDER_1LEAD_CKPT = "/path/to/1_lead_ECGFounder.pth"
```

Quick syntax check:

```bash
python -c "import ast, pathlib; files=['main.py','pipeline/train_core.py','pipeline/train_config.py','run_pooling_sensitivity.py','run_pca_sensitivity_q024.py','run_shap_analysis.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8-sig'), filename=f) for f in files]; print('syntax ok')"
```

## Environment Setup

Recommended environment:

| Component | Recommended |
|---|---|
| Python | 3.10 or 3.11 |
| PyTorch | Match your CUDA driver |
| CUDA | Optional but strongly recommended |
| GPU | Recommended for backbone feature extraction/training |
| Main packages | `numpy`, `scipy`, `scikit-learn`, `pandas`, `wfdb`, `tqdm`, `catboost`, `matplotlib`, `shap` |

Example:

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn pandas wfdb tqdm catboost matplotlib shap
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only PyTorch:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Data and Checkpoints

### I-CARE / CinC 2023 Data

Raw data and metadata are expected under:

```text
META_DIR
```

Preprocessed 500 Hz ECG arrays are expected under:

```text
PROCESSED_DIR
```

The preprocessing script is:

```bash
python process_easy_500hz.py
```

Edit these paths inside `process_easy_500hz.py` before preprocessing:

```python
INPUT_DIR
OUTPUT_DIR
```

### Pretrained Weights

Backbone checkpoint paths are configured in `pipeline/train_config.py`.

Important variables:

| Model | Variable |
|---|---|
| ECGFounder | `ECGFOUNDER_1LEAD_CKPT` |
| ECG-JEPA | `ECG_JEPA_CKPT` |
| ECG-FM | `ECGFM_CKPT` |
| ST-MEM | `ST_MEM_CKPT` |
| Stage-1 best checkpoint | `TRAIN_BEST_CKPT` |
| Stage-1 resume checkpoint | `TRAIN_RESUME_CKPT` |

PowerShell example:

```powershell
$env:ECGFOUNDER_1LEAD_CKPT="D:\checkpoints\1_lead_ECGFounder.pth"
python main.py train
```

## Main Workflows

### 1. Preprocess ECG Records

```bash
python process_easy_500hz.py
```

This creates processed ECG arrays and index/cache files.

### 2. Train or Load the ECG Backbone

```bash
python main.py train
```

The main pipeline performs:

1. Fixed patient-level train/validation/test split.
2. Segment-level ECG backbone training or checkpoint loading.
3. Patient-level deep feature extraction.
4. Quantile pooling with `q = 0.24`.
5. PCA compression of deep ECG embeddings.
6. CatBoost evaluation of static, deep, and deep/static variants.
7. Paired patient-level bootstrap comparison between `static_only` and `deep_pca64_static`.

### 3. Run q Sensitivity

```bash
python run_pooling_sensitivity.py
```

Current behavior:

- Uses deep ECG and static covariates only.
- Selects q using fixed five-fold patient-level cross-validation on the train+validation cohort.
- Keeps the test set untouched until q is fixed.

### 4. Run PCA Sensitivity at q = 0.24

```bash
python run_pca_sensitivity_q024.py
```

Current behavior:

- Compares raw deep features and PCA dimensions.
- Evaluates `deep_only` and `deep_static`.

Default PCA dimensions:

```text
16, 32, 64, 128, 256
```

Override dimensions:

```powershell
$env:PCA_SENS_DIMS="8,16,32,64,128,256"
python run_pca_sensitivity_q024.py
```

### 5. Run SHAP Analysis

```bash
python run_shap_analysis.py
```

Current SHAP target:

```text
deep_pca64_static
```

The SHAP analysis uses deep PCA features plus static clinical covariates only.

### 6. Plot SHAP Summary Figure

```bash
python visualizations/make_shap_combined_from_csv.py
```

Expected input files:

```text
C:\Users\34007\OneDrive\Desktop\SHAP\shap_deep_pca64_static_q24_feature_importance.csv
C:\Users\34007\OneDrive\Desktop\SHAP\shap_deep_pca64_static_q24_per_sample_values.csv
```

Outputs:

```text
visualizations/shap_combined_top15_deep_pca64_static_q24_times.pdf
visualizations/shap_combined_top15_deep_pca64_static_q24_times.svg
visualizations/shap_combined_top15_deep_pca64_static_q24_times.png
```

## Project Structure

```text
D:/ECG_I_CARE
|-- main.py                         # Main training entry point
|-- process_easy_500hz.py            # ECG preprocessing and QC mask generation
|-- run_pooling_sensitivity.py       # q sensitivity, deep/static only
|-- run_pca_sensitivity_q024.py      # PCA sensitivity, q fixed at 0.24
|-- run_shap_analysis.py             # SHAP analysis for deep_pca64_static
|-- visualize_ecg_waveforms.py       # ECG waveform visualization helper
|-- visualizations/
|   `-- make_shap_combined_from_csv.py
|-- data/
|   |-- cache.py                     # Processed ECG cache and QC mask loading
|   |-- datasets.py                  # ICARE segment dataset and indexing
|   |-- splits.py                    # Patient-level train/val/test split
|   `-- train_metadata.py            # Metadata filtering utilities
|-- evaluation/
|   |-- metrics.py                   # AUROC, AUPRC, F1, sensitivity, specificity, CPC metrics
|   `-- reports.py                   # CSV writing and summary reporting
|-- feature/
|   `-- deep.py                      # Deep feature extraction, pooling, PCA
|-- model/
|   |-- ecgfounder.py                # ECGFounder adapter
|   |-- ecgfounder_net1d.py          # ECGFounder 1D model
|   |-- ecgfm_adapter.py             # ECG-FM adapter
|   |-- ecg_jepa_adapter.py          # ECG-JEPA adapter
|   |-- stmem_adapter.py             # ST-MEM adapter
|   |-- seresnet.py                  # SE-ResNet backbone
|   |-- convnext1d.py                # ConvNeXt1D backbone
|   `-- heads.py                     # Classifier/projector heads
|-- pipeline/
|   |-- train_config.py              # Central configuration
|   |-- args.py                      # CLI defaults
|   |-- train.py                     # Pipeline export
|   |-- train_core.py                # Main training orchestration
|   `-- experiments.py               # CatBoost design matrices and evaluation
`-- utils/
    |-- checkpoint.py                # Checkpoint helpers
    `-- runtime.py                   # GPU/CPU/memory helpers
```

## Key Configuration

Important settings live in:

```text
pipeline/train_config.py
```

| Setting | Meaning | Current default |
|---|---|---|
| `STAGE1_BACKBONE` | ECG backbone | `ecgfounder` |
| `WINDOW_SIZE` | 10-second ECG window at 500 Hz | `5000` |
| `STRIDE` | segment stride | `10000` |
| `MAX_OBSERVATION_HOUR` | observation window | `72.0` |
| `TEMPORAL_AGG_QUANTILE` | patient pooling q | `0.24` |
| `DEEP_PCA_DIM` | deep PCA bottleneck | `64` |
| `STAGE1_EPOCHS` | segment-level epochs | `25` |
| `STAGE1_MAX_SEGMENTS_PER_PATIENT_PER_EPOCH` | max sampled segments per patient per epoch | `256` |
| `CATBOOST_ITERATIONS` | CatBoost iterations | `100` |
| `CATBOOST_DEPTH` | CatBoost depth | `4` |
| `CATBOOST_LR` | CatBoost learning rate | `0.03` |
| `CLASS_WEIGHTS` | class weights | `[1.7, 1.0]` |
| `NUM_RUNS` | downstream CatBoost repeats | `5` |
| `BASE_SEED` | base seed | `42` |

## Outputs and Metrics

Typical outputs:

```text
checkpoint/*_best.pth
checkpoint/*_resume.pth
*_raw.csv
*_summary.csv
*_config.json
*_test_patient_probabilities_long.csv
*_static_vs_neuroecg_test_patient_probabilities.csv
*_static_vs_neuroecg_ensemble_probabilities.csv
*_static_vs_neuroecg_paired_bootstrap.csv
```

Metrics:

| Metric | Meaning |
|---|---|
| AUROC | Binary outcome discrimination |
| AUPRC | Precision-recall performance |
| Accuracy | Thresholded outcome accuracy |
| F1 | F1 score |
| Sensitivity | True positive rate |
| Specificity | True negative rate |
| CPC MAE | Mean absolute CPC regression error |

Manuscript table template:

| Feature set | Test AUROC | Test AUPRC | F1 | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Static only | fill from `*_summary.csv` | fill | fill | fill | fill |
| Deep ECG only | fill from `*_summary.csv` | fill | fill | fill | fill |
| Deep PCA64 only | fill from `*_summary.csv` | fill | fill | fill | fill |
| Deep PCA64 + static | fill from `*_summary.csv` | fill | fill | fill | fill |

For the incremental value of the deep/static model over static covariates alone, use:

```text
*_static_vs_neuroecg_paired_bootstrap.csv
```

This file reports ensemble AUROC confidence intervals and the paired bootstrap CI for:

```text
Delta AUROC = AUROC(deep_pca64_static) - AUROC(static_only)
```

## Reproducibility Notes

The patient split is deterministic when the patient cohort and seed are unchanged:

```python
BASE_SEED = 42
SPLIT_TRAIN_SIZE = 0.70
SPLIT_VAL_SIZE = 0.15
```

Stage-1 segment sampling is deterministic for a given seed:

- Each epoch samples at most 256 segments per patient.
- Patients with 256 or fewer valid segments use all available segments.
- Patients with more than 256 valid segments are sampled without replacement.
- The epoch seed is `BASE_SEED + epoch`.

For q selection, `run_pooling_sensitivity.py` uses fixed five-fold patient-level cross-validation on the combined train+validation cohort. The held-out test set is used only after q is fixed.

PCA is fitted only on the training split and then applied to validation/test features to avoid leakage.

If only one Stage-1 best checkpoint is used, repeated downstream runs reflect CatBoost/random downstream variability, not full repeated backbone training.

## Common Issues

### Dataset paths do not exist

The default paths may point to a server. Update:

```python
META_DIR
PROCESSED_DIR
RECORD_CACHE
```

in `pipeline/train_config.py`.

### Checkpoint not found

Check:

```python
ECGFOUNDER_1LEAD_CKPT
TRAIN_BEST_CKPT
TRAIN_RESUME_CKPT
```

### CatBoost GPU failure

If CatBoost GPU mode fails, run CatBoost on CPU or adjust GPU settings in:

```text
utils/runtime.py
```

### CUDA / PyTorch mismatch

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### High memory use

Reduce:

```python
BATCH_SIZE
STAGE1_SEGMENT_TRAIN_BATCH_SIZE
```

## Citation and Acknowledgement

Please cite the original resources used in your experiments:

- I-CARE / CinC 2023 cardiac arrest outcome prediction dataset.
- ECGFounder if using the ECGFounder backbone.
- ECG-FM, ECG-JEPA, ST-MEM, SE-ResNet, or ConvNeXt1D if using those baselines.
- CatBoost for downstream patient-level classification/regression.

## License

No project-specific license file was found. Until a license is added, treat this repository as research code with all rights reserved by the author. Add a `LICENSE` file before public redistribution.
