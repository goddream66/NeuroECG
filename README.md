# NeuroECG: ECGFounder-Based Deep ECG Representation for EEG-Free Neurological Prognostication from Bedside ECG

[![Conference](https://img.shields.io/badge/BIBM-2026-brightgreen.svg?style=flat-square)](https://ieeebibm.org/)

This is the official PyTorch implementation of **NeuroECG**, an EEG-free deep learning framework for neurological prognostication after cardiac arrest using bedside monitoring ECG and clinical covariates.

---

## 📌 Overview

`NeuroECG` is a modular framework for **neurological outcome prediction in comatose patients after cardiac arrest without relying on EEG signals**.

<p align="center">
  <img src="docs/BIBM_framework.svg" alt="NeuroECG Framework" width="85%">
</p>

<p align="center">
  <em>Figure 1: Overall architecture of the proposed NeuroECG framework.</em>
</p>

The framework adapts a pre-trained ECG foundation model to bedside ECG recordings and integrates patient-level deep ECG representations with routinely available clinical covariates.

### Key Features

* EEG-free neurological prognostication from bedside ECG.
* ECGFounder adaptation for post-cardiac-arrest ECG analysis.
* Feature-wise quantile pooling for patient-level ECG aggregation.
* Compact deep ECG representations using PCA compression.
* Modular fusion of ECG representations and clinical covariates.
* CatBoost-based patient-level neurological outcome prediction.

---

## 📊 Dataset and Pre-trained Model

### I-CARE Dataset

Experiments are conducted using the **International Cardiac Arrest REsearch consortium (I-CARE) Database**.

The dataset is available from PhysioNet:

https://physionet.org/content/i-care/

Please follow the PhysioNet data access requirements and data use agreement before downloading the dataset.

### ECGFounder

NeuroECG uses **ECGFounder** as the pre-trained ECG backbone.

The ECGFounder implementation and pre-trained model are available at:

https://github.com/PKUDigitalHealth/ECGFounder

---

## ⚙️ NeuroECG Framework

NeuroECG consists of two main stages.

### Stage I: ECG Representation Learning

Single-lead bedside ECG segments are used to adapt the pre-trained ECGFounder backbone to neurological outcome prediction.

After adaptation, the segment-level prediction head is removed, and the ECGFounder backbone is used to extract 1024-dimensional deep ECG representations.

### Stage II: Patient-Level Prognostication

Segment-level ECG representations are aggregated using feature-wise quantile pooling.

The patient-level ECG representation is compressed from 1024 to 64 dimensions using PCA and concatenated with six static clinical covariates.

The resulting 70-dimensional fused representation is classified using CatBoost.

The primary NeuroECG configuration is:

```text
Quantile pooling: q = 0.24
Deep ECG representation: 1024 dimensions
PCA-compressed representation: 64 dimensions
Static clinical covariates: 6
Fused representation: 70 dimensions
Classifier: CatBoost
```

---

## 🚀 Quick Start

### 1. Obtain the Source Code

The anonymous source code for double-blind review is available at:

https://anonymous.4open.science/r/NeuroECG-2DE2/

Enter the project directory:

```bash
cd NeuroECG
```

### 2. Create the Environment

```bash
conda create -n neuroecg python=3.10
conda activate neuroecg
pip install -r requirements.txt
```

### 3. Prepare the ECG Data

Configure the local I-CARE dataset path and run the ECG preprocessing script:

```bash
python process_easy_500hz.py
```

The generated ECG cache is used by the downstream NeuroECG pipeline.

### 4. Run NeuroECG

Check the available experiment arguments:

```bash
python main.py --help
```

Run the NeuroECG pipeline:

```bash
python main.py
```

Experiment arguments and training configurations are defined under `pipeline/`.

---

## 📈 Evaluation

NeuroECG reports the following patient-level evaluation metrics:

* AUROC
* AUPRC
* Accuracy
* F1 Score
* Recall / Sensitivity
* Specificity
* CPC-MAE

---

## 📂 Repository Structure

```text
NeuroECG/
├── main.py                  # Main experiment entry point
├── process_easy_500hz.py    # ECG preprocessing pipeline
├── requirements.txt         # Python environment dependencies
├── README.md
│
├── data/
│   ├── cache.py             # Processed data cache utilities
│   ├── datasets.py          # Dataset definitions and data loading
│   ├── splits.py            # Patient-level data splitting
│   ├── train_metadata.py    # Clinical metadata processing
│   └── __init__.py
│
├── docs/
│   └── BIBM_framework.svg   # NeuroECG framework figure
│
├── evaluation/
│   ├── metrics.py           # Evaluation metrics
│   ├── reports.py           # Experiment reporting
│   ├── feature_engineering/
│   │   ├── hrv.py           # Additional HRV feature utilities
│   │   └── __init__.py
│   └── __init__.py
│
├── feature/
│   ├── deep.py              # Deep ECG representation extraction
│   └── __init__.py
│
├── model/
│   ├── ecgfounder.py        # ECGFounder model interface
│   ├── ecgfounder_net1d.py  # ECGFounder 1-D network implementation
│   ├── heads.py             # Prediction heads
│   └── __init__.py
│
├── pipeline/
│   ├── args.py              # Command-line argument definitions
│   ├── experiments.py       # Experiment management
│   ├── train.py             # Model training pipeline
│   ├── train_config.py      # Training configuration
│   ├── train_core.py        # Core training procedures
│   └── __init__.py
│
└── utils/
    ├── checkpoint.py        # Checkpoint utilities
    ├── runtime.py           # Runtime utilities
    └── __init__.py
```

`__pycache__` directories and generated ECG cache files are omitted for clarity.

---

## 🔄 Reproducibility

All dataset partitions are defined at the patient level to prevent ECG segments from the same patient from appearing in different partitions.

Patient-level separation is maintained throughout ECGFounder adaptation, deep representation extraction, patient-level aggregation, PCA fitting, and downstream classification.

The primary experiments are repeated using five random seeds:

```text
42, 43, 44, 45, 46
```

---

## 🙏 Acknowledgements

We thank the contributors of the I-CARE Database and the PhysioNet community for providing the physiological recordings and clinical data used in this study.

We also thank the ECGFounder authors for releasing the pre-trained ECG foundation model and related resources.
