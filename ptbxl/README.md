# PTB-XL statement71 bridge

The full `ecg_ptbxl_benchmarking/` repository is no longer required by the
main ECG I-CARE pipeline. The statement71 model code used by the pipeline has
been extracted into:

```text
D:\ECG_I_CARE\model\ptbxl_xresnet1d.py
D:\ECG_I_CARE\model\ptbxl_basic_conv1d.py
```

The default model weight is centralized under:

```text
D:\ECG_I_CARE\checkpoint\fastai_xresnet1d101.pth
```

The PTB-XL statement71 auxiliary files are kept under:

```text
D:\ECG_I_CARE\ptbxl\resources\mlb.pkl
D:\ECG_I_CARE\ptbxl\resources\standard_scaler.pkl
```

You can override them on the server without editing code:

```bash
export PTBXL_PTH_PATH=/path/to/fastai_xresnet1d101.pth
export PTBXL_MLB_PATH=/path/to/mlb.pkl
export PTBXL_STANDARD_SCALER_PATH=/path/to/standard_scaler.pkl
```
