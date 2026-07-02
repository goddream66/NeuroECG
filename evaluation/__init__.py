"""
Evaluation package: metrics, reporting, and feature engineering utilities.

Sub-packages:
  - metrics.py           : Segment-level and patient-level evaluation metrics
  - reports.py           : Training summary printing and CSV writing
  - feature_engineering/ : Auxiliary feature extractors isolated from the main
                           deep backbone training pipeline:
                             * statement71.py  (PTB-XL 71-class semantic features)
                             * hrv.py          (Time-domain HRV features)
"""
