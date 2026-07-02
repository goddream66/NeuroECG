# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from evaluation.metrics import compute_binary_operating_metrics, compute_challenge_score

def report(model_cls, model_reg, X, y_outcome, y_cpc, patient_ids, pid_meta, name):
    probs = model_cls.predict_proba(X)[:, 1]
    preds = model_cls.predict(X).reshape(-1)
    cpc_preds = model_reg.predict(X)
    hospitals = [pid_meta[pid]["hospital"] for pid in patient_ids]
    challenge_score = compute_challenge_score(y_outcome, probs, hospitals)
    accuracy = accuracy_score(y_outcome, preds)
    f_measure = f1_score(y_outcome, preds, zero_division=0)
    cpc_mse = mean_squared_error(y_cpc, cpc_preds)
    cpc_mae = mean_absolute_error(y_cpc, cpc_preds)
    operating_metrics = compute_binary_operating_metrics(y_outcome, preds)

    print(f"\n--- {name} Report ---")
    print(f"  Challenge Score: {challenge_score:.4f}")
    print(f"  Outcome AUROC: {roc_auc_score(y_outcome, probs):.4f}")
    print(f"  Outcome AUPRC: {average_precision_score(y_outcome, probs):.4f}")
    print(f"  Outcome Accuracy: {accuracy:.4f}")
    print(f"  Outcome F-measure: {f_measure:.4f}")
    print(f"  Outcome TPR: {operating_metrics['tpr']:.4f}")
    print(f"  Outcome FPR: {operating_metrics['fpr']:.4f}")
    print(f"  Outcome Sensitivity: {operating_metrics['sensitivity']:.4f}")
    print(f"  Outcome Specificity: {operating_metrics['specificity']:.4f}")
    print(
        "  Confusion Matrix: "
        f"TN={operating_metrics['tn']} FP={operating_metrics['fp']} "
        f"FN={operating_metrics['fn']} TP={operating_metrics['tp']}"
    )
    print(f"  CPC MSE: {cpc_mse:.4f}")
    print(f"  CPC MAE: {cpc_mae:.4f}")


def report_if_available(model_cls, model_reg, X, y_outcome, y_cpc, patient_ids, pid_meta, name):
    if len(X) == 0:
        print(f"\n--- {name} Report ---")
        print("  Skipped: no patients available for this subset")
        return
    if len(np.unique(y_outcome)) < 2:
        print(f"\n--- {name} Report ---")
        print("  Skipped: fewer than two outcome classes in this subset")
        return
    report(model_cls, model_reg, X, y_outcome, y_cpc, patient_ids, pid_meta, name)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    metric_order = [
        "auroc", "auprc", "accuracy", "f1", "recall",
        "sensitivity", "specificity", "fpr", "challenge_score", "cpc_mae",
    ]
    split_order = ["train", "val", "test"]
    preferred = ["variant", "representation", "seed", "runs", "feature_dim", "train_n", "val_n", "test_n"]
    preferred.extend(f"{split}_{metric}" for split in split_order for metric in metric_order)
    preferred.extend(
        f"{split}_{metric}_{stat}"
        for split in split_order
        for metric in metric_order
        for stat in ("mean", "std")
    )
    keys = sorted({key for row in rows for key in row.keys()})
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_keys = [
        "train_auroc", "train_auprc", "train_accuracy", "train_f1", "train_recall",
        "train_sensitivity", "train_specificity", "train_fpr", "train_challenge_score",
        "train_cpc_mae",
        "val_auroc", "val_auprc", "val_accuracy", "val_f1", "val_recall",
        "val_sensitivity", "val_specificity", "val_fpr", "val_challenge_score",
        "val_cpc_mae",
        "test_auroc", "test_auprc", "test_accuracy", "test_f1", "test_recall",
        "test_sensitivity", "test_specificity", "test_fpr", "test_challenge_score",
        "test_cpc_mae",
    ]
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant"]), str(row["representation"])), []).append(row)

    summary_rows = []
    for (variant, representation), group in grouped.items():
        out = {
            "variant": variant,
            "representation": representation,
            "runs": len(group),
            "feature_dim": int(group[0]["feature_dim"]),
            "train_n": int(group[0]["train_n"]),
            "val_n": int(group[0]["val_n"]),
            "test_n": int(group[0]["test_n"]),
        }
        for key in metric_keys:
            vals = np.asarray([float(row.get(key, np.nan)) for row in group], dtype=float)
            out[f"{key}_mean"] = float(np.nanmean(vals))
            out[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def print_key_summary(summary_rows: List[Dict[str, object]]) -> None:
    by_key = {(str(row["variant"]), str(row["representation"])): row for row in summary_rows}
    def get(variant: str, rep: str, metric: str) -> float:
        return float(by_key.get((variant, rep), {}).get(metric, np.nan))
    static_auroc = get("static_only", "static_patient_level", "test_auroc_mean")
    hrv_static_auroc = get("hrv_static", f"static_plus_hrv_{PATIENT_POOLING_TAG}", "test_auroc_mean")
    semantic_static_auroc = get("statement71_static", "static_plus_statement71", "test_auroc_mean")
    semantic_hrv_static_auroc = get("statement71_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_statement71", "test_auroc_mean")
    semantic_rep = f"{SEMANTIC_FEATURE_NAME}_wvnum{WVNUM}_{ENTRY_STATEMENT_AGG_MODE}_patient_{PATIENT_ENTRY_AGG_MODE}"
    print("\n=== PTBXL71 statement semantic key comparisons ===")
    print(f"static_only              : AUROC={static_auroc:.4f}, AUPRC={get('static_only', 'static_patient_level', 'test_auprc_mean'):.4f}")
    print(f"hrv_only                 : AUROC={get('hrv_only', f'hrv_{PATIENT_POOLING_TAG}', 'test_auroc_mean'):.4f}, AUPRC={get('hrv_only', f'hrv_{PATIENT_POOLING_TAG}', 'test_auprc_mean'):.4f}")
    print(f"statement71_only         : AUROC={get('statement71_only', semantic_rep, 'test_auroc_mean'):.4f}, AUPRC={get('statement71_only', semantic_rep, 'test_auprc_mean'):.4f}")
    print(f"hrv_static               : AUROC={hrv_static_auroc:.4f}, dAUROC_vs_static={hrv_static_auroc - static_auroc:+.4f}")
    print(f"statement71_static       : AUROC={semantic_static_auroc:.4f}, dAUROC_vs_static={semantic_static_auroc - static_auroc:+.4f}")
    print(f"statement71_hrv_static   : AUROC={semantic_hrv_static_auroc:.4f}, dAUROC_vs_hrv_static={semantic_hrv_static_auroc - hrv_static_auroc:+.4f}")


def print_training_key_summary(summary_rows: List[Dict[str, object]]) -> None:
    by_key = {(str(row["variant"]), str(row["representation"])): row for row in summary_rows}
    key_order = [
        ("static_only", "static_patient_level"),
        ("hrv_only", f"hrv_{PATIENT_POOLING_TAG}"),
        ("statement71_only", f"{SEMANTIC_FEATURE_NAME}_wvnum{WVNUM}_{ENTRY_STATEMENT_AGG_MODE}_patient_{PATIENT_ENTRY_AGG_MODE}"),
        ("deep_only", f"deep_patient_{PATIENT_POOLING_TAG}"),
        ("deep_statement71_only", "deep_statement71"),
        ("deep_pca64_only", f"deep_pca64_patient_{PATIENT_POOLING_TAG}"),
        ("deep_pca64_static", f"static_plus_deep_pca64_{PATIENT_POOLING_TAG}"),
        ("deep_pca64_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_deep_pca64"),
        ("deep_pca64_statement71_static", f"static_deep_pca64_{PATIENT_POOLING_TAG}_statement71"),
        ("deep_pca64_statement71_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_deep_pca64_statement71"),
        ("deep_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_deep"),
        ("statement71_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_statement71"),
        ("deep_statement71_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_deep_statement71"),
        ("gated_deep_static", f"deep_static_projected{FUSION_ALIGN_DIM}_softmax_gated"),
        (
            "gated_deep_static_hrv_statement71",
            f"deep_static_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_hrv_statement71",
        ),
        (
            "gated_deep_statement71_static_hrv",
            f"deep_statement71_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_static_hrv",
        ),
        (
            "fixedgate_deep_static",
            f"deep_static_projected{FUSION_ALIGN_DIM}_fixedgate_d{FIXED_GATE_DEEP_WEIGHT:.3f}_s{FIXED_GATE_STATIC_WEIGHT:.3f}",
        ),
        (
            "fixedgate_deep_static_hrv_statement71",
            f"deep_static_projected{FUSION_ALIGN_DIM}_fixedgate_d{FIXED_GATE_DEEP_WEIGHT:.3f}_s{FIXED_GATE_STATIC_WEIGHT:.3f}_plus_hrv_statement71",
        ),
        (
            "clinical_residual_deep_pca64",
            "clinical_static_hrv_statement71_plus_patient_residual_deep_pca64",
        ),
        (
            "clinical_residual_deep1024",
            "clinical_static_hrv_statement71_plus_patient_residual_deep1024",
        ),
    ]

    print("\n=== PTBXL71 training/fusion key comparisons ===")
    for variant, rep in key_order:
        row = by_key.get((variant, rep))
        if row is None:
            continue
        print(
            f"{variant:37s}: "
            f"train AUROC={float(row['train_auroc_mean']):.4f}, "
            f"AUPRC={float(row['train_auprc_mean']):.4f}, "
            f"F1={float(row['train_f1_mean']):.4f}, "
            f"Recall={float(row['train_recall_mean']):.4f}, "
            f"Spec={float(row['train_specificity_mean']):.4f} | "
            f"val AUROC={float(row['val_auroc_mean']):.4f}, "
            f"AUPRC={float(row['val_auprc_mean']):.4f}, "
            f"F1={float(row['val_f1_mean']):.4f}, "
            f"Recall={float(row['val_recall_mean']):.4f}, "
            f"Spec={float(row['val_specificity_mean']):.4f} | "
            f"test AUROC={float(row['test_auroc_mean']):.4f}, "
            f"AUPRC={float(row['test_auprc_mean']):.4f}, "
            f"F1={float(row['test_f1_mean']):.4f}, "
            f"Recall={float(row['test_recall_mean']):.4f}, "
            f"Spec={float(row['test_specificity_mean']):.4f}"
        )
