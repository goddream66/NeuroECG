# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from utils.runtime import _gpu_clean_feature_tensor, _gpu_to_numpy

def evaluate_segment_metrics(
    extractor,
    projector,
    classifier_head,
    data_loader,
    criterion,
    device,
    amp_enabled,
):
    """Evaluate segment-level classifier with patient labels assigned to segments.

    Every ECG segment is treated as one sample, and the parent patient's
    Good/Poor outcome is used as the segment label. No entry-level or
    patient-level MIL aggregation is applied here.
    """
    extractor.eval()
    projector.eval()
    classifier_head.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for ecg, channel_mask, labels, _, _, _, _, _, _ in data_loader:
            ecg = ecg.to(device).float()
            channel_mask = channel_mask.to(device).float()
            labels = labels.to(device).float()

            with autocast(enabled=amp_enabled):
                logits = classifier_head(projector(extractor(ecg, channel_mask=channel_mask)))
                batch_loss = criterion(logits, labels)

            total_loss += float(batch_loss.item())
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    avg_loss = total_loss / max(len(data_loader), 1)
    if not all_labels:
        return avg_loss, float("nan"), float("nan")

    labels_np = np.concatenate(all_labels).astype(int)
    probs_np = np.concatenate(all_probs).astype(float)

    if len(np.unique(labels_np)) >= 2:
        val_auroc = float(roc_auc_score(labels_np, probs_np))
        val_auprc = float(average_precision_score(labels_np, probs_np))
    else:
        val_auroc = float("nan")
        val_auprc = float("nan")

    return avg_loss, val_auroc, val_auprc


def compute_feature_quantile(feature_matrix: np.ndarray, quantile: float) -> np.ndarray:
    feature_tensor = _gpu_clean_feature_tensor(feature_matrix)
    if feature_tensor.ndim != 2 or feature_tensor.shape[0] == 0:
        raise ValueError("feature_matrix must be a non-empty 2D array")
    out = torch.quantile(feature_tensor, float(quantile), dim=0)
    result = _gpu_to_numpy(out)
    del feature_tensor, out
    return result


def safe_divide(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_challenge_score(y_true, y_prob, hospitals, max_fpr=0.05):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    hospitals = np.asarray(hospitals)

    total_tp = 0
    total_fn = 0

    for hospital in np.unique(hospitals):
        hospital_mask = hospitals == hospital
        hospital_true = y_true[hospital_mask]
        hospital_prob = y_prob[hospital_mask]

        negative_scores = hospital_prob[hospital_true == 0]
        if negative_scores.size == 0:
            threshold = -np.inf
        else:
            sorted_negatives = np.sort(negative_scores)[::-1]
            max_fp = int(np.floor(max_fpr * negative_scores.size))
            if max_fp <= 0:
                threshold = sorted_negatives[0] + 1e-12
            elif max_fp >= negative_scores.size:
                threshold = -np.inf
            else:
                threshold = sorted_negatives[max_fp - 1]

        predicted_positive = hospital_prob >= threshold
        total_tp += int(np.sum((hospital_true == 1) & predicted_positive))
        total_fn += int(np.sum((hospital_true == 1) & (~predicted_positive)))

    return safe_divide(total_tp, total_tp + total_fn)


def compute_binary_operating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = safe_divide(tp, tp + fn)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tpr": recall,
        "recall": recall,
        "sensitivity": recall,
        "specificity": safe_divide(tn, tn + fp),
        "fpr": safe_divide(fp, fp + tn),
    }


def compute_challenge_score_safe(y_true: np.ndarray, y_prob: np.ndarray, patient_ids: Sequence[str], pid_meta: Dict[str, dict]) -> float:
    hospitals = [pid_meta[str(pid)]["hospital"] for pid in patient_ids]
    return compute_challenge_score(y_true, y_prob, hospitals)


def evaluate_model(cls: CatBoostClassifier, reg: CatBoostRegressor, X: np.ndarray, y: np.ndarray, cpc: np.ndarray, patient_ids: Sequence[str], pid_meta: Dict[str, dict]) -> Dict[str, float]:
    probs = cls.predict_proba(X)[:, 1]
    preds = cls.predict(X).reshape(-1).astype(int)
    cpc_preds = reg.predict(X)
    out = {"n": int(len(y))}
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, probs))
        out["auprc"] = float(average_precision_score(y, probs))
        out["challenge_score"] = float(compute_challenge_score_safe(y, probs, patient_ids, pid_meta))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
        out["challenge_score"] = np.nan
    out["accuracy"] = float(accuracy_score(y, preds))
    out["f1"] = float(f1_score(y, preds, zero_division=0))
    out.update(compute_binary_operating_metrics(y, preds))
    out["cpc_mse"] = float(mean_squared_error(cpc, cpc_preds))
    out["cpc_mae"] = float(mean_absolute_error(cpc, cpc_preds))
    return out
