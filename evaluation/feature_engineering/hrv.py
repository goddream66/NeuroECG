# ─────────────────────────────────────────────────────────────────────────────
# HRV (Heart Rate Variability) feature computation.
#
# Canonical location: evaluation/feature_engineering/hrv.py
# The original feature/hrv.py is now a backward-compatible re-export shim
# that points here, so all existing run_*.py imports continue to work.
# ─────────────────────────────────────────────────────────────────────────────
# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *

def _detect_r_peaks(ecg_lead, fs):
    ecg_lead = np.asarray(ecg_lead, dtype=np.float32)
    if ecg_lead.size < max(int(fs), 8):
        return np.array([], dtype=np.int64)

    centered = ecg_lead - np.median(ecg_lead)
    scaled = centered / (np.std(centered) + 1e-6)
    min_distance = max(int(0.25 * fs), 1)
    prominence = max(0.3, 0.15 * np.std(scaled))

    peaks_pos, _ = signal.find_peaks(
        scaled,
        distance=min_distance,
        prominence=prominence,
    )
    peaks_neg, _ = signal.find_peaks(
        -scaled,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks_neg) > len(peaks_pos):
        return peaks_neg.astype(np.int64)
    return peaks_pos.astype(np.int64)


def _compute_hrv_features_from_lead(ecg_lead, fs=TARGET_FS):
    peaks = _detect_r_peaks(ecg_lead, fs)
    if len(peaks) < 3:
        return np.zeros(len(HRV_FEATURE_NAMES), dtype=np.float32), 0

    rr_ms = np.diff(peaks).astype(np.float32) / float(fs) * 1000.0
    rr_ms = rr_ms[(rr_ms > 250.0) & (rr_ms < 2500.0)]
    if len(rr_ms) < 2:
        return np.zeros(len(HRV_FEATURE_NAMES), dtype=np.float32), int(len(peaks))

    diff_rr = np.diff(rr_ms)
    if len(diff_rr) == 0:
        return np.zeros(len(HRV_FEATURE_NAMES), dtype=np.float32), int(len(peaks))

    nn50_count = float(np.sum(np.abs(diff_rr) > 50.0))
    pnn50 = nn50_count / float(len(diff_rr))
    rmssd = float(np.sqrt(np.mean(np.square(diff_rr))))

    sd_diff = float(np.std(diff_rr, ddof=0))
    sd_rr = float(np.std(rr_ms, ddof=0))
    sd1 = float(np.sqrt(max(0.5 * np.var(diff_rr, ddof=0), 0.0)))
    sd2_sq = max(2.0 * (sd_rr ** 2) - 0.5 * (sd_diff ** 2), 0.0)
    sd2 = float(np.sqrt(sd2_sq))
    sd1_sd2_ratio = sd1 / (sd2 + 1e-6)
    sd2_sd1_ratio = sd2 / (sd1 + 1e-6)
    csi = sd2_sd1_ratio
    cvi = float(np.log10((sd1 * sd2) + 1e-6))
    rr_diff_std_mean_ratio = float(
        np.std(diff_rr, ddof=0) / (np.mean(np.abs(diff_rr)) + 1e-6)
    )

    return np.asarray(
        [
            nn50_count,
            pnn50,
            rmssd,
            sd1,
            sd2,
            sd1_sd2_ratio,
            sd2_sd1_ratio,
            csi,
            cvi,
            rr_diff_std_mean_ratio,
        ],
        dtype=np.float32,
    ), int(len(peaks))


def _compute_best_channel_hrv_features(ecg_mat, fs=TARGET_FS, channel_mask=None):
    """Compute HRV from the best valid ECG channel.

    Old code always used channel 0.  For ECG1+ECG2 or ECGL+ECGR, channel 1 may
    contain clearer R peaks.  This version computes candidate HRV on valid
    channels and keeps the one with more detected R peaks.  For a copied
    single-channel record, channel_mask=[1,0] makes only the real channel valid.
    """
    ecg_mat = np.asarray(ecg_mat, dtype=np.float32)
    if ecg_mat.ndim == 1:
        ecg_mat = ecg_mat[np.newaxis, :]

    if channel_mask is None:
        valid_indices = list(range(min(ecg_mat.shape[0], 2)))
    else:
        mask = np.asarray(channel_mask, dtype=np.float32).reshape(-1)
        valid_indices = [i for i in range(min(ecg_mat.shape[0], len(mask))) if mask[i] > 0.5]
        if not valid_indices:
            valid_indices = [0]

    best_features = np.zeros(len(HRV_FEATURE_NAMES), dtype=np.float32)
    best_peak_count = -1
    for ch in valid_indices:
        features, peak_count = _compute_hrv_features_from_lead(ecg_mat[ch], fs=fs)
        if peak_count > best_peak_count:
            best_features = features
            best_peak_count = peak_count

    return best_features, int(best_peak_count)


def compute_hrv_features(ecg_mat, fs=TARGET_FS, channel_mask=None):
    features, _ = _compute_best_channel_hrv_features(ecg_mat, fs=fs, channel_mask=channel_mask)
    return features
