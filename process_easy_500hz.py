import json
import os
import re
import pickle
from math import gcd
from multiprocessing import Pool

import numpy as np
import scipy.signal as signal
import wfdb
from tqdm import tqdm


INPUT_DIR  = "/data/xcy_group/gjj/cinc2023/cinc2023_dataset/training/"
OUTPUT_DIR = "/data/xcy_group/gjj/cinc2023/cinc2023_dataset/processed_npy_500hz_qc_mask_flatline_only/"
LOG_FILE   = "error_log_qc_mask_500hz_flatline_only.txt"
PROCESSES  = 3

TARGET_FS         = 500
LOWCUT_HZ         = 0.5
HIGHCUT_HZ        = 40.0
CLIP_VALUE        = 5.0
SAVE_METADATA     = True
TIME_BUCKETS_HOURS = (12, 24, 48, 72)
INDEX_JSON        = "processed_index.json"
INDEX_PKL         = "processed_index.pkl"


QC_MIN_STD        = 0.05
QC_MAX_CLIP_RATIO = 0.30
QC_HF_RATIO_MAX   = 8.0
QC_MIN_HR         = 20
QC_MAX_HR         = 300


QC_MIN_QRS_SNR   = 1.8
QC_MAX_RR_CV     = 0.80


QC_WINDOW_SIZE    = 5000
QC_STRIDE         = 10000
QC_MIN_GOOD_RATIO = 0.05


ENABLE_SEGMENT_POLARITY_NORMALIZATION = False
SEG_POLARITY_MIN_PEAKS = 3
SEG_POLARITY_FLIP_MARGIN = 1.05


def normalize_lead_name(name):
    return "".join(ch for ch in str(name).upper() if ch.isalnum())

def parse_record_time_info(record_stem):
    record_name = os.path.basename(record_stem)
    match = re.match(
        r"^(?P<pid>\d+)_(?P<segment>\d+)_(?P<end_hour>\d+)_(?P<kind>[A-Za-z]+)$",
        record_name,
    )
    info = {
        "record": record_name,
        "patient_id": None,
        "segment_index": None,
        "segment_end_hour": None,
        "time_bucket": "unknown",
        "time_bucket_upper_hour": None,
    }
    if not match:
        return info

    segment_index = int(match.group("segment"))
    end_hour      = int(match.group("end_hour"))
    patient_id    = match.group("pid")

    time_bucket           = f">{TIME_BUCKETS_HOURS[-1]}h"
    time_bucket_upper_hour = None
    for hour_limit in TIME_BUCKETS_HOURS:
        if end_hour <= hour_limit:
            time_bucket = (
                f"0-{hour_limit}h"
                if hour_limit == TIME_BUCKETS_HOURS[0]
                else f"{TIME_BUCKETS_HOURS[TIME_BUCKETS_HOURS.index(hour_limit)-1]}-{hour_limit}h"
            )
            time_bucket_upper_hour = hour_limit
            break
    if time_bucket_upper_hour is None:
        time_bucket_upper_hour = end_hour

    info.update({
        "patient_id":            patient_id,
        "segment_index":         segment_index,
        "segment_end_hour":      end_hour,
        "time_bucket":           time_bucket,
        "time_bucket_upper_hour": time_bucket_upper_hour,
    })
    return info

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def extract_header_time_fields(record_path):
    header         = wfdb.rdheader(record_path)
    base_time      = getattr(header, "base_time",  None)
    base_date      = getattr(header, "base_date",  None)
    sig_len        = getattr(header, "sig_len",    None)
    fs             = getattr(header, "fs",         None)
    comments       = list(getattr(header, "comments", []) or [])
    duration_seconds = None
    if sig_len is not None and fs:
        duration_seconds = float(sig_len) / float(fs)
    return {
        "header_base_time":       str(base_time) if base_time is not None else None,
        "header_base_date":       str(base_date) if base_date is not None else None,
        "header_duration_seconds": duration_seconds,
        "header_comments":        [str(c) for c in comments],
    }


def choose_channel_indices(sig_names):
    normalized = [normalize_lead_name(name) for name in sig_names]

    name_to_idx = {}
    for idx, name in enumerate(normalized):
        if name not in name_to_idx:
            name_to_idx[name] = idx

    if "ECG1" in name_to_idx and "ECG2" in name_to_idx:
        return [name_to_idx["ECG1"], name_to_idx["ECG2"]]

    if "ECGL" in name_to_idx and "ECGR" in name_to_idx:
        return [name_to_idx["ECGL"], name_to_idx["ECGR"]]

    for single_name in ("ECG", "ECG1", "ECG2", "ECGL", "ECGR"):
        if single_name in name_to_idx:
            return [name_to_idx[single_name]]

    standard_priority = [
        {"II", "ECGII", "MLII", "LEADII"},
        {"I",  "ECGI",  "MLI",  "LEADI"},
        {"V1"}, {"V2"}, {"V3"}, {"V4"}, {"V5"}, {"V6"},
        {"III", "LEADIII"},
        {"AVR"}, {"AVL"}, {"AVF"},
    ]
    selected = []
    for aliases in standard_priority:
        for idx, lead_name in enumerate(normalized):
            if lead_name in aliases and idx not in selected:
                selected.append(idx)
                break
        if len(selected) == 2:
            return selected

    for idx in range(len(sig_names)):
        if idx not in selected:
            selected.append(idx)
        if len(selected) == 2:
            break
    return selected


def infer_channel_metadata(selected_names):
    norm = [normalize_lead_name(n) for n in selected_names]
    if len(norm) == 0:
        return {"channel_type": "none", "channel_mask": [0, 0], "effective_input_channels": 2}

    if len(norm) == 1:
        return {"channel_type": f"single_{norm[0]}", "channel_mask": [1, 0], "effective_input_channels": 2}

    pair = norm[:2]
    if pair == ["ECG1", "ECG2"]:
        channel_type = "pair_ECG1_ECG2"
    elif pair == ["ECGL", "ECGR"]:
        channel_type = "pair_ECGL_ECGR"
    elif pair == ["ECG", "ECG"]:
        channel_type = "single_ECG_duplicated"
    else:
        channel_type = "pair_" + "_".join(pair)

    return {"channel_type": channel_type, "channel_mask": [1, 1], "effective_input_channels": 2}


def extract_hospital_from_patient_txt(record_stem):
    time_info = parse_record_time_info(record_stem)
    pid = time_info.get("patient_id")
    if not pid:
        return "Unknown"

    txt_path = os.path.join(INPUT_DIR, pid, f"{pid}.txt")
    if not os.path.exists(txt_path):
        return "Unknown"

    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                if k.strip().lower() == "hospital":
                    value = v.strip().upper()
                    return value if value in {"A", "B", "C", "D", "E", "F"} else value
    except OSError:
        return "Unknown"
    return "Unknown"

def bandpass_filter(mat, fs, lowcut_hz=LOWCUT_HZ, highcut_hz=HIGHCUT_HZ):
    nyquist = fs * 0.5
    if nyquist <= 0:
        raise ValueError(f"Invalid sampling rate: {fs}")
    low  = max(lowcut_hz  / nyquist, 1e-5)
    high = min(highcut_hz / nyquist, 0.99)
    if low >= high:
        raise ValueError(f"Invalid bandpass range for fs={fs}: low={lowcut_hz}Hz high={highcut_hz}Hz")
    sos = signal.butter(3, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, mat, axis=0)

def resample_to_target_fs(mat, original_fs, target_fs=TARGET_FS):
    if int(round(original_fs)) == int(round(target_fs)):
        return mat.astype(np.float32)
    up   = int(round(target_fs))
    down = int(round(original_fs))
    scale = gcd(up, down)
    up   //= scale
    down //= scale
    return signal.resample_poly(mat, up=up, down=down, axis=0).astype(np.float32)

def robust_normalize(mat, clip_value=CLIP_VALUE):
    centered   = mat - np.median(mat, axis=0, keepdims=True)
    scale      = np.percentile(np.abs(centered), 95, axis=0, keepdims=True)
    scale      = np.maximum(scale, 1e-6)
    normalized = centered / scale
    return np.clip(normalized, -clip_value, clip_value).astype(np.float32)


def normalize_polarity(mat):
    """Normalize signal polarity independently for each channel.

    Flip a channel when negative peaks dominate positive peaks.
    The input and output shape is [time, channels].
    """
    min_dist = int(0.25 * TARGET_FS)
    for ch in range(mat.shape[1]):
        sig = mat[:, ch]
        peaks_pos, _ = signal.find_peaks( sig, distance=min_dist, prominence=0.2)
        peaks_neg, _ = signal.find_peaks(-sig, distance=min_dist, prominence=0.2)

        amp_pos = float(np.mean( sig[peaks_pos])) if len(peaks_pos) > 0 else 0.0
        amp_neg = float(np.mean(-sig[peaks_neg])) if len(peaks_neg) > 0 else 0.0


        should_flip = (
            (len(peaks_neg) > len(peaks_pos)) or
            (amp_neg > amp_pos * 1.3 and len(peaks_neg) > 0)
        )
        if should_flip:
            mat[:, ch] = -sig
    return mat

def _robust_segment_polarity_decision(w, fs=TARGET_FS):
    """Return whether a single segment/channel should be sign-flipped.

    The decision is made on a temporary robustly scaled copy of the segment.
    This is more stable than using a fixed peak prominence on raw amplitudes.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.size < max(int(fs), 8):
        return False, {
            "method": "too_short",
            "n_peaks": 0,
            "median_peak_amplitude": 0.0,
            "p99": 0.0,
            "p01": 0.0,
        }

    centered = w - np.median(w)
    scale = float(np.percentile(np.abs(centered), 95))
    if (not np.isfinite(scale)) or scale < 1e-6:
        return False, {
            "method": "near_flat",
            "n_peaks": 0,
            "median_peak_amplitude": 0.0,
            "p99": 0.0,
            "p01": 0.0,
        }

    z = centered / scale
    p99 = float(np.percentile(z, 99))
    p01 = float(np.percentile(z, 1))

    min_distance = max(int(0.25 * fs), 1)
    prominence = max(0.25, 0.15 * float(np.std(z)))
    peaks, _ = signal.find_peaks(
        np.abs(z),
        distance=min_distance,
        prominence=prominence,
    )

    if peaks.size >= SEG_POLARITY_MIN_PEAKS:
        strengths = np.abs(z[peaks])
        keep_n = min(max(20, SEG_POLARITY_MIN_PEAKS), peaks.size)
        keep = np.argsort(strengths)[-keep_n:]
        peak_values = centered[peaks[keep]]
        median_peak_amp = float(np.median(peak_values))
        flip = bool(median_peak_amp < 0.0)
        return flip, {
            "method": "dominant_abs_peak_sign",
            "n_peaks": int(peaks.size),
            "median_peak_amplitude": median_peak_amp,
            "p99": p99,
            "p01": p01,
        }


    flip = bool(abs(p01) > abs(p99) * SEG_POLARITY_FLIP_MARGIN)
    return flip, {
        "method": "percentile_tail_asymmetry",
        "n_peaks": int(peaks.size),
        "median_peak_amplitude": 0.0,
        "p99": p99,
        "p01": p01,
    }


def apply_segment_level_polarity(
    mat,
    fs=TARGET_FS,
    window_size=QC_WINDOW_SIZE,
    stride=QC_STRIDE,
    segment_channel_good_mask=None,
):
    """Apply polarity normalization to each training-aligned segment.

    Segments rejected by the channel-level QC mask are not flipped.
    """
    mat = np.asarray(mat, dtype=np.float32).copy()
    n_channels, n = mat.shape
    starts = np.arange(0, max(n - window_size + 1, 0), stride, dtype=np.int64)
    n_segments = int(len(starts))

    flip_mask = np.zeros((n_segments, n_channels), dtype=np.bool_)
    skipped_by_qc_mask = np.zeros((n_segments, n_channels), dtype=np.bool_)
    method_counts = {}

    if segment_channel_good_mask is not None:
        segment_channel_good_mask = np.asarray(segment_channel_good_mask, dtype=np.bool_)
        if segment_channel_good_mask.shape != (n_segments, n_channels):
            segment_channel_good_mask = None

    if not ENABLE_SEGMENT_POLARITY_NORMALIZATION or n_segments == 0:
        return mat, {
            "enabled": bool(ENABLE_SEGMENT_POLARITY_NORMALIZATION),
            "segment_window_size": int(window_size),
            "segment_stride": int(stride),
            "n_segments": int(n_segments),
            "flip_count": 0,
            "flip_ratio": 0.0,
            "flip_mask": flip_mask,
            "skipped_by_qc_mask": skipped_by_qc_mask,
            "method_counts": method_counts,
        }

    for seg_i, start in enumerate(starts):
        end = int(start + window_size)
        for ch in range(n_channels):
            if segment_channel_good_mask is not None and not bool(segment_channel_good_mask[seg_i, ch]):
                skipped_by_qc_mask[seg_i, ch] = True
                continue

            flip, info = _robust_segment_polarity_decision(mat[ch, int(start):end], fs=fs)
            method = str(info.get("method", "unknown"))
            method_counts[method] = method_counts.get(method, 0) + 1
            if flip:
                mat[ch, int(start):end] *= -1.0
                flip_mask[seg_i, ch] = True

    flip_count = int(np.sum(flip_mask))
    denom = max(int(n_segments * max(n_channels, 1)), 1)
    return mat.astype(np.float32), {
        "enabled": True,
        "segment_window_size": int(window_size),
        "segment_stride": int(stride),
        "n_segments": int(n_segments),
        "flip_count": flip_count,
        "flip_ratio": round(float(flip_count) / float(denom), 4),
        "flip_mask": flip_mask,
        "skipped_by_qc_mask": skipped_by_qc_mask,
        "skipped_by_qc_count": int(np.sum(skipped_by_qc_mask)),
        "method_counts": method_counts,
    }


def _segment_channel_quality(w, fs=TARGET_FS, clip_value=CLIP_VALUE, hp_sos=None):
    """Evaluate channel quality within a training-aligned segment.

    Only low standard deviation is used for the good/bad decision.
    Other metrics are retained for diagnostics.
    """
    w = np.asarray(w, dtype=np.float32)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    centered = w - np.median(w)
    std = float(np.std(centered))
    clip_ratio = float(np.mean(np.abs(w) >= clip_value * 0.95))


    mad = float(np.median(np.abs(centered)) + 1e-6)


    low_std = std < QC_MIN_STD
    high_clip = clip_ratio > QC_MAX_CLIP_RATIO


    if hp_sos is not None and std > 1e-6:
        try:
            w_hp = signal.sosfilt(hp_sos, centered)
            hf_proxy = float(np.std(w_hp) / max(std, 1e-6))
        except Exception:
            hf_proxy = 0.0
    else:
        hf_proxy = 0.0
    high_hf = hf_proxy > 0.70


    try:
        min_dist = max(int(0.20 * fs), 1)


        adaptive_prominence = max(0.20, 0.35 * mad, 0.10 * std)
        peaks, props = signal.find_peaks(
            centered,
            distance=min_dist,
            prominence=adaptive_prominence,
        )

        duration_sec = len(centered) / float(fs)
        hr_est = float(len(peaks) * 60.0 / max(duration_sec, 1e-6))

        prominences = np.asarray(props.get("prominences", []), dtype=np.float32)
        if prominences.size > 0:
            qrs_snr = float(np.median(prominences) / mad)
            median_prominence = float(np.median(prominences))
        else:
            qrs_snr = 0.0
            median_prominence = 0.0

        if len(peaks) >= 3:
            rr = np.diff(peaks).astype(np.float32) / float(fs)
            rr_cv = float(np.std(rr) / (np.mean(rr) + 1e-6))
        else:
            rr_cv = np.inf
    except Exception:
        peaks = []
        hr_est = 0.0
        qrs_snr = 0.0
        median_prominence = 0.0
        rr_cv = np.inf

    bad_hr = bool((hr_est < QC_MIN_HR) or (hr_est > QC_MAX_HR))
    bad_qrs_snr = bool(qrs_snr < QC_MIN_QRS_SNR)
    bad_rr_cv = bool(rr_cv > QC_MAX_RR_CV)


    score = 0.0 if low_std else 1.0
    is_good = not low_std

    stats = {
        "std": std,
        "clip_ratio": clip_ratio,
        "hf_proxy": hf_proxy,
        "hr_est": hr_est,
        "n_peaks": int(len(peaks)),
        "qrs_snr": qrs_snr,
        "median_prominence": median_prominence,
        "rr_cv": None if not np.isfinite(rr_cv) else float(rr_cv),
        "low_std": bool(low_std),
        "high_clip_ratio": bool(high_clip),
        "high_freq_noise": bool(high_hf),
        "bad_hr": bool(bad_hr),
        "bad_qrs_snr": bool(bad_qrs_snr),
        "bad_rr_cv": bool(bad_rr_cv),
    }
    return bool(is_good), score, stats


def compute_segment_qc_mask(
    mat,
    fs=TARGET_FS,
    clip_value=CLIP_VALUE,
    window_size=QC_WINDOW_SIZE,
    stride=QC_STRIDE,
):
    """Compute a training-aligned segment QC mask.

    Args:
        mat: preprocessed ECG array shaped [channels, time], after polarity norm
             and robust normalization.
        window_size: should match training record-cache window_size.
        stride: should match training record-cache stride.

    Returns:
        dict containing:
            good_mask: bool array [n_segments]
            quality_scores: float array [n_segments]
            starts: int array [n_segments]
            summary metadata fields
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Expected mat with shape [channels, time], got {mat.shape}")

    n_channels, n = mat.shape
    starts = np.arange(0, max(n - window_size + 1, 0), stride, dtype=np.int64)
    n_segments = int(len(starts))

    if n_segments == 0:
        return {
            "good_mask": np.zeros(0, dtype=np.bool_),
            "quality_scores": np.zeros(0, dtype=np.float32),
            "channel_good_mask": np.zeros((0, n_channels), dtype=np.bool_),
            "channel_quality_scores": np.zeros((0, n_channels), dtype=np.float32),
            "best_channel_index": np.zeros(0, dtype=np.int64),
            "starts": starts,
            "quality_score": 0.0,
            "bad_segment_ratio": 1.0,
            "good_segment_count": 0,
            "n_qc_segments": 0,
            "clip_ratio": 0.0,
            "hf_ratio": 0.0,
            "qrs_snr_mean": 0.0,
            "rr_cv_mean": 0.0,
            "quality_flags": {
                "no_qc_segments": True,
                "low_good_segment_ratio": True,
            },
        }


    try:
        hp_sos = signal.butter(2, 10.0 / (fs * 0.5), btype="high", output="sos")
    except Exception:
        hp_sos = None

    good_mask = np.zeros(n_segments, dtype=np.bool_)
    quality_scores = np.zeros(n_segments, dtype=np.float32)
    channel_good_mask = np.zeros((n_segments, n_channels), dtype=np.bool_)
    channel_quality_scores = np.zeros((n_segments, n_channels), dtype=np.float32)
    best_channel_index = np.zeros(n_segments, dtype=np.int64)

    clip_ratios = []
    hf_ratios = []
    low_std_count = 0
    high_clip_count = 0
    high_hf_count = 0
    bad_hr_count = 0
    bad_qrs_snr_count = 0
    bad_rr_cv_count = 0
    qrs_snr_values = []
    rr_cv_values = []

    for i, start in enumerate(starts):
        end = int(start + window_size)
        segment = mat[:, int(start):end]

        ch_good = []
        ch_scores = []
        ch_stats = []

        for ch in range(n_channels):
            is_good, score, stats = _segment_channel_quality(
                segment[ch],
                fs=fs,
                clip_value=clip_value,
                hp_sos=hp_sos,
            )
            ch_good.append(is_good)
            ch_scores.append(score)
            ch_stats.append(stats)


        if ch_good:
            channel_good_mask[i, :len(ch_good)] = np.asarray(ch_good, dtype=np.bool_)
            channel_quality_scores[i, :len(ch_scores)] = np.asarray(ch_scores, dtype=np.float32)


        good_mask[i] = bool(np.any(ch_good))
        quality_scores[i] = float(np.max(ch_scores)) if ch_scores else 0.0


        best_ch = int(np.argmax(ch_scores)) if ch_scores else 0
        best_channel_index[i] = best_ch
        stats = ch_stats[best_ch] if ch_stats else {}
        clip_ratios.append(float(stats.get("clip_ratio", 0.0)))
        hf_ratios.append(float(stats.get("hf_proxy", 0.0)))
        low_std_count += int(stats.get("low_std", False))
        high_clip_count += int(stats.get("high_clip_ratio", False))
        high_hf_count += int(stats.get("high_freq_noise", False))
        bad_hr_count += int(stats.get("bad_hr", False))
        bad_qrs_snr_count += int(stats.get("bad_qrs_snr", False))
        bad_rr_cv_count += int(stats.get("bad_rr_cv", False))
        qrs_snr_values.append(float(stats.get("qrs_snr", 0.0)))
        rr_cv_val = stats.get("rr_cv", None)
        if rr_cv_val is not None and np.isfinite(float(rr_cv_val)):
            rr_cv_values.append(float(rr_cv_val))

    good_count = int(np.sum(good_mask))
    bad_ratio = float(1.0 - good_count / max(n_segments, 1))
    mean_score = float(np.mean(quality_scores)) if n_segments else 0.0
    good_ratio = float(good_count / max(n_segments, 1))

    flags = {
        "no_qc_segments": False,
        "low_good_segment_ratio": bool(good_ratio < QC_MIN_GOOD_RATIO),
        "low_std_segment_ratio": round(low_std_count / max(n_segments, 1), 4),
        "high_clip_segment_ratio": round(high_clip_count / max(n_segments, 1), 4),
        "high_freq_segment_ratio": round(high_hf_count / max(n_segments, 1), 4),
        "bad_hr_segment_ratio": round(bad_hr_count / max(n_segments, 1), 4),
        "bad_qrs_snr_segment_ratio": round(bad_qrs_snr_count / max(n_segments, 1), 4),
        "bad_rr_cv_segment_ratio": round(bad_rr_cv_count / max(n_segments, 1), 4),
    }

    return {
        "good_mask": good_mask.astype(np.bool_),
        "quality_scores": quality_scores.astype(np.float32),
        "channel_good_mask": channel_good_mask.astype(np.bool_),
        "channel_quality_scores": channel_quality_scores.astype(np.float32),
        "best_channel_index": best_channel_index.astype(np.int64),
        "channel_good_segment_count": np.sum(channel_good_mask, axis=0).astype(np.int64).tolist(),
        "channel_bad_segment_ratio": (1.0 - np.mean(channel_good_mask, axis=0)).astype(float).round(4).tolist(),
        "starts": starts.astype(np.int64),
        "quality_score": round(mean_score, 4),
        "bad_segment_ratio": round(bad_ratio, 4),
        "good_segment_count": good_count,
        "n_qc_segments": n_segments,
        "clip_ratio": round(float(np.mean(clip_ratios)), 4) if clip_ratios else 0.0,
        "hf_ratio": round(float(np.mean(hf_ratios)), 4) if hf_ratios else 0.0,
        "qrs_snr_mean": round(float(np.mean(qrs_snr_values)), 4) if qrs_snr_values else 0.0,
        "rr_cv_mean": round(float(np.mean(rr_cv_values)), 4) if rr_cv_values else 0.0,
        "quality_flags": flags,
    }


def preprocess_record(record_path):
    record = wfdb.rdrecord(record_path, physical=True)
    if record.p_signal is None:
        raise ValueError(f"No physical signal found for record: {record_path}")

    fs        = float(record.fs)
    sig_names = list(record.sig_name)
    data      = np.asarray(record.p_signal, dtype=np.float32)
    data      = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


    selected_indices = choose_channel_indices(sig_names)
    selected_names   = [sig_names[idx] for idx in selected_indices]
    channel_info     = infer_channel_metadata(selected_names)
    data             = data[:, selected_indices]


    data = bandpass_filter(data, fs)
    data = resample_to_target_fs(data, fs, TARGET_FS)


    data = robust_normalize(data, CLIP_VALUE)
    processed = data.T.astype(np.float32)


    quality_info = compute_segment_qc_mask(processed, fs=TARGET_FS)

    n_segments = int(quality_info.get("n_qc_segments", 0))
    n_channels = int(processed.shape[0])
    segment_polarity_info = {
        "enabled": False,
        "segment_window_size": int(QC_WINDOW_SIZE),
        "segment_stride": int(QC_STRIDE),
        "n_segments": n_segments,
        "flip_count": 0,
        "flip_ratio": 0.0,
        "flip_mask": np.zeros((n_segments, n_channels), dtype=np.bool_),
        "skipped_by_qc_mask": np.zeros((n_segments, n_channels), dtype=np.bool_),
        "skipped_by_qc_count": 0,
        "method_counts": {},
    }

    quality_info["segment_polarity_info"] = segment_polarity_info
    quality_info["pre_polarity_bad_segment_ratio"] = quality_info.get("bad_segment_ratio", 0.0)
    quality_info["pre_polarity_quality_score"] = quality_info.get("quality_score", 0.0)

    return processed, fs, selected_names, sig_names, quality_info, channel_info


def build_output_paths(record_stem):
    rel_stem  = os.path.relpath(record_stem, INPUT_DIR)
    save_path = os.path.join(OUTPUT_DIR, rel_stem + ".npy")
    meta_path = os.path.join(OUTPUT_DIR, rel_stem + ".json")
    qc_path   = os.path.join(OUTPUT_DIR, rel_stem + "_qc_mask.npz")
    return save_path, meta_path, qc_path

def outputs_are_complete(save_path, meta_path, qc_path):
    if not os.path.exists(save_path):
        return False
    if not os.path.exists(qc_path):
        return False
    if not SAVE_METADATA:
        return True
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    required_keys = {
        "segment_end_hour", "time_bucket", "patient_id", "segment_index",
        "quality_score", "qc_mask_path", "qc_window_size", "qc_stride",
        "n_qc_segments", "good_segment_count", "segment_polarity_normalization",
        "hospital", "channel_type", "channel_mask", "effective_input_channels",
        "preprocess_version",
    }
    if not required_keys.issubset(metadata.keys()):
        return False


    return metadata.get("preprocess_version") == "qc_mask_v7_500hz_channel_mask_channel_qc_no_polarity_flatline_only"

def process_single_record(record_stem):
    save_path, meta_path, qc_path = build_output_paths(record_stem)
    if outputs_are_complete(save_path, meta_path, qc_path):
        return ("skipped", record_stem, None)

    try:
        processed, original_fs, selected_names, all_sig_names, quality_info, channel_info = preprocess_record(record_stem)
        time_info        = parse_record_time_info(record_stem)
        header_time_info = extract_header_time_fields(record_stem)
        hospital         = extract_hospital_from_patient_txt(record_stem)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, processed)
        segment_polarity_info = quality_info.get("segment_polarity_info", {})
        segment_polarity_flip_mask = segment_polarity_info.get(
            "flip_mask",
            np.zeros((len(quality_info["good_mask"]), processed.shape[0]), dtype=np.bool_),
        )
        np.savez_compressed(
            qc_path,
            good_mask=quality_info["good_mask"].astype(np.bool_),
            quality_scores=quality_info["quality_scores"].astype(np.float32),
            channel_good_mask=quality_info["channel_good_mask"].astype(np.bool_),
            channel_quality_scores=quality_info["channel_quality_scores"].astype(np.float32),
            best_channel_index=quality_info["best_channel_index"].astype(np.int64),
            starts=quality_info["starts"].astype(np.int64),
            segment_polarity_flip_mask=np.asarray(segment_polarity_flip_mask, dtype=np.bool_),
            window_size=np.asarray([QC_WINDOW_SIZE], dtype=np.int64),
            stride=np.asarray([QC_STRIDE], dtype=np.int64),
        )

        if SAVE_METADATA:
            metadata = {
                "record":            os.path.basename(record_stem),
                "path":              save_path,
                "original_fs":       float(original_fs),
                "target_fs":         TARGET_FS,
                "shape":             list(processed.shape),
                "signal_len":        int(processed.shape[1]),
                "n_channels":        int(processed.shape[0]),
                "all_channels":      [str(n) for n in all_sig_names],
                "selected_channels": [str(n) for n in selected_names],
                "hospital": hospital,
                "channel_type": channel_info.get("channel_type", "unknown"),
                "channel_mask": channel_info.get("channel_mask", [1, 1]),
                "effective_input_channels": channel_info.get("effective_input_channels", 2),
                "channel_selection_method": (
                    "icare_pair_or_single_name_priority"
                    if any(normalize_lead_name(n) in {"ECG","ECG1","ECG2","ECGL","ECGR"}
                           for n in selected_names)
                    else "standard_lead_priority_or_fallback"
                ),
                "preprocess_version": "qc_mask_v7_500hz_channel_mask_channel_qc_no_polarity_flatline_only",
                "normalization":  "median_center + p95_abs_scale + clip; no polarity normalization",
                "bandpass_hz":    [LOWCUT_HZ, HIGHCUT_HZ],
                "time_anchor":    "segment_end_hour_from_filename",
                "time_label_confidence": "bucket_aligned",

                "qc_mask_path":      qc_path,
                "qc_window_size":    QC_WINDOW_SIZE,
                "qc_stride":         QC_STRIDE,
                "segment_polarity_normalization": False,
                "segment_polarity_flip_count": int(segment_polarity_info.get("flip_count", 0)),
                "segment_polarity_flip_ratio": segment_polarity_info.get("flip_ratio", 0.0),
                "segment_polarity_method_counts": segment_polarity_info.get("method_counts", {}),
                "segment_polarity_skipped_by_qc_count": int(segment_polarity_info.get("skipped_by_qc_count", 0)),
                "n_qc_segments":     int(quality_info["n_qc_segments"]),
                "good_segment_count": int(quality_info["good_segment_count"]),
                "channel_good_segment_count": quality_info.get("channel_good_segment_count", []),
                "channel_bad_segment_ratio": quality_info.get("channel_bad_segment_ratio", []),
                "pre_polarity_bad_segment_ratio": quality_info.get("pre_polarity_bad_segment_ratio", 0.0),
                "pre_polarity_quality_score": quality_info.get("pre_polarity_quality_score", 0.0),
                "quality_score":     quality_info["quality_score"],
                "bad_segment_ratio": quality_info["bad_segment_ratio"],
                "hf_ratio":          quality_info["hf_ratio"],
                "qrs_snr_mean":      quality_info.get("qrs_snr_mean", 0.0),
                "rr_cv_mean":        quality_info.get("rr_cv_mean", 0.0),
                "clip_ratio":        quality_info["clip_ratio"],
                "quality_flags":     quality_info["quality_flags"],
            }
            metadata.update(time_info)
            metadata.update(header_time_info)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=True, indent=2)

        return ("ok", record_stem, None)

    except Exception as exc:
        return ("error", record_stem, str(exc))


def build_processed_index():
    index_records = []
    for current_root, _, files in os.walk(OUTPUT_DIR):
        for file_name in sorted(files):
            if not file_name.endswith(".json"):
                continue
            meta_path = os.path.join(current_root, file_name)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            npy_path = os.path.splitext(meta_path)[0] + ".npy"
            if not os.path.exists(npy_path):
                continue

            qc_path = metadata.get("qc_mask_path")
            if qc_path is None:
                qc_path = os.path.splitext(meta_path)[0] + "_qc_mask.npz"

            shape      = metadata.get("shape", [])
            signal_len = metadata.get("signal_len")
            if signal_len is None and len(shape) >= 2:
                signal_len = int(shape[1])

            index_records.append({
                "path":                    npy_path,
                "patient_id":              metadata.get("patient_id"),
                "hospital":                metadata.get("hospital", "Unknown"),
                "selected_channels":       metadata.get("selected_channels", []),
                "channel_type":            metadata.get("channel_type", "unknown"),
                "channel_mask":            metadata.get("channel_mask", [1, 1]),
                "effective_input_channels": metadata.get("effective_input_channels", 2),
                "time_bucket":             metadata.get("time_bucket", "unknown"),
                "segment_end_hour":        metadata.get("segment_end_hour"),
                "time_bucket_upper_hour":  metadata.get("time_bucket_upper_hour"),
                "time_anchor":             metadata.get("time_anchor"),
                "time_label_confidence":   metadata.get("time_label_confidence"),
                "signal_len":              signal_len,
                "shape":                   shape,

                "quality_score":           metadata.get("quality_score", 1.0),
                "bad_segment_ratio":       metadata.get("bad_segment_ratio", 0.0),
                "channel_good_segment_count": metadata.get("channel_good_segment_count", []),
                "channel_bad_segment_ratio": metadata.get("channel_bad_segment_ratio", []),
                "qc_mask_path":            qc_path,
                "qc_window_size":          metadata.get("qc_window_size", QC_WINDOW_SIZE),
                "qc_stride":               metadata.get("qc_stride", QC_STRIDE),
                "n_qc_segments":           metadata.get("n_qc_segments", 0),
                "good_segment_count":      metadata.get("good_segment_count", 0),
                "qrs_snr_mean":            metadata.get("qrs_snr_mean", 0.0),
                "rr_cv_mean":              metadata.get("rr_cv_mean", 0.0),
                "segment_polarity_normalization": metadata.get("segment_polarity_normalization", False),
                "segment_polarity_flip_ratio": metadata.get("segment_polarity_flip_ratio", 0.0),
            })

    json_path = os.path.join(OUTPUT_DIR, INDEX_JSON)
    pkl_path  = os.path.join(OUTPUT_DIR, INDEX_PKL)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=True, indent=2)
    with open(pkl_path, "wb") as f:
        pickle.dump(index_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(index_records), json_path, pkl_path

def collect_record_stems():
    record_stems = []
    for root, dirs, files in os.walk(INPUT_DIR):
        dirs.sort()
        files.sort()
        for file_name in files:
            if not file_name.endswith("_ECG.hea"):
                continue

            pid = os.path.basename(root)
            if not pid.isdigit():
                continue
            record_stem = os.path.join(root, os.path.splitext(file_name)[0])
            record_stems.append(record_stem)
    return record_stems

def run_conversion():
    tasks = collect_record_stems()
    print(
        f"[Info] Starting QC-mask flatline-only preprocessing | Records: {len(tasks)} | "
        f"Processes: {PROCESSES} | Target fs: {TARGET_FS} Hz | "
        f"Output: {OUTPUT_DIR} | Polarity normalization: disabled",
        flush=True,
    )

    ok_count = skipped_count = error_count = 0
    errors = []

    progress_bar = tqdm(total=len(tasks), desc="Preprocessing (flatline-only QC mask)", dynamic_ncols=True, mininterval=0.2)

    with Pool(processes=PROCESSES) as pool:
        for status, record_stem, message in pool.imap_unordered(
            process_single_record, tasks, chunksize=1
        ):
            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                error_count += 1
                errors.append(f"{record_stem}: {message}")
            progress_bar.update(1)
            progress_bar.set_postfix(ok=ok_count, skipped=skipped_count, error=error_count, refresh=False)

    progress_bar.close()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for item in errors:
            f.write(item + "\n")

    index_count, index_json_path, index_pkl_path = build_processed_index()

    print("\n[Done] QC-mask flatline-only preprocessing completed", flush=True)
    print(f"[Summary] Success: {ok_count}", flush=True)
    print(f"[Summary] Skipped existing: {skipped_count}", flush=True)
    print(f"[Summary] Failed: {error_count} | Log: {LOG_FILE}", flush=True)
    print(
        f"[Summary] Built processed index: {index_count} records | "
        f"JSON: {index_json_path} | PKL: {index_pkl_path}",
        flush=True,
    )

if __name__ == "__main__":
    run_conversion()
