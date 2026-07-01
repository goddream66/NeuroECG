# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from data.cache import ensure_channel_first_with_mask, get_good_segment_indices_for_record, load_processed_array, load_qc_mask_full
from feature.hrv import compute_hrv_features

class ICAREDataset(Dataset):
    """Segment-level I-CARE ECG dataset.

    One item is one preprocessed ECG segment.  It returns the waveform, the
    effective channel mask, the patient-level labels/metadata, and HRV features.
    This segment dataset is intentionally separate from PatientBagDataset: the
    latter decides how to group segments into WVNUM-style local entries.
    """

    def __init__(self, samples, segment_len=5000, is_train=False, clip_value=5.0, compute_hrv=True):
        self.samples = samples
        self.segment_len = int(segment_len)
        self.is_train = bool(is_train)
        self.clip_value = clip_value
        self.compute_hrv = bool(compute_hrv)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            mat, default_channel_mask = ensure_channel_first_with_mask(
                load_processed_array(sample["path"]),
                sample["path"],
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load processed ECG from {sample['path']}") from exc

        channel_mask = np.asarray(
            sample.get("channel_mask", default_channel_mask),
            dtype=np.float32,
        ).reshape(-1)
        if channel_mask.size == 0:
            channel_mask = default_channel_mask.astype(np.float32)
        if channel_mask.size == 1:
            channel_mask = np.asarray([float(channel_mask[0]), 0.0], dtype=np.float32)
        elif channel_mask.size > 2:
            channel_mask = channel_mask[:2].astype(np.float32)
        else:
            channel_mask = channel_mask.astype(np.float32)
        if channel_mask.size < 2:
            channel_mask = np.pad(channel_mask, (0, 2 - channel_mask.size), constant_values=0.0).astype(np.float32)
        if float(np.sum(channel_mask)) <= 0.0:
            channel_mask = default_channel_mask.astype(np.float32)

        start = int(sample.get("start", 0))
        end = start + int(sample.get("window_size", self.segment_len))

        selected_channel_index = sample.get("selected_channel_index", None)
        if selected_channel_index is not None:
            selected_channel_index = int(selected_channel_index)
            selected_channel_index = max(0, min(selected_channel_index, mat.shape[0] - 1))
            # Single-lead sample: keep exactly one channel, no channel fusion.
            mat = mat[selected_channel_index:selected_channel_index + 1, start:end]
            channel_mask = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            # Backward-compatible fallback for older cached samples.
            mat = mat[:, start:end]

        if mat.shape[1] != self.segment_len:
            mat = signal.resample(mat, self.segment_len, axis=-1).astype(np.float32)

        if mat.shape[1] < self.segment_len:
            pad_width = self.segment_len - mat.shape[1]
            mat = np.pad(mat, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mat = mat[:, : self.segment_len]

        if self.clip_value is not None:
            mat = np.clip(mat, -float(self.clip_value), float(self.clip_value))

        # Preprocessed files are already ECGFounder-style globally z-scored.
        # Leave this off unless running on raw/non-z-scored ECG arrays.
        if ECGFOUNDER_INPUT_ZSCORE:
            mean = float(np.mean(mat))
            std = float(np.std(mat))
            mat = ((mat - mean) / max(std, 1e-8)).astype(np.float32)

        if self.compute_hrv:
            hrv_features = compute_hrv_features(mat, fs=TARGET_FS, channel_mask=channel_mask)
        else:
            # Stage 1 ECGFounder training does not use HRV. Avoid expensive
            # per-segment CPU HRV extraction in DataLoader workers. HRV is
            # recomputed later only for Stage 2/Stage 3 feature blocks.
            hrv_features = np.zeros(len(HRV_FEATURE_NAMES), dtype=np.float32)

        return (
            torch.from_numpy(mat.copy()).float(),
            torch.tensor(channel_mask, dtype=torch.float32),
            torch.tensor(sample["label"], dtype=torch.long),
            torch.tensor(sample["cpc"], dtype=torch.float32),
            torch.tensor(sample["static"], dtype=torch.float32),
            sample["pid"],
            sample.get("time_bucket", "unknown"),
            torch.tensor(
                safe_hour_value(sample.get("segment_end_hour"), default=-1.0),
                dtype=torch.float32,
            ),
            torch.tensor(hrv_features, dtype=torch.float32),
        )


def _as_two_value_float_mask(mask, default=(1.0, 1.0)):
    """Convert a mask-like object to a length-2 float32 array."""
    try:
        arr = np.asarray(mask, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.asarray(default, dtype=np.float32)
    if arr.size == 0:
        arr = np.asarray(default, dtype=np.float32)
    if arr.size == 1:
        arr = np.asarray([float(arr[0]), 0.0], dtype=np.float32)
    elif arr.size >= 2:
        arr = arr[:2].astype(np.float32)
    else:
        arr = np.asarray(default, dtype=np.float32)
    return arr


def get_effective_channel_mask_for_segment(record_channel_mask, channel_good_mask, segment_idx):
    """Combine record-level channel availability with segment-level channel QC.

    record_channel_mask describes which channels are real in the record:
        single ECG copied to two channels -> [1, 0]
        real two-channel ECG             -> [1, 1]

    channel_good_mask describes which channel is usable in this specific segment.
    In the flatline-only preprocessing, a channel is bad only when it is nearly
    flat.  The effective mask is:
        effective_channel_mask = record_channel_mask * segment_channel_good_mask

    If channel_good_mask is missing or the multiplication removes all channels,
    fall back to the record-level mask so old preprocessed files remain usable.
    """
    base_mask = _as_two_value_float_mask(record_channel_mask)

    if channel_good_mask is None:
        return base_mask, np.asarray([np.nan, np.nan], dtype=np.float32), False

    try:
        ch_mask = np.asarray(channel_good_mask[segment_idx], dtype=np.float32).reshape(-1)
    except Exception:
        return base_mask, np.asarray([np.nan, np.nan], dtype=np.float32), False

    if ch_mask.size == 0:
        return base_mask, np.asarray([np.nan, np.nan], dtype=np.float32), False
    if ch_mask.size == 1:
        # Single-channel preprocessed record. The copied second channel is fake.
        ch_mask2 = np.asarray([float(ch_mask[0]), 0.0], dtype=np.float32)
    else:
        ch_mask2 = ch_mask[:2].astype(np.float32)

    effective_mask = base_mask * ch_mask2

    # Fallback for older / inconsistent QC files or records where all channels
    # are bad but the segment is kept to avoid losing the whole patient.
    if float(effective_mask.sum()) < 0.5:
        return base_mask, ch_mask2, False

    return effective_mask.astype(np.float32), ch_mask2.astype(np.float32), True


def build_indices(records, pid_set, use_qc_mask=True):
    """Build single-lead segment samples, optionally filtering by QC mask.

    New behavior for ECGFounder/single-lead training:
        - A two-channel segment is split into two independent samples when both
          channels are usable.
        - If only one channel is usable, only that channel is kept.
        - Each returned sample contains exactly one channel_index and will be
          loaded as [1, T] by ICAREDataset.
        - No feature-level channel fusion is used for new samples.

    Segment selection is still controlled by good_mask from preprocessing. With
    flatline-only preprocessing, good_mask removes only segments where all
    available channels are flat/bad.

    Fallback:
        If a record has no QC mask or all segments are bad, use all segments so
        that no patient disappears solely because of missing QC metadata.
    """
    samples = []
    total_segments_before_qc = 0
    total_segments_after_qc = 0
    records_with_qc = 0
    records_fallback = 0
    quality_values = []
    channel_qc_available_segments = 0
    channel_qc_applied_segments = 0
    channel_qc_changed_segments = 0

    for record in records:
        if record["pid"] not in pid_set:
            continue

        num_segments = int(record["num_segments"])
        total_segments_before_qc += num_segments

        good_mask = None
        quality_scores = None
        channel_good_mask = None
        channel_quality_scores = None
        best_channel_index = None

        if use_qc_mask:
            qc_path = record.get("qc_mask_path")
            qc_full = load_qc_mask_full(qc_path)
            if qc_full is not None:
                good_mask = qc_full.get("good_mask")
                quality_scores = qc_full.get("quality_scores")
                channel_good_mask = qc_full.get("channel_good_mask")
                channel_quality_scores = qc_full.get("channel_quality_scores")
                best_channel_index = qc_full.get("best_channel_index")
            if qc_path and os.path.exists(str(qc_path)):
                records_with_qc += 1

            segment_indices = get_good_segment_indices_for_record(record)

            if len(segment_indices) == num_segments:
                if good_mask is None or np.sum(good_mask[: min(len(good_mask), num_segments)]) == 0:
                    records_fallback += 1
        else:
            segment_indices = np.arange(num_segments, dtype=np.int64)

        total_segments_after_qc += int(len(segment_indices))

        record_channel_mask = record.get("channel_mask", [1.0, 1.0])

        for segment_idx in segment_indices:
            segment_idx = int(segment_idx)
            if quality_scores is not None and segment_idx < len(quality_scores):
                seg_quality = float(quality_scores[segment_idx])
            else:
                seg_quality = 1.0
            quality_values.append(seg_quality)

            effective_channel_mask, raw_channel_good_mask, channel_qc_applied = get_effective_channel_mask_for_segment(
                record_channel_mask=record_channel_mask,
                channel_good_mask=channel_good_mask,
                segment_idx=segment_idx,
            )
            if channel_good_mask is not None:
                channel_qc_available_segments += 1
            if channel_qc_applied:
                channel_qc_applied_segments += 1
            if not np.allclose(effective_channel_mask, _as_two_value_float_mask(record_channel_mask)):
                channel_qc_changed_segments += 1

            if channel_quality_scores is not None and segment_idx < len(channel_quality_scores):
                ch_quality = _as_two_value_float_mask(channel_quality_scores[segment_idx], default=(seg_quality, seg_quality))
            else:
                ch_quality = np.asarray([np.nan, np.nan], dtype=np.float32)

            if best_channel_index is not None and segment_idx < len(best_channel_index):
                best_ch = int(best_channel_index[segment_idx])
            else:
                best_ch = -1

            # Split usable channels into independent single-lead samples.
            # A two-channel ECG segment therefore contributes up to two samples:
            # channel 0 -> one [1, T] sample, channel 1 -> another [1, T] sample.
            valid_channel_indices = [
                int(ch) for ch, is_valid in enumerate(effective_channel_mask[:2])
                if float(is_valid) > 0.5
            ]
            if not valid_channel_indices:
                # Defensive fallback: keep the best channel if QC metadata is missing
                # or inconsistent. For single-channel records, this will be channel 0.
                valid_channel_indices = [int(best_ch) if int(best_ch) in (0, 1) else 0]

            for selected_channel_index in valid_channel_indices:
                selected_channel_quality = (
                    float(ch_quality[selected_channel_index])
                    if selected_channel_index < len(ch_quality) and np.isfinite(ch_quality[selected_channel_index])
                    else float(seg_quality)
                )
                samples.append(
                    {
                        "path": record["path"],
                        "start": segment_idx * record["stride"],
                        "segment_idx": segment_idx,
                        "window_size": record["window_size"],
                        "pid": record["pid"],
                        "label": record["label"],
                        "cpc": record["cpc"],
                        "static": record["static"],
                        "hospital": record["hospital"],
                        "time_bucket": record["time_bucket"],
                        "segment_end_hour": record["segment_end_hour"],
                        "qc_mask_path": record.get("qc_mask_path"),
                        "segment_quality_score": seg_quality,
                        "selected_channel_quality_score": selected_channel_quality,
                        "quality_score": record.get("quality_score", 1.0),
                        "bad_segment_ratio": record.get("bad_segment_ratio", 0.0),
                        "record_channel_mask": _as_two_value_float_mask(record_channel_mask).tolist(),
                        "raw_channel_good_mask": raw_channel_good_mask.tolist(),
                        "effective_channel_mask_before_split": effective_channel_mask.tolist(),
                        "channel_quality_scores": ch_quality.tolist(),
                        "best_channel_index": best_ch,
                        "selected_channel_index": int(selected_channel_index),
                        "selected_channel_name": f"ch{int(selected_channel_index)}",
                        "channel_qc_applied": bool(channel_qc_applied),
                        # The loaded tensor will be [1, T], so this mask only marks
                        # that the single input lead is real/valid. No fusion happens.
                        "channel_mask": [1.0, 0.0],
                        "single_lead_sample": True,
                    }
                )

    if use_qc_mask:
        if quality_values:
            q = np.quantile(np.asarray(quality_values, dtype=np.float32), [0.1, 0.5, 0.9])
            q_msg = f"seg_quality_q10/50/90={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}"
        else:
            q_msg = "seg_quality=NA"
        print(
            "[QC Mask] build_indices | "
            f"before={total_segments_before_qc} | after={total_segments_after_qc} | "
            f"kept={total_segments_after_qc / max(total_segments_before_qc, 1):.3f} | "
            f"records_with_qc={records_with_qc} | fallback_records={records_fallback} | {q_msg}"
        )
        print(
            "[Channel QC] single-lead split | "
            f"available_segments={channel_qc_available_segments} | "
            f"applied_segments={channel_qc_applied_segments} | "
            f"changed_from_record_mask={channel_qc_changed_segments} | "
            f"single_lead_samples={len(samples)}"
        )

    return samples


def summarize_time_buckets(records):
    counts = {}
    for record in records:
        bucket = record.get("time_bucket", "unknown")
        counts[bucket] = counts.get(bucket, 0) + int(record.get("num_segments", 0))
    return counts


def safe_hour_value(raw_value, default=-1.0):
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def limit_segments_per_patient(samples, indices, max_segments_per_patient, random_state=42):
    if max_segments_per_patient is None:
        return list(indices)

    rng = np.random.default_rng(random_state)
    grouped_indices = {}
    for idx in indices:
        pid = samples[idx]["pid"]
        grouped_indices.setdefault(pid, []).append(idx)

    limited_indices = []
    for pid, pid_indices in grouped_indices.items():
        if len(pid_indices) <= max_segments_per_patient:
            limited_indices.extend(pid_indices)
            continue
        selected = rng.choice(pid_indices, size=max_segments_per_patient, replace=False)
        limited_indices.extend(selected.tolist())

    return sorted(limited_indices)


def cap_samples_per_patient(samples: Sequence[dict], max_segments_per_patient):
    if max_segments_per_patient is None:
        return list(samples)

    max_segments_per_patient = int(max_segments_per_patient)
    if max_segments_per_patient <= 0:
        raise ValueError("max_segments_per_patient must be positive or None")

    grouped: Dict[str, List[dict]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["pid"]), []).append(sample)

    capped = []
    for pid in sorted(grouped.keys()):
        rows = grouped[pid]
        if len(rows) <= max_segments_per_patient:
            capped.extend(rows)
        else:
            idx = np.linspace(0, len(rows) - 1, max_segments_per_patient).round().astype(int)
            idx = np.unique(idx)
            capped.extend([rows[int(i)] for i in idx])

    print(
        f"[Cap Samples] before={len(samples)} | after={len(capped)} | "
        f"max_segments_per_patient={max_segments_per_patient}"
    )
    return capped


def build_wvnum_entry_samples(samples: Sequence[dict], split_name: str, max_entries_per_patient=None) -> List[dict]:
    """Build WVNUM continuous-entry samples while preserving local segment order.

    Each entry contains WVNUM consecutive segments from the same patient and source ECG file/path.
    The returned flat list contains all selected segment samples with added keys:
    entry_id, entry_pos, entry_start.
    """
    by_pid_path: Dict[Tuple[str, str], List[dict]] = {}
    for sample in samples:
        pid = str(sample["pid"])
        path = str(sample["path"])
        by_pid_path.setdefault((pid, path), []).append(sample)

    entries_by_pid: Dict[str, List[List[dict]]] = {}
    for (pid, path), rows in by_pid_path.items():
        rows = sorted(rows, key=lambda x: int(x.get("start", 0)))
        for i in range(0, len(rows), ENTRY_STRIDE_SEGMENTS):
            chunk = rows[i:i + WVNUM]
            if DROP_INCOMPLETE_ENTRY and len(chunk) < WVNUM:
                continue
            if len(chunk) == 0:
                continue
            entry_start = int(chunk[0].get("start", 0))
            entry_id = f"{pid}|{path}|{entry_start}"
            new_chunk = []
            for pos, s in enumerate(chunk):
                s2 = dict(s)
                s2["entry_id"] = entry_id
                s2["entry_pos"] = pos
                s2["entry_start"] = entry_start
                new_chunk.append(s2)
            entries_by_pid.setdefault(pid, []).append(new_chunk)

    selected_entries: List[List[dict]] = []
    for pid in sorted(entries_by_pid.keys()):
        entries = entries_by_pid[pid]
        if max_entries_per_patient is not None and len(entries) > int(max_entries_per_patient):
            idx = np.linspace(0, len(entries) - 1, int(max_entries_per_patient)).round().astype(int)
            idx = np.unique(idx)
            entries = [entries[int(i)] for i in idx]
        selected_entries.extend(entries)

    flat_samples = [s for entry in selected_entries for s in entry]
    print(
        f"[WVNUM Entries] {split_name}: original_segments={len(samples)} | "
        f"selected_entries={len(selected_entries)} | flat_segments={len(flat_samples)} | "
        f"WVNUM={WVNUM} | entry_stride={ENTRY_STRIDE_SEGMENTS} | "
        f"max_entries_per_patient={max_entries_per_patient}"
    )
    return flat_samples
