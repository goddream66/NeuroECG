# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from data.train_metadata import infer_pid_from_processed_path, load_patient_metadata

def load_processed_array(file_path, mmap_mode=None):
    loaded = np.load(file_path, allow_pickle=False, mmap_mode=mmap_mode)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.keys())
        if not keys:
            loaded.close()
            raise ValueError(f"Empty npz file: {file_path}")
        array = loaded[keys[0]]
        loaded.close()
        return array
    return loaded


def load_processed_metadata(file_path):
    meta_path = os.path.splitext(file_path)[0] + ".json"
    if not os.path.exists(meta_path):
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def infer_qc_mask_path_from_data_path(file_path):
    """Infer the segment-level QC mask path saved by process_qc_mask.py."""
    return os.path.splitext(file_path)[0] + "_qc_mask.npz"


def load_qc_mask_full(qc_mask_path):
    """Load segment-level and channel-level QC masks from preprocessing.

    The flatline-only preprocessing writes:
        good_mask: [n_segments], whether at least one channel is usable.
        quality_scores: [n_segments], segment diagnostic score.
        starts: [n_segments], segment start indices.
        channel_good_mask: [n_segments, n_channels], per-channel usability.

    channel_good_mask is the key field for channel-level fusion: if ECGL is
    valid but ECGR is flat in a segment, the effective channel mask becomes
    [1, 0], so the flat channel does not pollute the fused feature.
    """
    if not qc_mask_path or not os.path.exists(qc_mask_path):
        return None

    try:
        loaded = np.load(qc_mask_path, allow_pickle=False)
        good_mask = np.asarray(loaded["good_mask"], dtype=bool)
        quality_scores = (
            np.asarray(loaded["quality_scores"], dtype=np.float32)
            if "quality_scores" in loaded
            else np.ones_like(good_mask, dtype=np.float32)
        )
        starts = (
            np.asarray(loaded["starts"], dtype=np.int64)
            if "starts" in loaded
            else np.arange(len(good_mask), dtype=np.int64)
        )
        channel_good_mask = (
            np.asarray(loaded["channel_good_mask"], dtype=bool)
            if "channel_good_mask" in loaded
            else None
        )
        channel_quality_scores = (
            np.asarray(loaded["channel_quality_scores"], dtype=np.float32)
            if "channel_quality_scores" in loaded
            else None
        )
        best_channel_index = (
            np.asarray(loaded["best_channel_index"], dtype=np.int64)
            if "best_channel_index" in loaded
            else None
        )
        loaded.close()
        return {
            "good_mask": good_mask,
            "quality_scores": quality_scores,
            "starts": starts,
            "channel_good_mask": channel_good_mask,
            "channel_quality_scores": channel_quality_scores,
            "best_channel_index": best_channel_index,
        }
    except Exception:
        return None


def load_qc_mask(qc_mask_path):
    """Backward-compatible loader returning only segment-level fields."""
    qc = load_qc_mask_full(qc_mask_path)
    if qc is None:
        return None, None, None
    return qc["good_mask"], qc["quality_scores"], qc["starts"]


def get_record_qc_fields(path, metadata_or_index_item=None):
    """Collect QC-mask fields from processed metadata/index, with safe fallback."""
    info = metadata_or_index_item or {}
    qc_path = info.get("qc_mask_path")
    if not qc_path:
        qc_path = infer_qc_mask_path_from_data_path(path)

    return {
        "qc_mask_path": qc_path,
        "qc_window_size": info.get("qc_window_size"),
        "qc_stride": info.get("qc_stride"),
        "n_qc_segments": info.get("n_qc_segments", 0),
        "good_segment_count": info.get("good_segment_count", 0),
        "quality_score": info.get("quality_score", 1.0),
        "bad_segment_ratio": info.get("bad_segment_ratio", 0.0),
    }


def get_good_segment_indices_for_record(record):
    """Return training segment indices allowed by the segment-level QC mask.

    If a QC mask is unavailable or all segments are marked bad, fall back to all
    segments to avoid accidentally excluding an entire patient.
    """
    num_segments = int(record.get("num_segments", 0))
    if num_segments <= 0:
        return np.asarray([], dtype=np.int64)

    qc_path = record.get("qc_mask_path")
    good_mask, quality_scores, _ = load_qc_mask(qc_path)
    if good_mask is None or len(good_mask) == 0:
        return np.arange(num_segments, dtype=np.int64)

    n = min(num_segments, len(good_mask))
    good_indices = np.where(good_mask[:n])[0].astype(np.int64)

    if good_indices.size == 0:
        return np.arange(num_segments, dtype=np.int64)

    return good_indices


def save_record_cache(cache_path, records):
    with open(cache_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_record_cache(cache_path):
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_processed_index(index_path):
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def ensure_channel_first_with_mask(mat, file_path):
    """Return ECG as [2, T] plus a channel validity mask.

    The processed npy files may be [1, T] or [2, T].  A single-channel record is
    copied to [2, T] only to keep tensor shapes consistent, but channel_mask=[1,0]
    tells the model that the second channel is not a real independent signal.
    """
    mat = np.asarray(mat, dtype=np.float32).squeeze()
    if mat.ndim == 1:
        mat = mat[np.newaxis, :]
    elif mat.ndim != 2:
        raise RuntimeError(f"Unexpected processed ECG shape {mat.shape} in {file_path}")

    # Convert [T, C] to [C, T] when C is 1 or 2.
    if mat.shape[0] not in (1, 2) and mat.shape[1] in (1, 2):
        mat = mat.T

    original_channels = int(mat.shape[0])
    if original_channels == 1:
        mat = np.tile(mat, (2, 1))
        channel_mask = np.asarray([1.0, 0.0], dtype=np.float32)
    elif original_channels >= 2:
        mat = mat[:2, :]
        channel_mask = np.asarray([1.0, 1.0], dtype=np.float32)
    else:
        raise RuntimeError(f"No ECG channel found in {file_path}")

    return np.asarray(mat, dtype=np.float32), channel_mask


def ensure_channel_first(mat, file_path):
    mat, _ = ensure_channel_first_with_mask(mat, file_path)
    return mat


def infer_channel_mask_from_metadata(info):
    """Infer [1,0] or [1,1] from processed metadata/index when available."""
    if not info:
        return [1.0, 1.0]
    mask = info.get("channel_mask")
    if isinstance(mask, (list, tuple)) and len(mask) >= 2:
        return [float(mask[0]), float(mask[1])]
    n_channels = info.get("n_channels")
    if n_channels is None:
        shape = info.get("shape", [])
        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
            try:
                n_channels = int(shape[0])
            except Exception:
                n_channels = None
    if n_channels is not None and int(n_channels) <= 1:
        return [1.0, 0.0]
    return [1.0, 1.0]


def is_record_within_observation_window(time_bucket, segment_end_hour=None, selected_time_buckets=None):
    """Return True if a processed ECG record belongs to the configured observation window.

    For the 72h prediction setting, training/validation/test must only use ECG
    information available up to ROSC + 72h.  We therefore keep records whose
    time_bucket is in FIRST72_TIME_BUCKETS and, when segment_end_hour is present,
    require segment_end_hour <= MAX_OBSERVATION_HOUR.
    """
    if selected_time_buckets is not None and time_bucket not in set(selected_time_buckets):
        return False

    max_hour = globals().get("MAX_OBSERVATION_HOUR", None)
    if max_hour is None:
        return True

    if segment_end_hour is None:
        return True

    try:
        hour = float(segment_end_hour)
    except (TypeError, ValueError):
        return True

    return hour <= float(max_hour)


def scan_single_processed_file(task):
    (
        path,
        processed_root,
        pid_meta,
        window_size,
        stride,
        selected_time_buckets,
    ) = task

    pid = infer_pid_from_processed_path(path, processed_root, pid_meta)
    if pid is None:
        return None, "pid_or_load_failure", None

    processed_meta = load_processed_metadata(path)
    time_bucket = processed_meta.get("time_bucket", "unknown")
    segment_end_hour = processed_meta.get("segment_end_hour")
    if not is_record_within_observation_window(time_bucket, segment_end_hour, selected_time_buckets):
        return None, "time_filtered", None

    try:
        mat, channel_mask = ensure_channel_first_with_mask(load_processed_array(path, mmap_mode="r"), path)
    except Exception:
        return None, "pid_or_load_failure", None

    signal_len = int(mat.shape[1])
    if signal_len < window_size:
        return None, "short_file", None

    meta = pid_meta[pid]
    num_segments = 1 + (signal_len - window_size) // stride
    qc_fields = get_record_qc_fields(path, processed_meta)
    record_entry = {
        "path": path,
        "pid": pid,
        "label": meta["label"],
        "cpc": meta["cpc"],
        "static": meta["static"],
        "hospital": meta["hospital"],
        "time_bucket": time_bucket,
        "segment_end_hour": segment_end_hour,
        "window_size": window_size,
        "stride": stride,
        "signal_len": signal_len,
        "num_segments": num_segments,
        "channel_mask": channel_mask.tolist(),
        **qc_fields,
    }
    return record_entry, "ok", time_bucket


def build_records_from_index(index_records, processed_root, pid_meta, window_size, stride, selected_time_buckets):
    records = []
    skipped_files = 0
    skipped_short_files = 0
    skipped_time_files = 0
    selected_time_buckets = (
        None if selected_time_buckets is None else set(selected_time_buckets)
    )

    for item in index_records:
        path = item.get("path")
        if not path:
            skipped_files += 1
            continue

        pid = infer_pid_from_processed_path(path, processed_root, pid_meta)
        if pid is None:
            pid = item.get("patient_id")
        if pid is None or pid not in pid_meta:
            skipped_files += 1
            continue

        time_bucket = item.get("time_bucket", "unknown")
        segment_end_hour = item.get("segment_end_hour")
        if not is_record_within_observation_window(time_bucket, segment_end_hour, selected_time_buckets):
            skipped_time_files += 1
            continue

        signal_len = item.get("signal_len")
        if signal_len is None:
            shape = item.get("shape", [])
            if len(shape) >= 2:
                signal_len = int(shape[1])
        if signal_len is None:
            skipped_files += 1
            continue
        signal_len = int(signal_len)

        if signal_len < window_size:
            skipped_short_files += 1
            continue

        meta = pid_meta[pid]
        num_segments = 1 + (signal_len - window_size) // stride
        qc_fields = get_record_qc_fields(path, item)
        records.append(
            {
                "path": path,
                "pid": pid,
                "label": meta["label"],
                "cpc": meta["cpc"],
                "static": meta["static"],
                "hospital": meta["hospital"],
                "time_bucket": time_bucket,
                "segment_end_hour": segment_end_hour,
                "window_size": window_size,
                "stride": stride,
                "signal_len": signal_len,
                "num_segments": num_segments,
                "channel_mask": infer_channel_mask_from_metadata(item),
                **qc_fields,
            }
        )

    total_segments = int(sum(record["num_segments"] for record in records))
    print(
        f"[Index] Loaded records from processed index: {len(records)} | "
        f"Total segments: {total_segments} | "
        f"Skipped files: {skipped_files} | "
        f"Skipped short files: {skipped_short_files} | "
        f"Skipped by time filter: {skipped_time_files}"
    )
    return records


def scan_processed_samples(
    meta_root,
    processed_root,
    window_size=1000,
    stride=None,
    selected_time_buckets=None,
    scan_processes=3,
):
    scan_processes = max(1, min(int(scan_processes), CPU_WORKERS))
    pid_meta = load_patient_metadata(meta_root)
    records = []
    skipped_files = 0
    skipped_short_files = 0
    skipped_time_files = 0
    supported_suffixes = {".npy", ".npz"}
    stride = window_size if stride is None else stride
    selected_time_buckets = (
        None if selected_time_buckets is None else set(selected_time_buckets)
    )

    print(f"[Info] Scanning processed files in: {processed_root}")
    candidate_paths = []
    for current_root, _, files in os.walk(processed_root):
        for file_name in sorted(files):
            suffix = os.path.splitext(file_name)[1].lower()
            if suffix not in supported_suffixes:
                continue
            if file_name.endswith("_qc_mask.npz"):
                continue
            candidate_paths.append(os.path.join(current_root, file_name))

    tasks = [
        (
            path,
            processed_root,
            pid_meta,
            window_size,
            stride,
            selected_time_buckets,
        )
        for path in candidate_paths
    ]
    with Pool(processes=scan_processes) as pool:
        for record_entry, status, _ in pool.imap_unordered(
            scan_single_processed_file,
            tasks,
            chunksize=1,
        ):
            if status == "ok":
                records.append(record_entry)
            elif status == "time_filtered":
                skipped_time_files += 1
            elif status == "short_file":
                skipped_short_files += 1
            else:
                skipped_files += 1

    total_segments = int(sum(record["num_segments"] for record in records))
    print(
        f"[Done] Cached records: {len(records)} | "
        f"Total processed segments found: {total_segments} | "
        f"Skipped files without matched pid/load failure: {skipped_files} | "
        f"Skipped short files: {skipped_short_files} | "
        f"Skipped by time filter: {skipped_time_files}"
    )
    return records, pid_meta


def get_or_build_record_cache(
    meta_root,
    processed_root,
    cache_path,
    window_size=1000,
    stride=None,
    selected_time_buckets=None,
    scan_processes=3,
):
    pid_meta = load_patient_metadata(meta_root)
    if os.path.exists(cache_path):
        print(f"[Cache] Loading record cache from: {cache_path}")
        records = load_record_cache(cache_path)
        print(
            f"[Cache] Loaded cached records: {len(records)} | "
            f"Total segments: {int(sum(record['num_segments'] for record in records))}"
        )
        return records, pid_meta

    if bool(globals().get("REQUIRE_RECORD_CACHE", False)):
        raise FileNotFoundError(
            f"Required record cache not found: {cache_path}. "
            "REQUIRE_RECORD_CACHE=True, so the training script will not rebuild it from processed_index."
        )

    index_path = os.path.join(processed_root, PROCESSED_INDEX_PKL)
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Required processed index not found: {index_path}. "
            "This training run uses the processed index configured in pipeline/train_config.py "
            "and will not silently fall back to another processed_index file."
        )
    if os.path.exists(index_path):
        print(f"[Index] Loading processed index from: {index_path}")
        index_records = load_processed_index(index_path)
        records = build_records_from_index(
            index_records,
            processed_root,
            pid_meta,
            window_size,
            stride if stride is not None else window_size,
            selected_time_buckets,
        )
        save_record_cache(cache_path, records)
        print(f"[Cache] Saved record cache to: {cache_path}")
        return records, pid_meta

    print(f"[Cache] Cache not found. Building: {cache_path}")
    records, _ = scan_processed_samples(
        meta_root,
        processed_root,
        window_size=window_size,
        stride=stride,
        selected_time_buckets=selected_time_buckets,
        scan_processes=scan_processes,
    )
    save_record_cache(cache_path, records)
    print(f"[Cache] Saved record cache to: {cache_path}")
    return records, pid_meta
