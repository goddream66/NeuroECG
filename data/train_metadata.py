# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *

def _safe_float(raw_value, default):
    try:
        return float(raw_value.strip())
    except (TypeError, ValueError, AttributeError):
        return default


def _parse_bool_text(raw_value):
    value = str(raw_value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return 1.0
    if value in {"false", "0", "no", "n"}:
        return 0.0
    return None


def _parse_ttm_value(raw_value, default=np.nan):
    value = str(raw_value).strip()
    if not value:
        return default
    if value.lower() == "nan":
        return np.nan
    return _safe_float(value, default)


def load_patient_metadata(root_dir):
    pid_meta = {}
    skipped_records = 0
    folders = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )

    print(f"[Info] Loading patient metadata from: {root_dir}")
    for pid in folders:
        p_path = os.path.join(root_dir, pid)
        txt_path = os.path.join(p_path, f"{pid}.txt")
        if not os.path.exists(txt_path):
            skipped_records += 1
            continue

        static_features = {
            "age_years": 60.0,
            "sex_male": 0.0,
            "rosc_minutes": 20.0,
            "ohca": 0.0,
            "shockable_rhythm": 0.0,
            "ttm_celsius": np.nan,
        }
        label = None
        cpc = None
        hospital = "Unknown"

        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError:
            skipped_records += 1
            continue

        if "Outcome: Poor" in content:
            label = 1
        elif "Outcome: Good" in content:
            label = 0
        else:
            skipped_records += 1
            continue

        for line in content.splitlines():
            line = line.strip()
            if ":" not in line:
                continue

            field_name, raw_value = line.split(":", 1)
            field_name = field_name.strip().lower()
            raw_value = raw_value.strip()

            if field_name == "cpc":
                cpc = _safe_float(line.split(":")[-1], cpc)
            elif field_name == "age":
                static_features["age_years"] = _safe_float(
                    raw_value, static_features["age_years"]
                )
            elif field_name == "hospital":
                hospital = raw_value
            elif field_name == "sex":
                if raw_value.lower() == "male":
                    static_features["sex_male"] = 1.0
                elif raw_value.lower() == "female":
                    static_features["sex_male"] = 0.0
            elif field_name == "rosc":
                static_features["rosc_minutes"] = _safe_float(
                    raw_value, static_features["rosc_minutes"]
                )
            elif field_name == "ohca":
                parsed = _parse_bool_text(raw_value)
                if parsed is not None:
                    static_features["ohca"] = parsed
            elif field_name in {"shockable rhythm", "shockable_rhythm", "shockable"}:
                parsed = _parse_bool_text(raw_value)
                if parsed is not None:
                    static_features["shockable_rhythm"] = parsed
            elif field_name == "ttm":
                static_features["ttm_celsius"] = _parse_ttm_value(
                    raw_value, static_features["ttm_celsius"]
                )

        if cpc is None:
            cpc = 5.0 if label == 1 else 1.0

        pid_meta[pid] = {
            "static": np.array(
                [static_features[name] for name in STATIC_FEATURE_NAMES],
                dtype=np.float32,
            ),
            "static_feature_names": STATIC_FEATURE_NAMES,
            "label": label,
            "cpc": float(cpc),
            "hospital": hospital,
        }

    print(f"[Done] Loaded metadata for {len(pid_meta)} patients | Skipped records: {skipped_records}")
    return pid_meta


def infer_pid_from_processed_path(file_path, processed_root, pid_meta):
    rel_path = os.path.relpath(file_path, processed_root)
    path_parts = rel_path.split(os.sep)

    if path_parts and path_parts[0] in pid_meta:
        return path_parts[0]

    stem = os.path.splitext(os.path.basename(file_path))[0]
    candidate_tokens = stem.replace("-", "_").split("_")
    for token in candidate_tokens:
        if token in pid_meta:
            return token

    for part in path_parts:
        if part in pid_meta:
            return part

    return None


def filter_metadata_to_ecg_available(pid_meta, records):
    """Keep only patients that actually have usable processed ECG records.

    The raw metadata may contain patients without ECG files. For an ECG model,
    the correct cohort is the intersection between labeled metadata patients and
    patient IDs appearing in the processed record cache. Splitting should happen
    after this filtering step, otherwise logs show 607 metadata patients but only
    409 aggregated ECG patients.
    """
    metadata_pids = set(pid_meta.keys())
    ecg_available_pids = set(str(record["pid"]) for record in records)
    usable_pids = metadata_pids & ecg_available_pids
    missing_pids = metadata_pids - ecg_available_pids

    filtered_meta = {
        pid: meta
        for pid, meta in pid_meta.items()
        if pid in usable_pids
    }

    print(
        "[ECG Cohort] Metadata patients: "
        f"{len(metadata_pids)} | ECG-available patients: {len(filtered_meta)} | "
        f"Excluded without usable ECG: {len(missing_pids)}"
    )
    if missing_pids:
        print(f"[ECG Cohort] First 20 excluded patient IDs: {sorted(missing_pids)[:20]}")

    return filtered_meta, usable_pids, missing_pids


def filter_pid_meta_to_ecg_available(pid_meta: Dict[str, dict], all_records: Sequence[dict]) -> Dict[str, dict]:
    ecg_available_pids = set(str(record["pid"]) for record in all_records)
    pid_meta_ecg = {str(pid): meta for pid, meta in pid_meta.items() if str(pid) in ecg_available_pids}
    print(
        "[ECG Availability] "
        f"metadata patients={len(pid_meta)} | ECG-available patients={len(pid_meta_ecg)} | "
        f"excluded={len(pid_meta) - len(pid_meta_ecg)}"
    )
    return pid_meta_ecg


