# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from utils.checkpoint import load_resume_checkpoint

def _outcome_name(label):
    return "Poor" if int(label) == 1 else "Good"


def _hospital_name(pid_meta_item):
    hospital = str(pid_meta_item.get("hospital", "Unknown")).strip().upper()
    return hospital if hospital else "UNKNOWN"


def _hospital_outcome_strata(pid_meta, pids):
    """Build joint Hospital x Outcome strata for patient-level split."""
    strata = []
    for pid in pids:
        meta = pid_meta[pid]
        hospital = _hospital_name(meta)
        label = int(meta["label"])
        strata.append(f"{hospital}_{label}")
    return strata


def _label_strata(pid_meta, pids):
    return [int(pid_meta[pid]["label"]) for pid in pids]


def _can_stratify(strata):
    """sklearn stratified split needs at least two samples in every class."""
    counts = {}
    for item in strata:
        counts[item] = counts.get(item, 0) + 1
    return len(counts) > 1 and min(counts.values()) >= 2


def _split_distribution(name, pids, pid_meta):
    hospital_counter = {}
    outcome_counter = {}
    joint_counter = {}

    for pid in sorted(map(str, pids)):
        meta = pid_meta[pid]
        hospital = _hospital_name(meta)
        outcome = _outcome_name(meta["label"])
        hospital_counter[hospital] = hospital_counter.get(hospital, 0) + 1
        outcome_counter[outcome] = outcome_counter.get(outcome, 0) + 1
        joint_key = f"{hospital}_{outcome}"
        joint_counter[joint_key] = joint_counter.get(joint_key, 0) + 1

    print(f"[Split Distribution] {name} patients={len(pids)}")
    print(f"  Hospital: {dict(sorted(hospital_counter.items()))}")
    print(f"  Outcome: {dict(sorted(outcome_counter.items()))}")
    print(f"  Hospital x Outcome: {dict(sorted(joint_counter.items()))}")


def print_split_distribution(train_pids, val_pids, test_pids, pid_meta):
    _split_distribution("Train", train_pids, pid_meta)
    _split_distribution("Val", val_pids, pid_meta)
    _split_distribution("Test", test_pids, pid_meta)


def stratified_pid_split(pid_meta, train_size=SPLIT_TRAIN_SIZE, val_size=SPLIT_VAL_SIZE, random_state=42):
    """Patient-level split with Hospital x Outcome stratification when possible."""
    all_pids = sorted(map(str, pid_meta.keys()))
    val_ratio = val_size / (1.0 - train_size)

    joint_strata = _hospital_outcome_strata(pid_meta, all_pids)
    label_strata = _label_strata(pid_meta, all_pids)

    if _can_stratify(joint_strata):
        first_strata = joint_strata
        first_method = "hospital_outcome"
    elif _can_stratify(label_strata):
        first_strata = label_strata
        first_method = "outcome_only_fallback"
    else:
        first_strata = None
        first_method = "random_fallback"

    try:
        train_pids, holdout_pids = train_test_split(
            all_pids,
            train_size=train_size,
            random_state=random_state,
            stratify=first_strata,
        )
    except ValueError as exc:
        print(f"[Split Warning] First split stratification failed ({exc}); using random split.")
        train_pids, holdout_pids = train_test_split(
            all_pids,
            train_size=train_size,
            random_state=random_state,
            stratify=None,
        )
        first_method = "random_fallback_after_error"

    holdout_joint_strata = _hospital_outcome_strata(pid_meta, holdout_pids)
    holdout_label_strata = _label_strata(pid_meta, holdout_pids)

    if _can_stratify(holdout_joint_strata):
        second_strata = holdout_joint_strata
        second_method = "hospital_outcome"
    elif _can_stratify(holdout_label_strata):
        second_strata = holdout_label_strata
        second_method = "outcome_only_fallback"
    else:
        second_strata = None
        second_method = "random_fallback"

    try:
        val_pids, test_pids = train_test_split(
            holdout_pids,
            train_size=val_ratio,
            random_state=random_state,
            stratify=second_strata,
        )
    except ValueError as exc:
        print(f"[Split Warning] Val/Test stratification failed ({exc}); using random split.")
        val_pids, test_pids = train_test_split(
            holdout_pids,
            train_size=val_ratio,
            random_state=random_state,
            stratify=None,
        )
        second_method = "random_fallback_after_error"

    print(
        "[Split] Patient-level split method | "
        f"train/holdout={first_method} | val/test={second_method} | "
        f"train_size={train_size} val_size={val_size} test_size={1.0 - train_size - val_size:.2f} | "
        f"random_state={random_state}"
    )
    print_split_distribution(set(train_pids), set(val_pids), set(test_pids), pid_meta)

    return set(train_pids), set(val_pids), set(test_pids)


def split_from_resume_if_compatible(resume_state, pid_meta):
    """Return saved split only if it exactly matches the ECG-available cohort."""
    if not SPLIT_REUSE_STAGE1_RESUME:
        if resume_state is not None:
            print(
                "[Split] Stage-1 resume split reuse is disabled; using the configured "
                "Hospital x Outcome split instead."
            )
        return None
    if resume_state is None:
        return None
    required_keys = {"train_pids", "val_pids", "test_pids"}
    if not required_keys.issubset(resume_state.keys()):
        return None

    train_pids = set(map(str, resume_state["train_pids"]))
    val_pids = set(map(str, resume_state["val_pids"]))
    test_pids = set(map(str, resume_state["test_pids"]))
    saved_union = train_pids | val_pids | test_pids
    current_union = set(pid_meta.keys())

    if saved_union != current_union:
        print(
            "[Resume] Existing checkpoint split is incompatible with the current "
            "ECG-available cohort. Ignoring saved split and creating a new split. "
            f"saved={len(saved_union)} current={len(current_union)}"
        )
        return None

    return train_pids, val_pids, test_pids


def load_or_create_split(pid_meta_ecg: Dict[str, dict], device: torch.device):
    if SPLIT_REUSE_EXTERNAL_RESUME and os.path.exists(RESUME_CHECKPOINT_PATH):
        try:
            state = load_resume_checkpoint(RESUME_CHECKPOINT_PATH, device)
            if state is not None and all(k in state for k in ("train_pids", "val_pids", "test_pids")):
                train_pids = set(map(str, state["train_pids"]))
                val_pids = set(map(str, state["val_pids"]))
                test_pids = set(map(str, state["test_pids"]))
                current_pids = set(map(str, pid_meta_ecg.keys()))
                if (train_pids | val_pids | test_pids) == current_pids:
                    print(f"[Split] Reusing split from {RESUME_CHECKPOINT_PATH}")
                    return train_pids, val_pids, test_pids, "random12_resume_checkpoint"
        except Exception as exc:
            print(f"[Warning] Failed to reuse split: {exc}")
    elif os.path.exists(RESUME_CHECKPOINT_PATH):
        print(
            "[Split] External resume split reuse is disabled; creating a fresh "
            "Hospital x Outcome stratified split for the current ECG-available cohort."
        )

    train_pids, val_pids, test_pids = stratified_pid_split(
        pid_meta_ecg,
        train_size=SPLIT_TRAIN_SIZE,
        val_size=SPLIT_VAL_SIZE,
        random_state=BASE_SEED,
    )
    print("[Split] Created new ECG-available Hospital x Outcome split with BASE_SEED.")
    return (
        set(map(str, train_pids)),
        set(map(str, val_pids)),
        set(map(str, test_pids)),
        "ecg_hospital_outcome_seed42",
    )
