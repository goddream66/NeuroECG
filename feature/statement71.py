# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from data.datasets import ICAREDataset
from evaluation.metrics import compute_feature_quantile
from utils.runtime import _gpu_clean_feature_tensor, _gpu_to_numpy, release_memory

def load_statement_classes() -> List[str]:
    if not os.path.exists(PTBXL_MLB_PATH):
        raise FileNotFoundError(f"mlb.pkl not found: {PTBXL_MLB_PATH}")
    with open(PTBXL_MLB_PATH, "rb") as f:
        mlb = pickle.load(f)
    classes = list(getattr(mlb, "classes_", []))
    if len(classes) != 71:
        raise RuntimeError(f"Expected 71 PTB-XL ALL-STATEMENTS classes, got {len(classes)}")
    print(f"[PTBXL71] Loaded mlb.pkl | n_classes={len(classes)} | first10={classes[:10]}")
    return classes


def load_fastai_xresnet_model(device: torch.device) -> nn.Module:
    if not os.path.exists(PTBXL_PTH_PATH):
        raise FileNotFoundError(f"PTBXL_PTH_PATH not found: {PTBXL_PTH_PATH}")

    from ptbxl.repository import import_xresnet1d101

    try:
        xresnet1d101 = import_xresnet1d101()
    except Exception as exc:
        raise RuntimeError(
            "Failed to import extracted xresnet1d101 from model.ptbxl_xresnet1d."
        ) from exc

    model = xresnet1d101(
        num_classes=71,
        input_channels=12,
        kernel_size=5,
        ps_head=0.5,
        lin_ftrs_head=[128],
    )

    # PyTorch 2.6 changed torch.load's default weights_only value to True.
    # This PTB-XL checkpoint contains numpy scalar metadata, so load it with
    # the pre-2.6 behavior. Only use this for trusted local checkpoints.
    state = torch.load(PTBXL_PTH_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = OrderedDict()
    for key, value in state.items():
        new_key = str(key)
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        new_state[new_key] = value

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[PTBXL71] Loaded xresnet1d101 weights from: {PTBXL_PTH_PATH}")
    print(f"[PTBXL71] load_state_dict strict=False | missing={len(missing)} | unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[PTBXL71] first missing keys: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[PTBXL71] first unexpected keys: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


def ecg_tensor_to_single_lead(ecg_tensor: torch.Tensor, channel_mask: torch.Tensor = None) -> np.ndarray:
    """Convert [B, C, L] ECG to one valid single-lead signal per sample.

    If channel_mask is provided, choose the valid channel with the largest
    segment standard deviation.  This keeps PTB-XL statement extraction aligned
    with the flatline-only channel QC used by the deep ECG encoder.  If the mask
    is unavailable, fall back to channel 0 for backward compatibility.
    """
    x = ecg_tensor.detach().cpu().numpy().astype(np.float32)
    mask_np = None
    if channel_mask is not None:
        mask_np = channel_mask.detach().cpu().numpy().astype(np.float32)

    if x.ndim == 3:
        if x.shape[1] <= 12:
            if mask_np is not None and mask_np.ndim == 2:
                b, c, _ = x.shape
                max_c = min(c, mask_np.shape[1], 2)
                sig_rows = []
                for i in range(b):
                    valid = [j for j in range(max_c) if mask_np[i, j] > 0.5]
                    if not valid:
                        valid = [0]
                    stds = [float(np.std(x[i, j])) for j in valid]
                    best = valid[int(np.argmax(stds))]
                    sig_rows.append(x[i, best, :])
                sig = np.stack(sig_rows, axis=0)
            else:
                sig = x[:, 0, :]
        else:
            sig = x[:, :, 0]
    elif x.ndim == 2:
        sig = x
    else:
        raise ValueError(f"Unexpected ECG tensor shape: {x.shape}")
    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_12lead_single_input(single_lead: np.ndarray) -> np.ndarray:
    single_lead = np.asarray(single_lead, dtype=np.float32)
    if single_lead.ndim != 2:
        raise ValueError("single_lead must have shape (B, L)")
    if single_lead.shape[1] != PTBXL_TARGET_LEN:
        single_lead = resample(single_lead, PTBXL_TARGET_LEN, axis=1).astype(np.float32)
    b = single_lead.shape[0]
    x12 = np.zeros((b, PTBXL_TARGET_LEN, PTBXL_N_LEADS), dtype=np.float32)
    if PTBXL_FEED_MODE == "leadII_zero":
        x12[:, :, 1] = single_lead
    elif PTBXL_FEED_MODE == "repeat12":
        x12[:, :, :] = single_lead[:, :, None]
    else:
        raise ValueError(f"Unknown PTBXL_FEED_MODE: {PTBXL_FEED_MODE}")
    return x12


def crop_12lead(x12: np.ndarray) -> Tuple[np.ndarray, int]:
    starts = list(range(0, PTBXL_TARGET_LEN - PTBXL_CROP_LEN + 1, PTBXL_CROP_STRIDE))
    if not starts:
        starts = [0]
    crops = []
    for s in starts:
        crop = x12[:, s:s + PTBXL_CROP_LEN, :]          # B, crop_len, 12
        crops.append(np.transpose(crop, (0, 2, 1)))     # B, 12, crop_len
    stacked = np.concatenate(crops, axis=0).astype(np.float32)
    return stacked, len(starts)


def select_single_lead_tensor(ecg: torch.Tensor, channel_mask: torch.Tensor = None) -> torch.Tensor:
    """Select the valid single ECG channel on GPU."""
    if ecg.ndim == 2:
        return torch.nan_to_num(ecg.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if ecg.ndim != 3:
        raise ValueError(f"Unexpected ECG tensor shape: {tuple(ecg.shape)}")
    if ecg.shape[1] > 12:
        single = ecg[:, :, 0]
        return torch.nan_to_num(single.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if channel_mask is None:
        single = ecg[:, 0, :]
        return torch.nan_to_num(single.float(), nan=0.0, posinf=0.0, neginf=0.0)

    mask = channel_mask.to(device=ecg.device, dtype=ecg.dtype)
    if mask.ndim != 2 or mask.shape[0] != ecg.shape[0]:
        single = ecg[:, 0, :]
        return torch.nan_to_num(single.float(), nan=0.0, posinf=0.0, neginf=0.0)

    max_c = min(ecg.shape[1], mask.shape[1], 2)
    if max_c <= 0:
        single = ecg[:, 0, :]
        return torch.nan_to_num(single.float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = mask[:, :max_c] > 0.5
    stds = ecg[:, :max_c, :].float().std(dim=-1)
    scores = stds.masked_fill(~valid_mask, -1.0)
    no_valid = ~valid_mask.any(dim=1)
    if bool(no_valid.any()):
        scores[no_valid, 0] = stds[no_valid, 0]
    best_idx = torch.argmax(scores, dim=1)
    batch_idx = torch.arange(ecg.shape[0], device=ecg.device)
    single = ecg[batch_idx, best_idx, :]
    return torch.nan_to_num(single.float(), nan=0.0, posinf=0.0, neginf=0.0)


def prepare_ptbxl_crops_tensor(ecg: torch.Tensor, channel_mask: torch.Tensor = None) -> Tuple[torch.Tensor, int]:
    """Prepare PTB-XL 12-lead crops fully on GPU."""
    single = select_single_lead_tensor(ecg, channel_mask=channel_mask)
    if single.shape[-1] != PTBXL_TARGET_LEN:
        single = F.interpolate(
            single.unsqueeze(1),
            size=PTBXL_TARGET_LEN,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    if PTBXL_FEED_MODE == "leadII_zero":
        x12 = torch.zeros(
            (single.shape[0], PTBXL_N_LEADS, PTBXL_TARGET_LEN),
            device=single.device,
            dtype=single.dtype,
        )
        x12[:, 1, :] = single
    elif PTBXL_FEED_MODE == "repeat12":
        x12 = single.unsqueeze(1).repeat(1, PTBXL_N_LEADS, 1)
    else:
        raise ValueError(f"Unknown PTBXL_FEED_MODE: {PTBXL_FEED_MODE}")

    starts = list(range(0, PTBXL_TARGET_LEN - PTBXL_CROP_LEN + 1, PTBXL_CROP_STRIDE))
    if not starts:
        starts = [0]
    crops = [x12[:, :, s:s + PTBXL_CROP_LEN] for s in starts]
    return torch.cat(crops, dim=0), len(starts)


def predict_statement71_probabilities(model: nn.Module, ecg: torch.Tensor, device: torch.device, channel_mask: torch.Tensor = None) -> np.ndarray:
    ecg = ecg.to(device, non_blocking=True).float()
    channel_mask = channel_mask.to(device, non_blocking=True).float() if channel_mask is not None else None
    crops, n_crops = prepare_ptbxl_crops_tensor(ecg, channel_mask=channel_mask)
    logits = model(crops)
    b = ecg.shape[0]
    probs = torch.sigmoid(logits).reshape(n_crops, b, -1).permute(1, 0, 2).contiguous()
    if PTBXL_CROP_AGG == "max":
        probs = probs.amax(dim=1)
        return _gpu_to_numpy(probs)
    if PTBXL_CROP_AGG == "mean":
        probs = probs.mean(dim=1)
        return _gpu_to_numpy(probs)
    raise ValueError(f"Unknown PTBXL_CROP_AGG: {PTBXL_CROP_AGG}")


def extract_patient_statement71_and_hrv(model: nn.Module, samples: Sequence[dict], split_name: str, device: torch.device, batch_size: int, num_workers: int):
    dataset = ICAREDataset(samples, is_train=False, clip_value=INPUT_CLIP_VALUE, compute_hrv=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=DATALOADER_PIN_MEMORY)
    grouped_entry_statement: Dict[str, Dict[str, List[np.ndarray]]] = {}
    grouped_hrv: Dict[str, List[np.ndarray]] = {}
    t0 = time.time()
    n_segments = 0
    cursor = 0

    for batch_idx, (ecg, channel_mask, labels, cpcs, _, pids_batch, _, _, hrv_batch) in enumerate(loader, start=1):
        probs = predict_statement71_probabilities(model, ecg, device, channel_mask=channel_mask)
        hrv_np = hrv_batch.detach().cpu().numpy().astype(np.float32)
        pids = [str(pid) for pid in pids_batch]
        batch_len = len(pids)
        batch_samples = samples[cursor:cursor + batch_len]
        cursor += batch_len

        for sample, pid, prob_row, hrv_row in zip(batch_samples, pids, probs, hrv_np):
            entry_id = str(sample.get("entry_id", f"{pid}|single|{sample.get('start', 0)}"))
            grouped_entry_statement.setdefault(pid, {}).setdefault(entry_id, []).append(prob_row.astype(np.float32))
            grouped_hrv.setdefault(pid, []).append(hrv_row.astype(np.float32))

        n_segments += batch_len
        del ecg, channel_mask, labels, cpcs, hrv_batch, probs, hrv_np, pids, batch_samples
        if batch_idx % 20 == 0:
            print(f"[{split_name}] processed segments={n_segments}/{len(samples)} | time={(time.time() - t0) / 60:.1f} min")
    print(f"[{split_name}] PTBXL-71 WVNUM extraction done: segments={n_segments} | time={(time.time() - t0) / 60:.1f} min")
    del loader, dataset
    release_memory(f"{split_name} PTBXL extraction loader")
    return grouped_entry_statement, grouped_hrv


def aggregate_entry_statement_rows(statement_rows: np.ndarray) -> np.ndarray:
    """Aggregate WVNUM segment-level statement probabilities into one entry-level feature."""
    statement_tensor = _gpu_clean_feature_tensor(statement_rows)
    if statement_tensor.ndim != 2 or statement_tensor.shape[0] == 0:
        raise ValueError("statement_rows must be a non-empty 2D array")
    mean_vec = statement_tensor.mean(dim=0)
    if ENTRY_STATEMENT_AGG_MODE == "mean":
        return _gpu_to_numpy(mean_vec)
    if ENTRY_STATEMENT_AGG_MODE == "mean_q88":
        q88_vec = torch.quantile(statement_tensor, 0.88, dim=0)
        return _gpu_to_numpy(torch.cat([mean_vec, q88_vec], dim=0))
    if ENTRY_STATEMENT_AGG_MODE == "mean_std":
        std_vec = statement_tensor.std(dim=0, unbiased=False)
        return _gpu_to_numpy(torch.cat([mean_vec, std_vec], dim=0))
    if ENTRY_STATEMENT_AGG_MODE == "mean_q88_std":
        q88_vec = torch.quantile(statement_tensor, 0.88, dim=0)
        std_vec = statement_tensor.std(dim=0, unbiased=False)
        return _gpu_to_numpy(torch.cat([mean_vec, q88_vec, std_vec], dim=0))
    raise ValueError(f"Unknown ENTRY_STATEMENT_AGG_MODE: {ENTRY_STATEMENT_AGG_MODE}")


def aggregate_patient_entries(entry_features: np.ndarray) -> np.ndarray:
    """Aggregate entry-level statement features into one patient-level feature."""
    entry_tensor = _gpu_clean_feature_tensor(entry_features)
    if entry_tensor.ndim != 2 or entry_tensor.shape[0] == 0:
        raise ValueError("entry_features must be a non-empty 2D array")
    if PATIENT_ENTRY_AGG_MODE == "mean":
        return _gpu_to_numpy(entry_tensor.mean(dim=0))
    raise ValueError(f"Unknown PATIENT_ENTRY_AGG_MODE: {PATIENT_ENTRY_AGG_MODE}")


def build_patient_blocks_from_grouped(
    grouped_entry_statement: Dict[str, Dict[str, List[np.ndarray]]],
    grouped_hrv: Dict[str, List[np.ndarray]],
    pid_meta: Dict[str, dict],
    split_name: str,
    pooling_strategy="quantile",
    quantile_val=0.88,
) -> Dict[str, object]:
    patient_ids = sorted(grouped_entry_statement.keys())
    static_rows, hrv_rows, statement_rows, y_rows, cpc_rows, kept_pids = [], [], [], [], [], []
    n_entries_total = 0
    for pid in patient_ids:
        entry_dict = grouped_entry_statement.get(pid, {})
        entry_features = []
        for entry_id in sorted(entry_dict.keys()):
            rows = np.asarray(entry_dict[entry_id], dtype=np.float32)
            if rows.ndim != 2 or rows.shape[0] == 0:
                continue
            entry_features.append(aggregate_entry_statement_rows(rows))
        hrv_mat = np.asarray(grouped_hrv.get(pid, []), dtype=np.float32)
        if len(entry_features) == 0 or hrv_mat.ndim != 2 or hrv_mat.shape[0] == 0:
            continue
        entry_features = np.vstack(entry_features).astype(np.float32)
        statement_summary = aggregate_patient_entries(entry_features)
        if pooling_strategy == "mean":
            hrv_summary = np.mean(hrv_mat, axis=0)
        elif pooling_strategy == "max":
            hrv_summary = np.max(hrv_mat, axis=0)
        else:
            hrv_summary = compute_feature_quantile(hrv_mat, quantile_val)
        static_rows.append(np.asarray(pid_meta[pid]["static"], dtype=np.float32))
        statement_rows.append(statement_summary.astype(np.float32))
        hrv_rows.append(hrv_summary.astype(np.float32))
        y_rows.append(int(pid_meta[pid]["label"]))
        cpc_rows.append(float(pid_meta[pid]["cpc"]))
        kept_pids.append(pid)
        n_entries_total += int(entry_features.shape[0])
    blocks = {
        "static": np.vstack(static_rows).astype(np.float32),
        "hrv": np.vstack(hrv_rows).astype(np.float32),
        "statement": np.vstack(statement_rows).astype(np.float32),
        "y": np.asarray(y_rows, dtype=np.int64),
        "cpc": np.asarray(cpc_rows, dtype=np.float32),
        "patient_ids": kept_pids,
    }
    print(
        f"  [Aggregate] {split_name}: patients={len(kept_pids)} | entries={n_entries_total} | "
        f"static_dim={blocks['static'].shape[1]} | hrv_dim={blocks['hrv'].shape[1]} "
        f"{pooling_strategy}{f'_q{quantile_val:.2f}' if pooling_strategy == 'quantile' else ''} | "
        f"semantic_dim={blocks['statement'].shape[1]} | entry_mode={ENTRY_STATEMENT_AGG_MODE} | "
        f"patient_mode={PATIENT_ENTRY_AGG_MODE}"
    )
    return blocks


def build_patient_blocks_from_entry_stats(
    grouped_entry_stats: Dict[str, Dict[str, List[object]]],
    grouped_hrv: Dict[str, List[np.ndarray]],
    pid_meta: Dict[str, dict],
    split_name: str,
    pooling_strategy="quantile",
    quantile_val=0.88,
) -> Dict[str, object]:
    """Build patient blocks from per-entry statement sums instead of per-segment rows."""
    if ENTRY_STATEMENT_AGG_MODE != "mean" or PATIENT_ENTRY_AGG_MODE != "mean":
        raise ValueError("Entry-stat streaming is only exact for mean entry and patient aggregation.")

    patient_ids = sorted(grouped_entry_stats.keys())
    static_rows, hrv_rows, statement_rows, y_rows, cpc_rows, kept_pids = [], [], [], [], [], []
    n_entries_total = 0
    for pid in patient_ids:
        entry_dict = grouped_entry_stats.get(pid, {})
        entry_features = []
        for entry_id in sorted(entry_dict.keys()):
            sum_vec, count = entry_dict[entry_id]
            count = int(count)
            if count <= 0:
                continue
            entry_features.append((np.asarray(sum_vec, dtype=np.float32) / float(count)).astype(np.float32))

        hrv_mat = np.asarray(grouped_hrv.get(pid, []), dtype=np.float32)
        if len(entry_features) == 0 or hrv_mat.ndim != 2 or hrv_mat.shape[0] == 0:
            continue
        entry_features = np.vstack(entry_features).astype(np.float32)
        statement_summary = aggregate_patient_entries(entry_features)
        if pooling_strategy == "mean":
            hrv_summary = np.mean(hrv_mat, axis=0)
        elif pooling_strategy == "max":
            hrv_summary = np.max(hrv_mat, axis=0)
        else:
            hrv_summary = compute_feature_quantile(hrv_mat, quantile_val)
        static_rows.append(np.asarray(pid_meta[pid]["static"], dtype=np.float32))
        statement_rows.append(statement_summary.astype(np.float32))
        hrv_rows.append(hrv_summary.astype(np.float32))
        y_rows.append(int(pid_meta[pid]["label"]))
        cpc_rows.append(float(pid_meta[pid]["cpc"]))
        kept_pids.append(pid)
        n_entries_total += int(entry_features.shape[0])

    blocks = {
        "static": np.vstack(static_rows).astype(np.float32),
        "hrv": np.vstack(hrv_rows).astype(np.float32),
        "statement": np.vstack(statement_rows).astype(np.float32),
        "y": np.asarray(y_rows, dtype=np.int64),
        "cpc": np.asarray(cpc_rows, dtype=np.float32),
        "patient_ids": kept_pids,
    }
    print(
        f"  [Aggregate-LowMem] {split_name}: patients={len(kept_pids)} | entries={n_entries_total} | "
        f"static_dim={blocks['static'].shape[1]} | hrv_dim={blocks['hrv'].shape[1]} "
        f"{pooling_strategy}{f'_q{quantile_val:.2f}' if pooling_strategy == 'quantile' else ''} | "
        f"semantic_dim={blocks['statement'].shape[1]} | entry_mode={ENTRY_STATEMENT_AGG_MODE} | "
        f"patient_mode={PATIENT_ENTRY_AGG_MODE}"
    )
    return blocks


def extract_patient_statement71_block_lowmem(
    model: nn.Module,
    samples: Sequence[dict],
    split_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pid_meta: Dict[str, dict],
    pooling_strategy="quantile",
    quantile_val=0.88,
) -> Dict[str, object]:
    """Extract statement71 features and aggregate a split before moving to the next split."""
    if ENTRY_STATEMENT_AGG_MODE != "mean" or PATIENT_ENTRY_AGG_MODE != "mean":
        grouped_statement, grouped_hrv = extract_patient_statement71_and_hrv(
            model, samples, split_name, device, batch_size, num_workers
        )
        block = build_patient_blocks_from_grouped(
            grouped_statement, grouped_hrv, pid_meta, split_name,
            pooling_strategy=pooling_strategy, quantile_val=quantile_val
        )
        del grouped_statement, grouped_hrv
        release_memory(f"{split_name} grouped statement/HRV fallback")
        return block

    dataset = ICAREDataset(samples, is_train=False, clip_value=INPUT_CLIP_VALUE, compute_hrv=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=DATALOADER_PIN_MEMORY)
    grouped_entry_stats: Dict[str, Dict[str, List[object]]] = {}
    grouped_hrv: Dict[str, List[np.ndarray]] = {}
    t0 = time.time()
    n_segments = 0
    cursor = 0

    for batch_idx, (ecg, channel_mask, labels, cpcs, _, pids_batch, _, _, hrv_batch) in enumerate(loader, start=1):
        probs = predict_statement71_probabilities(model, ecg, device, channel_mask=channel_mask)
        hrv_np = hrv_batch.detach().cpu().numpy().astype(np.float32)
        pids = [str(pid) for pid in pids_batch]
        batch_len = len(pids)
        batch_samples = samples[cursor:cursor + batch_len]
        cursor += batch_len

        for sample, pid, prob_row, hrv_row in zip(batch_samples, pids, probs, hrv_np):
            entry_id = str(sample.get("entry_id", f"{pid}|single|{sample.get('start', 0)}"))
            entry_dict = grouped_entry_stats.setdefault(pid, {})
            stat = entry_dict.get(entry_id)
            if stat is None:
                stat = [np.zeros(probs.shape[1], dtype=np.float32), 0]
                entry_dict[entry_id] = stat
            stat[0] += np.asarray(prob_row, dtype=np.float32)
            stat[1] = int(stat[1]) + 1
            grouped_hrv.setdefault(pid, []).append(np.asarray(hrv_row, dtype=np.float32))

        n_segments += batch_len
        del ecg, channel_mask, labels, cpcs, hrv_batch, probs, hrv_np, pids, batch_samples
        if batch_idx % 20 == 0:
            print(f"[{split_name}] processed segments={n_segments}/{len(samples)} | time={(time.time() - t0) / 60:.1f} min")

    print(f"[{split_name}] PTBXL-71 low-memory WVNUM extraction done: segments={n_segments} | time={(time.time() - t0) / 60:.1f} min")
    del loader, dataset
    release_memory(f"{split_name} PTBXL extraction loader")
    block = build_patient_blocks_from_entry_stats(
        grouped_entry_stats, grouped_hrv, pid_meta, split_name,
        pooling_strategy=pooling_strategy, quantile_val=quantile_val
    )
    del grouped_entry_stats, grouped_hrv
    release_memory(f"{split_name} streamed statement/HRV stats")
    return block
