# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from data.datasets import ICAREDataset
from evaluation.metrics import compute_feature_quantile
from sklearn.decomposition import PCA
from utils.runtime import gpu_concat_feature_blocks, gpu_concat_feature_vectors, release_memory

def get_feats_fast(
    extractor,
    projector,
    dataset,
    indices,
    device,
    amp_enabled,
    num_workers,
    batch_size=1024,
):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DATALOADER_PIN_MEMORY,
    )
    all_features = []
    all_labels = []
    all_cpcs = []
    all_pids = []
    all_time_buckets = []
    all_segment_end_hours = []
    all_hrv_features = []

    extractor.eval()
    projector.eval()
    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (ecg, channel_mask, labels, cpcs, _, pids_batch, time_buckets_batch, segment_end_hours_batch, hrv_batch) in enumerate(loader, start=1):
            with autocast(enabled=amp_enabled):
                features = projector(extractor(ecg.to(device).float(), channel_mask=channel_mask.to(device).float()))
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_cpcs.append(cpcs.numpy())
            all_pids.extend(pids_batch)
            all_time_buckets.extend(time_buckets_batch)
            all_segment_end_hours.append(segment_end_hours_batch.numpy())
            all_hrv_features.append(hrv_batch.numpy())
            del ecg, channel_mask, labels, cpcs, features, pids_batch, time_buckets_batch, segment_end_hours_batch, hrv_batch
            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                print(f"    [Deep Extract] batch={batch_idx}/{total_batches}")

    features_np = np.vstack(all_features)
    labels_np = np.concatenate(all_labels)
    cpcs_np = np.concatenate(all_cpcs)
    segment_end_hours_np = np.concatenate(all_segment_end_hours)
    hrv_features_np = np.vstack(all_hrv_features)
    del all_features, all_labels, all_cpcs, all_segment_end_hours, all_hrv_features
    release_memory("segment feature extraction buffers")
    return (
        features_np,
        labels_np,
        cpcs_np,
        all_pids,
        all_time_buckets,
        segment_end_hours_np,
        hrv_features_np,
    )


def aggregate_patient_features(
    unique_pids,
    features,
    hrv_features,
    segment_pids,
    segment_time_buckets,
    segment_end_hours,
    pid_meta,
    split_name,
    time_bucket_order=None,
    active_segment_buckets=None,
    pooling_strategy="quantile",
    quantile_val=None,
    hrv_pooling_strategy=None,
    hrv_quantile_val=None,
):
    """Build patient-level features WITHOUT time buckets.

    Difference from the original time-aware version:
    - All ECG segments from the same patient are pooled together globally.
    - No 0-12h / 12-24h / 24-48h / 48-72h / >72h bucket features are used.
    - No min/max/mean/std segment-end-hour statistics are used.
    - No bucket count vector is used.

    Default feature vector:
        [6 static clinical covariates]
        + [10 globally pooled HRV features]
        + [globally pooled ECGFounder deep embedding]

    Optional:
        + [log1p(total segment count)] if GLOBAL_AGG_INCLUDE_COUNT=True

    hrv_pooling_strategy can be set separately when q-search should only affect
    deep embeddings while HRV stays clinically interpretable, e.g. mean HRV.
    """
    del segment_time_buckets, time_bucket_order, active_segment_buckets
    quantile_val = TEMPORAL_AGG_QUANTILE if quantile_val is None else quantile_val
    hrv_pooling_strategy = pooling_strategy if hrv_pooling_strategy is None else hrv_pooling_strategy
    hrv_quantile_val = quantile_val if hrv_quantile_val is None else hrv_quantile_val

    patient_features = {pid: [] for pid in unique_pids}
    patient_segment_hrv_features = {pid: [] for pid in unique_pids}

    for feature_row, hrv_row, pid in zip(features, hrv_features, segment_pids):
        if pid not in patient_features:
            continue
        patient_features[pid].append(feature_row)
        patient_segment_hrv_features[pid].append(hrv_row)
    del segment_end_hours

    X = []
    y_outcome = []
    y_cpc = []
    kept_pids = []

    feature_dim = int(features.shape[1])
    hrv_dim = int(hrv_features.shape[1])

    for pid in unique_pids:
        deep_rows = np.asarray(patient_features[pid], dtype=np.float32)
        hrv_rows = np.asarray(patient_segment_hrv_features[pid], dtype=np.float32)

        if deep_rows.ndim != 2 or deep_rows.shape[0] == 0:
            continue

        if hrv_rows.ndim != 2 or hrv_rows.shape[0] == 0:
            hrv_summary = np.zeros(hrv_dim, dtype=np.float32)
        else:
            if hrv_pooling_strategy == "mean":
                hrv_summary = np.mean(hrv_rows, axis=0)
            elif hrv_pooling_strategy == "max":
                hrv_summary = np.max(hrv_rows, axis=0)
            else:
                hrv_summary = compute_feature_quantile(hrv_rows, hrv_quantile_val)

        if pooling_strategy == "mean":
            deep_summary = np.mean(deep_rows, axis=0)
        elif pooling_strategy == "max":
            deep_summary = np.max(deep_rows, axis=0)
        else:
            deep_summary = compute_feature_quantile(deep_rows, quantile_val)
        meta = pid_meta[pid]

        blocks = [
            meta["static"].astype(np.float32),
            hrv_summary.astype(np.float32),
            deep_summary.astype(np.float32),
        ]

        if GLOBAL_AGG_INCLUDE_COUNT:
            blocks.append(np.asarray([np.log1p(float(deep_rows.shape[0]))], dtype=np.float32))

        summary = gpu_concat_feature_vectors(blocks)
        X.append(summary)
        y_outcome.append(meta["label"])
        y_cpc.append(meta["cpc"])
        kept_pids.append(pid)

    X = np.asarray(X, dtype=np.float32)
    print(
        f"  [Aggregate-NoTimeBucket] {split_name}: reduced {len(kept_pids)} patients | "
        f"Deep pooling: {pooling_strategy} (q={quantile_val:.2f}) | "
        f"HRV pooling: {hrv_pooling_strategy}"
        f"{f' (q={hrv_quantile_val:.2f})' if hrv_pooling_strategy == 'quantile' else ''} | "
        f"Feature dim: {X.shape[1] if X.ndim == 2 and len(X) else 0} "
        f"(static=6, hrv={hrv_dim}, deep={feature_dim}, "
        f"count={'on' if GLOBAL_AGG_INCLUDE_COUNT else 'off'})"
    )
    return X, np.asarray(y_outcome), np.asarray(y_cpc), kept_pids


def build_deep_patient_block(
    X: np.ndarray,
    y: np.ndarray,
    cpc: np.ndarray,
    patient_ids: Sequence[str],
    split_name: str,
) -> Dict[str, object]:
    static_dim = len(STATIC_FEATURE_NAMES)
    hrv_dim = len(HRV_FEATURE_NAMES)
    static = X[:, :static_dim].astype(np.float32)
    hrv = X[:, static_dim:static_dim + hrv_dim].astype(np.float32)
    deep_end = -1 if GLOBAL_AGG_INCLUDE_COUNT else X.shape[1]
    deep = X[:, static_dim + hrv_dim:deep_end].astype(np.float32)
    block = {
        "static": static,
        "hrv": hrv,
        "deep": deep,
        "y": np.asarray(y, dtype=np.int64),
        "cpc": np.asarray(cpc, dtype=np.float32),
        "patient_ids": list(map(str, patient_ids)),
    }
    print(
        f"  [DeepBlock] {split_name}: patients={len(block['patient_ids'])} | "
        f"static_dim={static.shape[1]} | hrv_dim={hrv.shape[1]} | deep_dim={deep.shape[1]}"
    )
    return block


def extract_deep_patient_block_for_split(
    extractor: nn.Module,
    projector: nn.Module,
    samples: Sequence[dict],
    split_name: str,
    device: torch.device,
    amp_enabled: bool,
    num_workers: int,
    batch_size: int,
    pid_meta: Dict[str, dict],
    pooling_strategy="quantile",
    quantile_val=None,
    hrv_pooling_strategy=None,
    hrv_quantile_val=None,
) -> Dict[str, object]:
    feature_ds = ICAREDataset(samples, is_train=False, clip_value=INPUT_CLIP_VALUE, compute_hrv=True)
    features, _, _, segment_pids, time_buckets, segment_end_hours, hrv_features = get_feats_fast(
        extractor,
        projector,
        feature_ds,
        list(range(len(samples))),
        device,
        amp_enabled,
        num_workers,
        batch_size=batch_size,
    )
    pid_list = sorted(set(segment_pids))
    X, y, cpc, patient_ids = aggregate_patient_features(
        pid_list,
        features,
        hrv_features,
        segment_pids,
        time_buckets,
        segment_end_hours,
        pid_meta,
        split_name,
        MODEL_TIME_BUCKETS,
        pooling_strategy=pooling_strategy,
        quantile_val=quantile_val,
        hrv_pooling_strategy=hrv_pooling_strategy,
        hrv_quantile_val=hrv_quantile_val,
    )
    block = build_deep_patient_block(
        X,
        y,
        cpc,
        patient_ids,
        split_name,
    )
    del (
        feature_ds,
        features,
        hrv_features,
        segment_pids,
        time_buckets,
        segment_end_hours,
        pid_list,
        X,
        y,
        cpc,
        patient_ids,
    )
    release_memory(f"Stage 2A {split_name} segment matrices")
    return block


def merge_deep_and_statement_blocks(
    deep_block: Dict[str, object],
    statement_block: Dict[str, object],
    split_name: str,
) -> Dict[str, object]:
    deep_pids = list(map(str, deep_block["patient_ids"]))
    statement_idx = {str(pid): idx for idx, pid in enumerate(statement_block["patient_ids"])}
    keep = [(i, statement_idx[pid], pid) for i, pid in enumerate(deep_pids) if pid in statement_idx]
    if not keep:
        raise RuntimeError(f"No overlapping patients between deep and statement blocks for split={split_name}")
    deep_idx = [item[0] for item in keep]
    stmt_idx = [item[1] for item in keep]
    keep_pids = [item[2] for item in keep]
    merged = {
        "static": np.asarray(deep_block["static"])[deep_idx].astype(np.float32),
        "hrv": np.asarray(deep_block["hrv"])[deep_idx].astype(np.float32),
        "deep": np.asarray(deep_block["deep"])[deep_idx].astype(np.float32),
        "statement": np.asarray(statement_block["statement"])[stmt_idx].astype(np.float32),
        "y": np.asarray(deep_block["y"])[deep_idx].astype(np.int64),
        "cpc": np.asarray(deep_block["cpc"])[deep_idx].astype(np.float32),
        "patient_ids": keep_pids,
    }
    print(
        f"  [Merge] {split_name}: deep_patients={len(deep_pids)} | "
        f"statement_patients={len(statement_block['patient_ids'])} | kept={len(keep_pids)}"
    )
    return merged


def add_deep_pca_feature_blocks(
    blocks: Dict[str, Dict[str, object]],
    n_components: int = DEEP_PCA_DIM,
) -> Dict[str, object]:
    """Fit PCA on train deep features and transform train/val/test.

    PCA is fit only on the training split to avoid leaking validation/test
    information into the feature space.
    """
    train_deep = np.nan_to_num(
        np.asarray(blocks["train"]["deep"], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    requested_dim = int(n_components)
    actual_dim = min(requested_dim, int(train_deep.shape[0]), int(train_deep.shape[1]))
    if actual_dim <= 0:
        raise ValueError("Cannot fit deep PCA on an empty deep feature matrix.")

    mean = train_deep.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train_deep.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    train_z = np.nan_to_num((train_deep - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=actual_dim, svd_solver="full", random_state=BASE_SEED)
    pca.fit(train_z)

    key = f"deep_pca{requested_dim}"
    for split_name in ("train", "val", "test"):
        deep = np.nan_to_num(
            np.asarray(blocks[split_name]["deep"], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        deep_z = np.nan_to_num((deep - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        blocks[split_name][key] = np.nan_to_num(
            pca.transform(deep_z),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        print(
            f"  [DeepPCA] {split_name}: deep_dim={deep.shape[1]} -> "
            f"{key}_dim={blocks[split_name][key].shape[1]}"
        )

    info = {
        "enabled": True,
        "source": "deep",
        "method": "train_split_standardized_pca",
        "requested_dim": requested_dim,
        "actual_dim": actual_dim,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "feature_key": key,
        "fit_scope": "train_only",
    }
    del train_deep, train_z, mean, std, pca
    release_memory("deep PCA feature blocks")
    return info


def _make_aux_feature_matrix(block: Dict[str, object]) -> np.ndarray:
    return np.concatenate(
        [
            np.nan_to_num(np.asarray(block["static"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(np.asarray(block["hrv"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(np.asarray(block["statement"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        ],
        axis=1,
    ).astype(np.float32)


def add_aux_projected_feature_blocks(
    blocks: Dict[str, Dict[str, object]],
    out_dim: int = AUX_FEATURE_PROJECTOR_DIM,
    seed: int = AUX_FEATURE_PROJECTOR_SEED,
) -> Dict[str, object]:
    """Project [static, HRV, statement71] to 1024 dims, then concatenate with deep."""
    train_aux = _make_aux_feature_matrix(blocks["train"])
    mean = train_aux.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train_aux.std(axis=0, keepdims=True), 1e-6).astype(np.float32)

    in_dim = int(train_aux.shape[1])
    out_dim = int(out_dim)
    rng = np.random.default_rng(int(seed))
    weight = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    weight /= np.sqrt(float(max(in_dim, 1)))

    for split_name in ("train", "val", "test"):
        aux = _make_aux_feature_matrix(blocks[split_name])
        aux_z = np.nan_to_num((aux - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        aux_projected = np.nan_to_num(aux_z @ weight, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        blocks[split_name]["aux_projected"] = aux_projected
        blocks[split_name]["deep_aux_projected"] = gpu_concat_feature_blocks(
            [
                np.nan_to_num(np.asarray(blocks[split_name]["deep"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                aux_projected,
            ]
        )
        print(
            f"  [AuxProjector] {split_name}: static+hrv+statement_dim={in_dim} -> "
            f"aux_projected_dim={out_dim}; deep+aux_dim={blocks[split_name]['deep_aux_projected'].shape[1]}"
        )

    info = {
        "enabled": True,
        "source": "static_plus_hrv_plus_statement71",
        "projector": "fixed_random_linear",
        "fit_scope": "train_standardization_only",
        "input_dim": in_dim,
        "output_dim": out_dim,
        "seed": int(seed),
        "deep_aux_projected_dim": int(blocks["train"]["deep_aux_projected"].shape[1]),
    }
    del train_aux, mean, std, weight
    release_memory("aux feature projector")
    return info
