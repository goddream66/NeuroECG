# Patient-level gated fusion for selected modalities, with optional extra concat.
from pipeline.train_config import *
from evaluation.metrics import compute_binary_operating_metrics, compute_challenge_score_safe
from utils.runtime import get_gpu_device, release_memory, set_seed


GATED_FUSION_BASE_MODALITIES = ("deep", "static")


def fit_feature_standardizers(
    train_block: Dict[str, object],
    feature_names: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    standardizers = {}
    for name in feature_names:
        matrix = np.nan_to_num(
            np.asarray(train_block[name], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mean = matrix.mean(axis=0).astype(np.float32)
        std = np.maximum(matrix.std(axis=0), 1e-6).astype(np.float32)
        standardizers[name] = (mean, std)
    return standardizers


class PatientGatedFusionDataset(Dataset):
    def __init__(
        self,
        block: Dict[str, object],
        standardizers: Dict[str, Tuple[np.ndarray, np.ndarray]],
        gated_modalities: Sequence[str] = GATED_FUSION_BASE_MODALITIES,
        extra_modalities: Sequence[str] = (),
    ):
        self.gated_modalities = tuple(gated_modalities)
        self.extra_modalities = tuple(extra_modalities)
        self.modality_names = self.gated_modalities + self.extra_modalities
        self.features = {}
        for name in self.modality_names:
            matrix = np.nan_to_num(
                np.asarray(block[name], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            mean, std = standardizers[name]
            self.features[name] = np.nan_to_num(
                (matrix - mean.reshape(1, -1)) / std.reshape(1, -1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
        self.y = np.asarray(block["y"], dtype=np.int64)
        self.cpc = np.nan_to_num(
            np.asarray(block["cpc"], dtype=np.float32),
            nan=0.0,
            posinf=5.0,
            neginf=0.0,
        )
        self.patient_ids = list(map(str, block["patient_ids"]))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return (
            {name: torch.from_numpy(self.features[name][idx]).float() for name in self.modality_names},
            torch.tensor(float(self.y[idx]), dtype=torch.float32),
            torch.tensor(float(self.cpc[idx]), dtype=torch.float32),
            self.patient_ids[idx],
        )


def patient_gated_fusion_collate_fn(batch):
    feature_rows, labels, cpcs, pids = zip(*batch)
    modality_names = tuple(feature_rows[0].keys())
    features = {
        name: torch.stack([row[name] for row in feature_rows]).float()
        for name in modality_names
    }
    return features, torch.stack(labels).float(), torch.stack(cpcs).float(), list(pids)


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, align_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(align_dim)),
            nn.LayerNorm(int(align_dim)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(align_dim), int(align_dim)),
            nn.LayerNorm(int(align_dim)),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0))


class DeepStaticGatedFusionModel(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        gated_modalities: Sequence[str] = GATED_FUSION_BASE_MODALITIES,
        extra_modalities: Sequence[str] = (),
        align_dim: int = FUSION_ALIGN_DIM,
        hidden_dim: int = FUSION_HIDDEN_DIM,
        dropout: float = FUSION_DROPOUT,
        fixed_gate_weights: Dict[str, float] = None,
    ):
        super().__init__()
        self.gated_modalities = tuple(gated_modalities)
        self.extra_modalities = tuple(extra_modalities)
        self.uses_fixed_gate = fixed_gate_weights is not None
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(input_dims[name], align_dim, dropout)
                for name in self.gated_modalities
            }
        )
        if fixed_gate_weights is None:
            self.gate_logits = nn.Parameter(torch.zeros(len(self.gated_modalities), dtype=torch.float32))
        else:
            weights = torch.tensor(
                [float(fixed_gate_weights[name]) for name in self.gated_modalities],
                dtype=torch.float32,
            )
            weights = weights / torch.clamp(weights.sum(), min=1e-6)
            self.gate_logits = None
            self.register_buffer("fixed_gate_weights", weights)
        extra_dim = sum(int(input_dims[name]) for name in self.extra_modalities)
        fused_dim = int(align_dim) + int(extra_dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, int(hidden_dim)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.outcome_head = nn.Linear(int(hidden_dim), 1)
        self.cpc_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: Dict[str, torch.Tensor]):
        projected = [self.projectors[name](features[name]) for name in self.gated_modalities]
        tokens = torch.stack(projected, dim=1)
        if self.uses_fixed_gate:
            weights = self.fixed_gate_weights.to(device=tokens.device, dtype=tokens.dtype)
        else:
            weights = torch.softmax(self.gate_logits, dim=0)
        weights = weights.view(1, len(self.gated_modalities)).expand(tokens.shape[0], -1)
        self._last_gate_weights = weights.detach()
        weights = weights.view(tokens.shape[0], len(self.gated_modalities), 1)
        fused = torch.sum(tokens * weights, dim=1)
        if self.extra_modalities:
            extra = torch.cat(
                [
                    torch.nan_to_num(features[name].float(), nan=0.0, posinf=0.0, neginf=0.0)
                    for name in self.extra_modalities
                ],
                dim=1,
            )
            fused = torch.cat([fused, extra], dim=1)
        hidden = self.shared(fused)
        logits = self.outcome_head(hidden).squeeze(1)
        cpc_pred = self.cpc_head(hidden).squeeze(1)
        return (
            torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(cpc_pred, nan=0.0, posinf=5.0, neginf=0.0),
        )

    def gate_weights(self) -> Dict[str, float]:
        if self.uses_fixed_gate:
            weights = self.fixed_gate_weights.detach().cpu().numpy()
        else:
            weights = torch.softmax(self.gate_logits.detach().cpu(), dim=0).numpy()
        return {name: float(weights[i]) for i, name in enumerate(self.gated_modalities)}


@torch.no_grad()
def evaluate_gated_fusion_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pid_meta: Dict[str, dict],
) -> Dict[str, float]:
    model.eval()
    probs_all, preds_all, y_all, cpc_all, cpc_pred_all, pids_all = [], [], [], [], [], []
    for features, labels, cpcs, pids in loader:
        features = {name: value.to(device, non_blocking=True).float() for name, value in features.items()}
        logits, cpc_pred = model(features)
        probs = torch.sigmoid(logits).clamp(0.0, 1.0)
        preds = (probs >= 0.5).long()
        probs_all.append(probs.detach().cpu().numpy())
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(labels.numpy())
        cpc_all.append(cpcs.numpy())
        cpc_pred_all.append(cpc_pred.detach().cpu().numpy())
        pids_all.extend(pids)

    y = np.concatenate(y_all).astype(int)
    probs = np.nan_to_num(np.concatenate(probs_all).astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    preds = np.concatenate(preds_all).astype(int)
    cpc = np.nan_to_num(np.concatenate(cpc_all).astype(float), nan=0.0, posinf=5.0, neginf=0.0)
    cpc_pred = np.nan_to_num(np.concatenate(cpc_pred_all).astype(float), nan=0.0, posinf=5.0, neginf=0.0)

    out = {"n": int(len(y))}
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, probs))
        out["auprc"] = float(average_precision_score(y, probs))
        out["challenge_score"] = float(compute_challenge_score_safe(y, probs, pids_all, pid_meta))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
        out["challenge_score"] = np.nan
    out["accuracy"] = float(accuracy_score(y, preds))
    out["f1"] = float(f1_score(y, preds, zero_division=0))
    out.update(compute_binary_operating_metrics(y, preds))
    out["cpc_mse"] = float(mean_squared_error(cpc, cpc_pred))
    out["cpc_mae"] = float(mean_absolute_error(cpc, cpc_pred))
    return out


def train_and_eval_deep_static_gated_fusion(
    variant: str,
    representation: str,
    gated_modalities: Sequence[str],
    extra_modalities: Sequence[str],
    seed: int,
    blocks: Dict[str, Dict[str, object]],
    pid_meta: Dict[str, dict],
    fixed_gate_weights: Dict[str, float] = None,
) -> Dict[str, object]:
    set_seed(seed)
    device = get_gpu_device("patient-level gated fusion")
    gated_modalities = tuple(gated_modalities)
    extra_modalities = tuple(extra_modalities)
    feature_names = gated_modalities + extra_modalities

    standardizers = fit_feature_standardizers(blocks["train"], feature_names)
    datasets = {
        split: PatientGatedFusionDataset(
            blocks[split],
            standardizers,
            gated_modalities=gated_modalities,
            extra_modalities=extra_modalities,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=FUSION_BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
            collate_fn=patient_gated_fusion_collate_fn,
        )
        for split, dataset in datasets.items()
    }
    input_dims = {
        name: int(np.asarray(blocks["train"][name]).shape[1])
        for name in feature_names
    }
    model = DeepStaticGatedFusionModel(
        input_dims,
        gated_modalities=gated_modalities,
        extra_modalities=extra_modalities,
        fixed_gate_weights=fixed_gate_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=FUSION_WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    cpc_loss_fn = nn.SmoothL1Loss()
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    best_state = None
    best_score = -np.inf
    best_epoch = -1
    patience_left = int(FUSION_PATIENCE)

    for epoch in range(int(FUSION_EPOCHS)):
        model.train()
        epoch_losses = []
        for features, labels, cpcs, _ in loaders["train"]:
            features = {name: value.to(device, non_blocking=True).float() for name, value in features.items()}
            labels = labels.to(device, non_blocking=True).float().clamp(0.0, 1.0)
            cpcs = cpcs.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits, cpc_pred = model(features)
            sample_weights = class_weights[labels.long().clamp(0, 1)]
            outcome_loss = (bce(logits, labels) * sample_weights).mean()
            cpc_loss = cpc_loss_fn(cpc_pred, cpcs)
            loss = outcome_loss + float(FUSION_CPC_LOSS_WEIGHT) * cpc_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate_gated_fusion_model(model, loaders["val"], device, pid_meta)
        monitor = float(val_metrics.get("auroc", np.nan))
        if not np.isfinite(monitor):
            monitor = float(val_metrics.get("auprc", -np.inf))
        improved = monitor > best_score + 1e-6
        if improved:
            best_score = monitor
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(FUSION_PATIENCE)
        else:
            patience_left -= 1
        if improved or (epoch + 1) % 10 == 0 or patience_left <= 0:
            gates = model.gate_weights()
            gate_names = "/".join(gated_modalities)
            gate_values = "/".join(f"{gates[name]:.3f}" for name in gated_modalities)
            print(
                f"  [GatedFusion] {variant} seed={seed} epoch={epoch + 1}/{FUSION_EPOCHS} | "
                f"loss={(np.mean(epoch_losses) if epoch_losses else np.nan):.4f} | "
                f"val_AUROC={val_metrics.get('auroc', np.nan):.4f} | "
                f"val_AUPRC={val_metrics.get('auprc', np.nan):.4f} | "
                f"gates {gate_names}={gate_values} | "
                f"best_epoch={best_epoch + 1}"
            )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    gates = model.gate_weights()
    feature_dim = int(FUSION_ALIGN_DIM) + sum(input_dims[name] for name in extra_modalities)
    row = {
        "variant": variant,
        "representation": representation,
        "seed": int(seed),
        "feature_dim": int(feature_dim),
        "train_n": int(len(datasets["train"])),
        "val_n": int(len(datasets["val"])),
        "test_n": int(len(datasets["test"])),
        "best_epoch": int(best_epoch + 1),
    }
    for name, value in gates.items():
        row[f"fusion_gate_{name}"] = value
    for split_name in ("train", "val", "test"):
        metrics = evaluate_gated_fusion_model(model, loaders[split_name], device, pid_meta)
        for key, value in metrics.items():
            row[f"{split_name}_{key}"] = value
    print(
        f"  [Result] {variant:34s} rep={representation:42s} seed={seed} | "
        f"dim={feature_dim:4d} | test AUROC={row['test_auroc']:.4f}, "
        f"AUPRC={row['test_auprc']:.4f}, Acc={row['test_accuracy']:.4f}, "
        f"Spec={row['test_specificity']:.4f}, CPC_MAE={row['test_cpc_mae']:.4f}"
    )
    del model, optimizer, datasets, loaders, best_state
    release_memory(f"{variant} gated fusion")
    return row


def train_eval_deep_static_gated_fusion(
    blocks: Dict[str, Dict[str, object]],
    pid_meta: Dict[str, dict],
) -> List[Dict[str, object]]:
    if not USE_DEEP_STATIC_GATED_FUSION:
        return []
    deep_pca_key = f"deep_pca{DEEP_PCA_DIM}"
    fixed_deep_static_gate = {
        "deep": float(FIXED_GATE_DEEP_WEIGHT),
        "static": float(FIXED_GATE_STATIC_WEIGHT),
    }
    specs = [
        (
            "gated_deep_static",
            f"deep_static_projected{FUSION_ALIGN_DIM}_softmax_gated",
            ("deep", "static"),
            (),
            None,
        ),
        (
            "gated_deep_static_hrv_statement71",
            f"deep_static_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_hrv_statement71",
            ("deep", "static"),
            ("hrv", "statement"),
            None,
        ),
        (
            "gated_deep_statement71_static_hrv",
            f"deep_statement71_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_static_hrv",
            ("deep", "statement"),
            ("static", "hrv"),
            None,
        ),
        (
            "gated_deep_pca64_static",
            f"{deep_pca_key}_static_projected{FUSION_ALIGN_DIM}_softmax_gated",
            (deep_pca_key, "static"),
            (),
            None,
        ),
        (
            "gated_deep_pca64_static_hrv_statement71",
            f"{deep_pca_key}_static_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_hrv_statement71",
            (deep_pca_key, "static"),
            ("hrv", "statement"),
            None,
        ),
        (
            "fixedgate_deep_static",
            f"deep_static_projected{FUSION_ALIGN_DIM}_fixedgate_d{FIXED_GATE_DEEP_WEIGHT:.3f}_s{FIXED_GATE_STATIC_WEIGHT:.3f}",
            ("deep", "static"),
            (),
            fixed_deep_static_gate,
        ),
        (
            "fixedgate_deep_static_hrv_statement71",
            f"deep_static_projected{FUSION_ALIGN_DIM}_fixedgate_d{FIXED_GATE_DEEP_WEIGHT:.3f}_s{FIXED_GATE_STATIC_WEIGHT:.3f}_plus_hrv_statement71",
            ("deep", "static"),
            ("hrv", "statement"),
            fixed_deep_static_gate,
        ),
    ]
    rows = []
    for variant, representation, gated_modalities, extra_modalities, fixed_gate_weights in specs:
        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx
            print(f"\n[Fusion Ablation] variant={variant} | representation={representation} | seed={seed}")
            rows.append(
                train_and_eval_deep_static_gated_fusion(
                    variant,
                    representation,
                    gated_modalities,
                    extra_modalities,
                    seed,
                    blocks,
                    pid_meta,
                    fixed_gate_weights=fixed_gate_weights,
                )
            )
    return rows


class ClinicalResidualDeepFusionModel(nn.Module):
    """Clinical base plus patient-specific deep residual correction.

    clinical_logit = f(static, hrv, statement71)
    deep_delta = g(deep)
    alpha = sigmoid(h(clinical_hidden, deep_hidden))
    final_logit = clinical_logit + alpha * deep_delta
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        deep_modality: str,
        clinical_modalities: Sequence[str] = ("static", "hrv", "statement"),
        hidden_dim: int = FUSION_HIDDEN_DIM,
        dropout: float = FUSION_DROPOUT,
    ):
        super().__init__()
        self.deep_modality = str(deep_modality)
        self.clinical_modalities = tuple(clinical_modalities)
        clinical_dim = sum(int(input_dims[name]) for name in self.clinical_modalities)
        deep_dim = int(input_dims[self.deep_modality])
        hidden_dim = int(hidden_dim)

        self.clinical_net = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.deep_net = nn.Sequential(
            nn.LayerNorm(deep_dim),
            nn.Linear(deep_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.clinical_outcome_head = nn.Linear(hidden_dim, 1)
        self.deep_delta_head = nn.Linear(hidden_dim, 1)
        self.alpha_net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.cpc_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Dict[str, torch.Tensor], return_alpha: bool = False):
        clinical = torch.cat(
            [
                torch.nan_to_num(features[name].float(), nan=0.0, posinf=0.0, neginf=0.0)
                for name in self.clinical_modalities
            ],
            dim=1,
        )
        deep = torch.nan_to_num(features[self.deep_modality].float(), nan=0.0, posinf=0.0, neginf=0.0)

        clinical_hidden = self.clinical_net(clinical)
        deep_hidden = self.deep_net(deep)
        clinical_logit = self.clinical_outcome_head(clinical_hidden).squeeze(1)
        deep_delta = self.deep_delta_head(deep_hidden).squeeze(1)
        alpha = self.alpha_net(torch.cat([clinical_hidden, deep_hidden], dim=1)).squeeze(1)
        logits = clinical_logit + alpha * deep_delta

        cpc_input = torch.cat([clinical_hidden, alpha.unsqueeze(1) * deep_hidden], dim=1)
        cpc_pred = self.cpc_head(cpc_input).squeeze(1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        cpc_pred = torch.nan_to_num(cpc_pred, nan=0.0, posinf=5.0, neginf=0.0)
        if return_alpha:
            return logits, cpc_pred, torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        return logits, cpc_pred


@torch.no_grad()
def evaluate_clinical_residual_fusion_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pid_meta: Dict[str, dict],
) -> Dict[str, float]:
    model.eval()
    probs_all, preds_all, y_all, cpc_all, cpc_pred_all, alpha_all, pids_all = [], [], [], [], [], [], []
    for features, labels, cpcs, pids in loader:
        features = {name: value.to(device, non_blocking=True).float() for name, value in features.items()}
        logits, cpc_pred, alpha = model(features, return_alpha=True)
        probs = torch.sigmoid(logits).clamp(0.0, 1.0)
        preds = (probs >= 0.5).long()
        probs_all.append(probs.detach().cpu().numpy())
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(labels.numpy())
        cpc_all.append(cpcs.numpy())
        cpc_pred_all.append(cpc_pred.detach().cpu().numpy())
        alpha_all.append(alpha.detach().cpu().numpy())
        pids_all.extend(pids)

    y = np.concatenate(y_all).astype(int)
    probs = np.nan_to_num(np.concatenate(probs_all).astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    preds = np.concatenate(preds_all).astype(int)
    cpc = np.nan_to_num(np.concatenate(cpc_all).astype(float), nan=0.0, posinf=5.0, neginf=0.0)
    cpc_pred = np.nan_to_num(np.concatenate(cpc_pred_all).astype(float), nan=0.0, posinf=5.0, neginf=0.0)
    alpha = np.nan_to_num(np.concatenate(alpha_all).astype(float), nan=0.0, posinf=1.0, neginf=0.0)

    out = {"n": int(len(y))}
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, probs))
        out["auprc"] = float(average_precision_score(y, probs))
        out["challenge_score"] = float(compute_challenge_score_safe(y, probs, pids_all, pid_meta))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
        out["challenge_score"] = np.nan
    out["accuracy"] = float(accuracy_score(y, preds))
    out["f1"] = float(f1_score(y, preds, zero_division=0))
    out.update(compute_binary_operating_metrics(y, preds))
    out["cpc_mse"] = float(mean_squared_error(cpc, cpc_pred))
    out["cpc_mae"] = float(mean_absolute_error(cpc, cpc_pred))
    out["alpha_mean"] = float(np.mean(alpha))
    out["alpha_std"] = float(np.std(alpha, ddof=0))
    return out


def train_and_eval_clinical_residual_deep_fusion(
    variant: str,
    representation: str,
    deep_modality: str,
    seed: int,
    blocks: Dict[str, Dict[str, object]],
    pid_meta: Dict[str, dict],
) -> Dict[str, object]:
    set_seed(seed)
    device = get_gpu_device("clinical residual deep fusion")
    clinical_modalities = ("static", "hrv", "statement")
    feature_names = clinical_modalities + (str(deep_modality),)
    standardizers = fit_feature_standardizers(blocks["train"], feature_names)
    datasets = {
        split: PatientGatedFusionDataset(
            blocks[split],
            standardizers,
            gated_modalities=(str(deep_modality),),
            extra_modalities=clinical_modalities,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=FUSION_BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
            collate_fn=patient_gated_fusion_collate_fn,
        )
        for split, dataset in datasets.items()
    }
    input_dims = {
        name: int(np.asarray(blocks["train"][name]).shape[1])
        for name in feature_names
    }
    model = ClinicalResidualDeepFusionModel(
        input_dims,
        deep_modality=str(deep_modality),
        clinical_modalities=clinical_modalities,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=FUSION_WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    cpc_loss_fn = nn.SmoothL1Loss()
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    best_state = None
    best_score = -np.inf
    best_epoch = -1
    patience_left = int(FUSION_PATIENCE)

    for epoch in range(int(FUSION_EPOCHS)):
        model.train()
        epoch_losses = []
        for features, labels, cpcs, _ in loaders["train"]:
            features = {name: value.to(device, non_blocking=True).float() for name, value in features.items()}
            labels = labels.to(device, non_blocking=True).float().clamp(0.0, 1.0)
            cpcs = cpcs.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits, cpc_pred = model(features)
            sample_weights = class_weights[labels.long().clamp(0, 1)]
            outcome_loss = (bce(logits, labels) * sample_weights).mean()
            cpc_loss = cpc_loss_fn(cpc_pred, cpcs)
            loss = outcome_loss + float(FUSION_CPC_LOSS_WEIGHT) * cpc_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate_clinical_residual_fusion_model(model, loaders["val"], device, pid_meta)
        monitor = float(val_metrics.get("auroc", np.nan))
        if not np.isfinite(monitor):
            monitor = float(val_metrics.get("auprc", -np.inf))
        improved = monitor > best_score + 1e-6
        if improved:
            best_score = monitor
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(FUSION_PATIENCE)
        else:
            patience_left -= 1
        if improved or (epoch + 1) % 10 == 0 or patience_left <= 0:
            print(
                f"  [ClinicalResidual] {variant} seed={seed} epoch={epoch + 1}/{FUSION_EPOCHS} | "
                f"loss={(np.mean(epoch_losses) if epoch_losses else np.nan):.4f} | "
                f"val_AUROC={val_metrics.get('auroc', np.nan):.4f} | "
                f"val_AUPRC={val_metrics.get('auprc', np.nan):.4f} | "
                f"alpha={val_metrics.get('alpha_mean', np.nan):.3f}+/-{val_metrics.get('alpha_std', np.nan):.3f} | "
                f"best_epoch={best_epoch + 1}"
            )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    feature_dim = int(sum(input_dims[name] for name in feature_names))
    row = {
        "variant": variant,
        "representation": representation,
        "seed": int(seed),
        "feature_dim": int(feature_dim),
        "train_n": int(len(datasets["train"])),
        "val_n": int(len(datasets["val"])),
        "test_n": int(len(datasets["test"])),
        "best_epoch": int(best_epoch + 1),
    }
    for split_name in ("train", "val", "test"):
        metrics = evaluate_clinical_residual_fusion_model(model, loaders[split_name], device, pid_meta)
        for key, value in metrics.items():
            row[f"{split_name}_{key}"] = value
    print(
        f"  [Result] {variant:34s} rep={representation:42s} seed={seed} | "
        f"dim={feature_dim:4d} | test AUROC={row['test_auroc']:.4f}, "
        f"AUPRC={row['test_auprc']:.4f}, Acc={row['test_accuracy']:.4f}, "
        f"Spec={row['test_specificity']:.4f}, CPC_MAE={row['test_cpc_mae']:.4f}, "
        f"alpha={row['test_alpha_mean']:.3f}+/-{row['test_alpha_std']:.3f}"
    )
    del model, optimizer, datasets, loaders, best_state
    release_memory(f"{variant} clinical residual fusion")
    return row


def train_eval_clinical_residual_deep_fusion(
    blocks: Dict[str, Dict[str, object]],
    pid_meta: Dict[str, dict],
) -> List[Dict[str, object]]:
    if not USE_CLINICAL_RESIDUAL_DEEP_FUSION:
        return []
    deep_pca_key = f"deep_pca{DEEP_PCA_DIM}"
    specs = [
        (
            "clinical_residual_deep_pca64",
            "clinical_static_hrv_statement71_plus_patient_residual_deep_pca64",
            deep_pca_key,
        ),
        (
            "clinical_residual_deep1024",
            "clinical_static_hrv_statement71_plus_patient_residual_deep1024",
            "deep",
        ),
    ]
    rows = []
    for variant, representation, deep_modality in specs:
        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx
            print(f"\n[Fusion Ablation] variant={variant} | representation={representation} | seed={seed}")
            rows.append(
                train_and_eval_clinical_residual_deep_fusion(
                    variant,
                    representation,
                    deep_modality,
                    seed,
                    blocks,
                    pid_meta,
                )
            )
    return rows
