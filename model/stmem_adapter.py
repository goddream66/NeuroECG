# ST-MEM stage-1 ECG feature extractor.
from pipeline.train_config import *


def _load_stmem_encoder():
    if not os.path.isdir(ST_MEM_ROOT):
        raise FileNotFoundError(f"ST-MEM repository not found: {ST_MEM_ROOT}")
    if not os.path.exists(ST_MEM_CKPT):
        raise FileNotFoundError(f"ST-MEM checkpoint not found: {ST_MEM_CKPT}")

    root = str(ST_MEM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from models.encoder.st_mem_vit import st_mem_vit_base
    except Exception as exc:
        raise ImportError(
            "ST-MEM requires its repository on PYTHONPATH and dependencies such as einops. "
            "Check ST_MEM_ROOT or install ST-MEM requirements before using STAGE1_BACKBONE='stmem'."
        ) from exc

    model = st_mem_vit_base(
        num_leads=int(ST_MEM_N_LEADS),
        num_classes=None,
        seq_len=int(ST_MEM_TARGET_LEN),
        patch_size=int(ST_MEM_PATCH_SIZE),
    )
    checkpoint = torch.load(ST_MEM_CKPT, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model")
            or checkpoint.get("state_dict")
            or checkpoint.get("encoder")
            or checkpoint
        )
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported ST-MEM checkpoint format: {type(checkpoint)!r}")

    cleaned = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in ("module.", "encoder."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    msg = model.load_state_dict(cleaned, strict=False)
    print(
        f"[ST-MEM] Loaded encoder checkpoint: {ST_MEM_CKPT} | "
        f"missing={len(msg.missing_keys)} | unexpected={len(msg.unexpected_keys)}"
    )
    return model


def _encoder_blocks(extractor):
    encoder = getattr(extractor, "encoder", None)
    depth = int(getattr(encoder, "depth", 0) or 0)
    blocks = []
    for idx in range(depth):
        block = getattr(encoder, f"block{idx}", None)
        if block is not None:
            blocks.append(block)
    return blocks


def _params_from_modules(modules):
    return [p for module in modules for p in module.parameters()]


class STMEMFeatureExtractor(nn.Module):
    """ST-MEM wrapper for current single-lead I-CARE segments.

    ST-MEM expects [B, 12, 2250].  The existing I-CARE cache provides [B, 1, 5000],
    so this adapter resamples inside the model wrapper and places the single
    input channel into the configured 12-lead slot while zero-filling the rest.
    """

    def __init__(self):
        super().__init__()
        self.encoder = _load_stmem_encoder()
        self.feature_dim = int(ST_MEM_FEATURE_DIM)
        self.target_len = int(ST_MEM_TARGET_LEN)
        self.n_leads = int(ST_MEM_N_LEADS)
        self.single_lead_index = int(ST_MEM_SINGLE_LEAD_INDEX)

    def _to_12lead(self, x):
        if x.ndim != 3:
            raise RuntimeError(f"Expected ST-MEM input [B, 1, T], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            x = x[:, :1, :]
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        out = x.new_zeros((x.shape[0], self.n_leads, x.shape[-1]))
        lead_idx = max(0, min(self.single_lead_index, self.n_leads - 1))
        out[:, lead_idx:lead_idx + 1, :] = x
        return out

    def forward(self, x, channel_mask=None):
        del channel_mask
        source = self._to_12lead(x.float())
        features = self.encoder(source)
        return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def set_stmem_gradual_unfreezing(extractor, classifier_head, epoch):
    for p in extractor.parameters():
        p.requires_grad = False
    for p in classifier_head.parameters():
        p.requires_grad = True

    trainable_parts = ["classifier_head"]
    blocks = _encoder_blocks(extractor)
    if epoch >= int(ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH):
        for p in extractor.encoder.parameters():
            p.requires_grad = True
        trainable_parts.append("full_encoder")
    elif blocks and epoch >= int(ECGFOUNDER_UNFREEZE_LAST_FOUR_STAGES_EPOCH):
        for block in blocks[-4:]:
            for p in block.parameters():
                p.requires_grad = True
        trainable_parts.append("blocks[-4:]")
    elif blocks and epoch >= int(ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH):
        for block in blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        trainable_parts.append("blocks[-2:]")

    extractor.train() if any(p.requires_grad for p in extractor.parameters()) else extractor.eval()
    classifier_head.train()
    return "+".join(trainable_parts)


def make_stmem_optimizer(extractor, classifier_head, weight_decay):
    blocks = _encoder_blocks(extractor)
    last_two_params = _params_from_modules(blocks[-2:]) if blocks else []
    last_two_param_ids = {id(p) for p in last_two_params}
    last_four_extra_params = _params_from_modules(blocks[-4:-2]) if len(blocks) >= 4 else []
    last_four_extra_param_ids = {id(p) for p in last_four_extra_params}
    rest_params = [
        p for p in extractor.encoder.parameters()
        if id(p) not in last_two_param_ids and id(p) not in last_four_extra_param_ids
    ]

    param_groups = [
        {"params": classifier_head.parameters(), "lr": float(ECGFOUNDER_HEAD_LR), "name": "icare_dense_head"},
    ]
    if last_two_params:
        param_groups.append({
            "params": last_two_params,
            "lr": float(ECGFOUNDER_LAST_TWO_STAGES_LR),
            "name": "stmem_last_two_blocks",
        })
    if last_four_extra_params:
        param_groups.append({
            "params": last_four_extra_params,
            "lr": float(ECGFOUNDER_LAST_FOUR_STAGES_LR),
            "name": "stmem_last_four_extra_blocks",
        })
    if rest_params:
        param_groups.append({
            "params": rest_params,
            "lr": float(ECGFOUNDER_FULL_BACKBONE_LR),
            "name": "stmem_full_encoder_rest",
        })
    return optim.AdamW(param_groups, weight_decay=weight_decay)
