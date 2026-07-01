# ECG-FM stage-1 ECG feature extractor.
from pipeline.train_config import *


def _load_ecgfm_model():
    if not os.path.exists(ECGFM_CKPT):
        raise FileNotFoundError(f"ECG-FM checkpoint not found: {ECGFM_CKPT}")
    try:
        from fairseq_signals.models import build_model_from_checkpoint
    except Exception as exc:
        raise ImportError(
            "ECG-FM requires fairseq_signals. Install/activate the ECG-FM environment "
            "or add fairseq-signals to PYTHONPATH before using STAGE1_BACKBONE='ecgfm'."
        ) from exc
    model = build_model_from_checkpoint(checkpoint_path=str(ECGFM_CKPT))
    print(f"[ECG-FM] Loaded checkpoint: {ECGFM_CKPT}")
    return model


def _find_block_sequence(module):
    candidates = []
    for name, child in module.named_modules():
        if isinstance(child, nn.ModuleList):
            blocks = list(child)
            param_count = sum(p.numel() for block in blocks for p in block.parameters())
            if len(blocks) >= 2 and param_count > 0:
                candidates.append((name, blocks, param_count))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (len(item[1]), item[2]), reverse=True)
    return candidates[0][1]


def _params_from_modules(modules):
    return [p for module in modules for p in module.parameters()]


def _pool_encoder_out(encoder_out, batch_size):
    if isinstance(encoder_out, (list, tuple)):
        encoder_out = encoder_out[0]
    if isinstance(encoder_out, dict):
        for key in ("encoder_out", "x", "features"):
            if key in encoder_out:
                encoder_out = encoder_out[key]
                break
    if not torch.is_tensor(encoder_out):
        raise RuntimeError(f"Unsupported ECG-FM encoder_out type: {type(encoder_out)!r}")

    if encoder_out.ndim == 2:
        return encoder_out
    if encoder_out.ndim != 3:
        raise RuntimeError(f"Expected ECG-FM encoder_out 2D/3D tensor, got {tuple(encoder_out.shape)}")

    if encoder_out.shape[0] == batch_size:
        x = encoder_out
    elif encoder_out.shape[1] == batch_size:
        x = encoder_out.transpose(0, 1)
    else:
        x = encoder_out

    valid = (x != 0).any(dim=-1, keepdim=True).to(dtype=x.dtype)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


class ECGFMFeatureExtractor(nn.Module):
    """ECG-FM wrapper for current single-lead I-CARE segments.

    ECG-FM is a 12-lead model.  This adapter keeps the current sample/cache
    flow unchanged, resamples inside the wrapper if needed, and places the
    single input channel into the configured 12-lead slot while zero-filling
    all other leads.
    """

    def __init__(self):
        super().__init__()
        self.backbone = _load_ecgfm_model()
        self.feature_dim = int(ECGFM_FEATURE_DIM)
        self.target_len = int(ECGFM_TARGET_LEN)
        self.n_leads = int(ECGFM_N_LEADS)
        self.single_lead_index = int(ECGFM_SINGLE_LEAD_INDEX)

    def _to_12lead(self, x):
        if x.ndim != 3:
            raise RuntimeError(f"Expected ECG-FM input [B, 1, T], got {tuple(x.shape)}")
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
        # ECG-FM/fairseq masking assigns a float32 mask embedding into the
        # hidden tensor. Running this backbone under AMP can make that hidden
        # tensor float16 and raises a dtype mismatch, so keep ECG-FM in fp32.
        with autocast(enabled=False):
            out = self.backbone(source=source.float())
        if isinstance(out, dict):
            if "encoder_out" in out:
                features = _pool_encoder_out(out["encoder_out"], source.shape[0])
            elif "features" in out:
                features = _pool_encoder_out(out["features"], source.shape[0])
            elif "out" in out:
                features = _pool_encoder_out(out["out"], source.shape[0])
            else:
                raise RuntimeError(f"ECG-FM output dict has no usable feature key: {list(out.keys())}")
        else:
            features = _pool_encoder_out(out, source.shape[0])
        return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def set_ecgfm_gradual_unfreezing(extractor, classifier_head, epoch):
    for p in extractor.parameters():
        p.requires_grad = False
    for p in classifier_head.parameters():
        p.requires_grad = True

    trainable_parts = ["classifier_head"]
    blocks = _find_block_sequence(extractor.backbone)
    if epoch >= int(ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH):
        for p in extractor.backbone.parameters():
            p.requires_grad = True
        trainable_parts.append("full_backbone")
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


def make_ecgfm_optimizer(extractor, classifier_head, weight_decay):
    blocks = _find_block_sequence(extractor.backbone)
    last_two_params = _params_from_modules(blocks[-2:]) if blocks else []
    last_two_param_ids = {id(p) for p in last_two_params}
    last_four_extra_params = _params_from_modules(blocks[-4:-2]) if len(blocks) >= 4 else []
    last_four_extra_param_ids = {id(p) for p in last_four_extra_params}
    rest_params = [
        p for p in extractor.backbone.parameters()
        if id(p) not in last_two_param_ids and id(p) not in last_four_extra_param_ids
    ]

    param_groups = [
        {"params": classifier_head.parameters(), "lr": float(ECGFOUNDER_HEAD_LR), "name": "icare_dense_head"},
    ]
    if last_two_params:
        param_groups.append({
            "params": last_two_params,
            "lr": float(ECGFOUNDER_LAST_TWO_STAGES_LR),
            "name": "ecgfm_last_two_blocks",
        })
    if last_four_extra_params:
        param_groups.append({
            "params": last_four_extra_params,
            "lr": float(ECGFOUNDER_LAST_FOUR_STAGES_LR),
            "name": "ecgfm_last_four_extra_blocks",
        })
    if rest_params:
        param_groups.append({
            "params": rest_params,
            "lr": float(ECGFOUNDER_FULL_BACKBONE_LR),
            "name": "ecgfm_full_backbone_rest",
        })
    return optim.AdamW(param_groups, weight_decay=weight_decay)
