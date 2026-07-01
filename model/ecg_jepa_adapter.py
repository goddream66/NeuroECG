# ECG-JEPA stage-1 ECG feature extractor.
from pipeline.train_config import *

import importlib.util


def _load_ecg_jepa_encoder():
    root = str(ECG_JEPA_ROOT)
    models_py = os.path.join(root, "models.py")
    if not os.path.exists(models_py):
        raise FileNotFoundError(f"ECG-JEPA models.py not found: {models_py}")
    if not os.path.exists(ECG_JEPA_CKPT):
        raise FileNotFoundError(f"ECG-JEPA checkpoint not found: {ECG_JEPA_CKPT}")

    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("_icare_ecg_jepa_models", models_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder, embed_dim = module.load_encoder(ECG_JEPA_CKPT, leads=list(ECG_JEPA_LEADS))
    return encoder, int(embed_dim)


def _encoder_blocks(extractor):
    encoder = getattr(extractor, "encoder", None)
    blocks = getattr(getattr(encoder, "encoder_blocks", None), "blocks", None)
    return list(blocks) if blocks is not None else []


def _params_from_modules(modules):
    return [p for module in modules for p in module.parameters()]


class ECGJEPAFeatureExtractor(nn.Module):
    """ECG-JEPA wrapper for the existing single-lead I-CARE segment pipeline.

    The local ECG-JEPA encoder expects [B, selected_leads, 2500].  Current
    I-CARE samples are [B, 1, 5000], so the adapter only resamples inside the
    model wrapper and leaves cache/sample construction unchanged.
    """

    def __init__(self):
        super().__init__()
        self.encoder, loaded_dim = _load_ecg_jepa_encoder()
        self.feature_dim = int(loaded_dim or ECG_JEPA_FEATURE_DIM)
        self.target_len = int(ECG_JEPA_TARGET_LEN)

    def forward(self, x, channel_mask=None):
        del channel_mask
        if x.ndim != 3:
            raise RuntimeError(f"Expected ECG-JEPA input [B, 1, T], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            x = x[:, :1, :]
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        features = self.encoder.representation(x.float())
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def set_ecg_jepa_gradual_unfreezing(extractor, classifier_head, epoch):
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
        trainable_parts.append("encoder_blocks[-4:]")
    elif blocks and epoch >= int(ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH):
        for block in blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        trainable_parts.append("encoder_blocks[-2:]")

    extractor.train() if any(p.requires_grad for p in extractor.parameters()) else extractor.eval()
    classifier_head.train()
    return "+".join(trainable_parts)


def make_ecg_jepa_optimizer(extractor, classifier_head, weight_decay):
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
            "name": "ecg_jepa_last_two_blocks",
        })
    if last_four_extra_params:
        param_groups.append({
            "params": last_four_extra_params,
            "lr": float(ECGFOUNDER_LAST_FOUR_STAGES_LR),
            "name": "ecg_jepa_last_four_extra_blocks",
        })
    if rest_params:
        param_groups.append({
            "params": rest_params,
            "lr": float(ECGFOUNDER_FULL_BACKBONE_LR),
            "name": "ecg_jepa_full_encoder_rest",
        })
    return optim.AdamW(param_groups, weight_decay=weight_decay)
