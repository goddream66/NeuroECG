# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *

class ECGFounderFeatureExtractor(nn.Module):
    """Single-lead ECGFounder backbone wrapper.

    The uploaded ECGFounder Net1D can return both logits and deep_features when
    return_features=True.  For this I-CARE pipeline, the original 150-class
    dense head is kept only for checkpoint compatibility and is not used.  The
    forward method returns the 1024-dim deep_features before the dense layer, so
    the existing classifier_head can learn the Good/Poor segment objective.
    """

    def __init__(self, ecgfounder_root=ECGFOUNDER_ROOT, checkpoint_path=ECGFOUNDER_1LEAD_CKPT):
        super().__init__()
        self.ecgfounder_root = str(ecgfounder_root)
        self.checkpoint_path = str(checkpoint_path)
        try:
            from model.ecgfounder_net1d import Net1D
        except Exception as exc:
            raise ImportError(
                "Failed to import extracted ECGFounder Net1D from model.ecgfounder_net1d"
            ) from exc

        self.backbone = Net1D(
            in_channels=1,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=150,
            return_features=True,
        )
        self.feature_dim = int(ECGFOUNDER_FEATURE_DIM)
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"ECGFounder checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            # The original ECGFounder dense head is 150-class pretraining output.
            # We drop it and train a new I-CARE Good/Poor head outside the backbone.
            if key.startswith("dense."):
                continue
            cleaned[key] = value
        load_msg = self.backbone.load_state_dict(cleaned, strict=False)
        print(
            f"[ECGFounder] Loaded 1-lead checkpoint: {self.checkpoint_path} | "
            f"missing={len(load_msg.missing_keys)} | unexpected={len(load_msg.unexpected_keys)}"
        )
        for p in self.backbone.dense.parameters():
            p.requires_grad = False

    def forward(self, x, channel_mask=None):
        del channel_mask
        if x.ndim != 3:
            raise RuntimeError(f"Expected ECGFounder input [B, 1, T], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            # Defensive fallback: this script should already split channels into
            # independent single-lead samples in build_indices()/ICAREDataset.
            x = x[:, :1, :]
        _, deep_features = self.backbone(x)
        return deep_features


def set_ecgfounder_gradual_unfreezing(extractor, classifier_head, epoch):
    """Freeze/unfreeze ECGFounder by epoch.

    Epochs 1-7: train only the Good/Poor classifier head.
    Epochs 8-15: unfreeze the last two ECGFounder stages plus the head.
    Epochs 16-20: unfreeze the last four ECGFounder stages plus the head.
    Epochs 21+: unfreeze the full ECGFounder backbone plus the head.
    """
    for p in extractor.parameters():
        p.requires_grad = False
    for p in classifier_head.parameters():
        p.requires_grad = True

    trainable_parts = ["classifier_head"]
    if hasattr(extractor, "backbone") and hasattr(extractor.backbone, "stage_list"):
        stage_list = list(extractor.backbone.stage_list)
        if epoch >= int(ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH):
            for p in extractor.backbone.parameters():
                p.requires_grad = True
            for p in extractor.backbone.dense.parameters():
                p.requires_grad = False
            trainable_parts.append("full_backbone")
        elif epoch >= int(ECGFOUNDER_UNFREEZE_LAST_FOUR_STAGES_EPOCH):
            for stage in stage_list[-4:]:
                for p in stage.parameters():
                    p.requires_grad = True
            trainable_parts.append("stage_list[-4:]")
        elif epoch >= int(ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH):
            for stage in stage_list[-2:]:
                for p in stage.parameters():
                    p.requires_grad = True
            trainable_parts.append("stage_list[-2:]")

    if any(p.requires_grad for p in extractor.parameters()):
        extractor.train()
    else:
        extractor.eval()
    classifier_head.train()
    return "+".join(trainable_parts)


def make_ecgfounder_optimizer(extractor, classifier_head, weight_decay):
    """AdamW with discriminative learning rates for gradual ECGFounder fine-tuning."""
    param_groups = [
        {"params": classifier_head.parameters(), "lr": float(ECGFOUNDER_HEAD_LR), "name": "icare_dense_head"},
    ]
    if hasattr(extractor, "backbone") and hasattr(extractor.backbone, "stage_list"):
        stage_list = list(extractor.backbone.stage_list)
        dense_param_ids = {id(p) for p in extractor.backbone.dense.parameters()}
        last_two_params = [p for stage in stage_list[-2:] for p in stage.parameters()]
        last_two_param_ids = {id(p) for p in last_two_params}
        last_four_extra_params = [p for stage in stage_list[-4:-2] for p in stage.parameters()]
        last_four_extra_param_ids = {id(p) for p in last_four_extra_params}
        backbone_rest_params = [
            p
            for p in extractor.backbone.parameters()
            if id(p) not in last_two_param_ids
            and id(p) not in last_four_extra_param_ids
            and id(p) not in dense_param_ids
        ]
        param_groups.append({
            "params": last_two_params,
            "lr": float(ECGFOUNDER_LAST_TWO_STAGES_LR),
            "name": "ecgfounder_last_two_stages",
        })
        param_groups.append({
            "params": last_four_extra_params,
            "lr": float(ECGFOUNDER_LAST_FOUR_STAGES_LR),
            "name": "ecgfounder_last_four_extra_stages",
        })
        param_groups.append({
            "params": backbone_rest_params,
            "lr": float(ECGFOUNDER_FULL_BACKBONE_LR),
            "name": "ecgfounder_full_backbone_rest",
        })
    return optim.AdamW(param_groups, weight_decay=weight_decay)
