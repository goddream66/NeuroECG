# SE-ResNet stage-1 ECG feature extractor, adapted from train_change_process.py.
from pipeline.train_config import *


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(1, int(channel) // int(reduction))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x).view(x.size(0), -1))
        return x * scale.view(x.size(0), x.size(1), 1)


class SEResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.se = SEBlock(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.GroupNorm(8, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return torch.relu(self.se(self.conv(x)) + self.shortcut(x))


class SingleChannelECGEncoder(nn.Module):
    """Shared one-lead ECG encoder from the reference train_change_process.py."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, 4, 3, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = SEResBlock(64, 128, 2)
        self.layer2 = SEResBlock(128, int(SERESNET_FEATURE_DIM), 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(float(SERESNET_DROPOUT))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        return x.flatten(1)


class SEResNetFeatureExtractor(nn.Module):
    """Mask-aware shared-channel SE-ResNet feature extractor.

    New single-lead samples normally enter as [B, 1, T].  The two-channel branch
    is kept only so older caches with [B, 2, T] remain readable.
    """

    def __init__(self):
        super().__init__()
        self.single_encoder = SingleChannelECGEncoder()
        self.feature_dim = int(SERESNET_FEATURE_DIM)

    def forward(self, x, channel_mask=None):
        if x.ndim != 3:
            raise RuntimeError(f"Expected ECG tensor [B, C, T], got {tuple(x.shape)}")

        b, c, t = x.shape
        if c == 1:
            return self.single_encoder(x)

        if c > 2:
            x = x[:, :2, :]
        if channel_mask is None:
            channel_mask = torch.ones((b, 2), device=x.device, dtype=x.dtype)
        channel_mask = channel_mask.to(device=x.device, dtype=x.dtype)
        if channel_mask.ndim != 2 or channel_mask.shape[1] != 2:
            raise RuntimeError(f"Expected channel_mask [B, 2], got {tuple(channel_mask.shape)}")

        x_flat = x[:, :2, :].reshape(b * 2, 1, t)
        feat_flat = self.single_encoder(x_flat)
        feat = feat_flat.reshape(b, 2, -1)
        mask = channel_mask.unsqueeze(-1)
        feat = feat * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return feat.sum(dim=1) / denom


def set_seresnet_training_mode(extractor, classifier_head, epoch):
    """SE-ResNet is trained from scratch, so all layers stay trainable."""
    del epoch
    for p in extractor.parameters():
        p.requires_grad = True
    for p in classifier_head.parameters():
        p.requires_grad = True
    extractor.train()
    classifier_head.train()
    return "seresnet_full_train"


def get_seresnet_epoch_lr(epoch):
    """Three-stage increasing LR schedule: 1-10 low, 11-20 mid, 21+ high."""
    epoch = int(epoch)
    if epoch < int(SERESNET_LR_LOW_EPOCHS):
        return float(SERESNET_LR_LOW), "seresnet_lr_low"
    if epoch < int(SERESNET_LR_MID_EPOCHS):
        return float(SERESNET_LR_MID), "seresnet_lr_mid"
    return float(SERESNET_LR_HIGH), "seresnet_lr_high"


def set_seresnet_epoch_lr(optimizer, epoch):
    lr, phase = get_seresnet_epoch_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return f"{phase}(lr={lr:g})"


def make_seresnet_optimizer(extractor, classifier_head, weight_decay):
    initial_lr, _ = get_seresnet_epoch_lr(0)
    return optim.AdamW(
        [
            {
                "params": extractor.parameters(),
                "lr": initial_lr,
                "name": "seresnet_backbone",
            },
            {
                "params": classifier_head.parameters(),
                "lr": initial_lr,
                "name": "icare_dense_head",
            },
        ],
        weight_decay=weight_decay,
    )
