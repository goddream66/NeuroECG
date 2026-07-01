# ConvNeXt-style 1D ECG feature extractor for stage-1 training.
from pipeline.train_config import *


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ChannelLayerNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels), eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()
        dim = int(dim)
        hidden_dim = int(expansion) * dim
        padding = int(kernel_size) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=int(kernel_size), padding=padding, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = (
            nn.Parameter(float(layer_scale_init_value) * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + self.drop_path(x)


class SingleChannelConvNeXt1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = tuple(int(x) for x in CONVNEXT1D_DIMS)
        depths = tuple(int(x) for x in CONVNEXT1D_DEPTHS)
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("CONVNEXT1D_DIMS and CONVNEXT1D_DEPTHS must both contain four stages.")

        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=7, stride=4, padding=3),
            ChannelLayerNorm1d(dims[0]),
        )
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    ChannelLayerNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        total_blocks = int(sum(depths))
        drop_rates = torch.linspace(0.0, float(CONVNEXT1D_DROP_PATH_RATE), total_blocks).tolist()
        block_idx = 0
        self.stages = nn.ModuleList()
        for stage_idx, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock1D(
                        dims[stage_idx],
                        kernel_size=CONVNEXT1D_KERNEL_SIZE,
                        layer_scale_init_value=CONVNEXT1D_LAYER_SCALE_INIT,
                        drop_path=drop_rates[block_idx],
                    )
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(float(CONVNEXT1D_DROPOUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx < len(self.downsample_layers):
                x = self.downsample_layers[stage_idx](x)
        x = x.mean(dim=-1)
        x = self.final_norm(x)
        x = self.dropout(x)
        return x


class ConvNeXt1DFeatureExtractor(nn.Module):
    """Mask-aware single-lead ConvNeXt1D feature extractor.

    New samples are [B, 1, T]. The two-channel branch keeps older caches usable
    by encoding channels independently and averaging valid channel features.
    """

    def __init__(self):
        super().__init__()
        self.single_encoder = SingleChannelConvNeXt1DEncoder()
        self.feature_dim = int(CONVNEXT1D_FEATURE_DIM)

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


def set_convnext1d_training_mode(extractor, classifier_head, epoch):
    del epoch
    for p in extractor.parameters():
        p.requires_grad = True
    for p in classifier_head.parameters():
        p.requires_grad = True
    extractor.train()
    classifier_head.train()
    return "convnext1d_full_train"


def get_convnext1d_epoch_lr(epoch):
    epoch = int(epoch)
    if epoch < int(CONVNEXT1D_LR_LOW_EPOCHS):
        return float(CONVNEXT1D_LR_LOW), "convnext1d_lr_low"
    if epoch < int(CONVNEXT1D_LR_MID_EPOCHS):
        return float(CONVNEXT1D_LR_MID), "convnext1d_lr_mid"
    return float(CONVNEXT1D_LR_HIGH), "convnext1d_lr_high"


def set_convnext1d_epoch_lr(optimizer, epoch):
    lr, phase = get_convnext1d_epoch_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return f"{phase}(lr={lr:g})"


def make_convnext1d_optimizer(extractor, classifier_head, weight_decay):
    initial_lr, _ = get_convnext1d_epoch_lr(0)
    return optim.AdamW(
        [
            {
                "params": extractor.parameters(),
                "lr": initial_lr,
                "name": "convnext1d_backbone",
            },
            {
                "params": classifier_head.parameters(),
                "lr": initial_lr,
                "name": "icare_dense_head",
            },
        ],
        weight_decay=weight_decay,
    )
