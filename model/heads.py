# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *

class BinaryClassificationHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, inputs):
        return self.fc(inputs).squeeze(1)


class FeatureProjector(nn.Module):
    """Project ECGFounder backbone features to a compact embedding."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(in_features), int(hidden_features)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_features), int(out_features)),
        )
        self.norm = nn.LayerNorm(int(out_features))

    def forward(self, inputs):
        return self.norm(self.mlp(inputs))


class IdentityProjector(nn.Module):
    """Pass-through projector used when ECGFounder already provides deep features."""

    def forward(self, inputs):
        return inputs
