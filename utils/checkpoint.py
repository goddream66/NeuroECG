# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *

class EarlyStopping:
    def __init__(self, patience=1, min_delta=0.0, mode="min", path="rode_final.pth"):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def _is_improvement(self, score):
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        return score > (self.best_score + self.min_delta)

    def __call__(self, score, model):
        if self._is_improvement(score):
            self.counter = 0
            self.best_score = score
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), self.path)
            else:
                torch.save(model, self.path)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "path": self.path,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict.get("patience", self.patience)
        self.min_delta = state_dict.get("min_delta", self.min_delta)
        self.mode = state_dict.get("mode", self.mode)
        self.counter = state_dict.get("counter", 0)
        self.best_score = state_dict.get("best_score")
        self.early_stop = state_dict.get("early_stop", False)
        self.path = state_dict.get("path", self.path)


def load_resume_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, map_location=device)


def load_ptbxl_student_weights(extractor, projector, ckpt_path, device, load_projector=True):
    """Load PTB-XL single-lead student checkpoint into I-CARE extractor/projector.

    Expected checkpoint keys from pretrain_ptbxl_singlelead_student_kerasfix_v3.py:
        ckpt["extractor"]
        ckpt["projector"]
        ckpt["head"]

    We intentionally do NOT load ckpt["head"], because it predicts PTB-XL
    superclass outputs rather than I-CARE Good/Poor outcome.
    """
    if not ckpt_path:
        print("[PTB-XL Pretrain] No student checkpoint specified. Training from scratch.")
        return False

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"PTB-XL student checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported PTB-XL student checkpoint format: {type(ckpt)}")

    extractor_state = ckpt.get("extractor")
    if extractor_state is None:
        raise RuntimeError("PTB-XL student checkpoint has no key 'extractor'.")

    # Current extractor uses a shared one-channel encoder under prefix
    # single_encoder.*. Older checkpoints may have used stem/layer1/layer2
    # directly, sometimes with a 2-channel first convolution. Adapt keys and
    # average the first conv over input channels when needed.
    adapted_state = {}
    for key, value in extractor_state.items():
        new_key = str(key)
        if not new_key.startswith("single_encoder."):
            if new_key.startswith(("stem.", "layer1.", "layer2.", "avgpool.", "dropout.")):
                new_key = "single_encoder." + new_key
        if new_key.endswith("stem.0.weight") or new_key.endswith("single_encoder.stem.0.weight"):
            if hasattr(value, "ndim") and value.ndim == 3 and value.shape[1] != 1:
                value = value.mean(dim=1, keepdim=True)
        adapted_state[new_key] = value

    missing, unexpected = extractor.load_state_dict(adapted_state, strict=False)
    print(f"[PTB-XL Pretrain] Loaded/adapted extractor from: {ckpt_path}")
    print(f"[PTB-XL Pretrain] extractor load strict=False | missing={len(missing)} | unexpected={len(unexpected)}")

    if load_projector:
        projector_state = ckpt.get("projector")
        if projector_state is not None:
            try:
                projector.load_state_dict(projector_state, strict=False)
                print("[PTB-XL Pretrain] Loaded projector with strict=False.")
            except RuntimeError as exc:
                print(f"[PTB-XL Pretrain] Projector load skipped due to mismatch: {exc}")
        else:
            print("[PTB-XL Pretrain] No projector key found; projector remains randomly initialized.")

    print("[PTB-XL Pretrain] Student head is intentionally NOT loaded.")
    return True


def save_training_resume_checkpoint(
    checkpoint_path: str,
    epoch: int,
    extractor: nn.Module,
    projector: nn.Module,
    classifier_head: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    early_stop: EarlyStopping,
    train_pids: Sequence[str],
    val_pids: Sequence[str],
    test_pids: Sequence[str],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "extractor_state_dict": extractor.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "classifier_head_state_dict": classifier_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "early_stopping_state": early_stop.state_dict(),
            "train_pids": sorted(map(str, train_pids)),
            "val_pids": sorted(map(str, val_pids)),
            "test_pids": sorted(map(str, test_pids)),
        },
        checkpoint_path,
    )
