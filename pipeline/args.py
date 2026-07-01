# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *

def parse_stage1_args():
    parser = argparse.ArgumentParser(
        description="I-CARE Stage 1 plain segment-level outcome training without PTBXL auxiliary losses or MIL pooling."
    )
    parser.add_argument(
        "--ptbxl_student_ckpt",
        default="",
        help="Path to PTB-XL single-lead student best_student.pth. Empty means no external pretraining.",
    )
    parser.add_argument(
        "--pretrain_name",
        default="none",
        help="Name used in checkpoint/output tags, e.g. leadII, random12, weighted, none.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--early_stop_min_epochs", type=int, default=0)
    parser.add_argument(
        "--load_student_projector",
        type=int,
        default=1,
        help="1: load PTB-XL student projector; 0: load extractor only and initialize projector randomly.",
    )
    parser.add_argument("--segment_train_batch_size", type=int, default=256, help="Batch size for segment-level Stage 1 training/evaluation. No MIL bag is used.")
    parser.add_argument("--semantic_batch_size", type=int, default=512, help="Batch size for Stage 2 PTBXL71 semantic feature extraction.")
    return parser.parse_args()
