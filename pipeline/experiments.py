# Auto-split from pipeline/train_core.py.
from pipeline.train_config import *
from evaluation.metrics import evaluate_model
from utils.runtime import catboost_gpu_kwargs, gpu_concat_feature_blocks, release_memory

def make_design_matrix(blocks: Dict[str, object], variant: str) -> np.ndarray:
    static = blocks["static"]
    hrv = blocks["hrv"]
    statement = blocks["statement"]
    deep = blocks.get("deep") if isinstance(blocks, dict) else None
    deep_pca64 = blocks.get(f"deep_pca{DEEP_PCA_DIM}") if isinstance(blocks, dict) else None
    if variant == "static_only":
        return static
    if variant == "hrv_only":
        return hrv
    if variant in {"semantic150_only", "statement71_only"}:
        return statement
    if variant == "hrv_static":
        return gpu_concat_feature_blocks([static, hrv])
    if variant in {"semantic150_static", "statement71_static"}:
        return gpu_concat_feature_blocks([static, statement])
    if variant in {"semantic150_hrv_static", "statement71_hrv_static"}:
        return gpu_concat_feature_blocks([static, hrv, statement])
    if variant == "deep_only":
        return deep
    if variant == "deep_pca64_only":
        return deep_pca64
    if variant == "deep_static":
        return gpu_concat_feature_blocks([static, deep])
    if variant == "deep_pca64_static":
        return gpu_concat_feature_blocks([static, deep_pca64])
    if variant == "deep_hrv_static":
        return gpu_concat_feature_blocks([static, hrv, deep])
    if variant == "deep_pca64_hrv_static":
        return gpu_concat_feature_blocks([static, hrv, deep_pca64])
    if variant in {"deep_semantic150_only", "deep_statement71_only"}:
        return gpu_concat_feature_blocks([deep, statement])
    if variant == "deep_pca64_statement71_only":
        return gpu_concat_feature_blocks([deep_pca64, statement])
    if variant in {"deep_semantic150_static", "deep_statement71_static"}:
        return gpu_concat_feature_blocks([static, deep, statement])
    if variant == "deep_pca64_statement71_static":
        return gpu_concat_feature_blocks([static, deep_pca64, statement])
    if variant in {"deep_semantic150_hrv_static", "deep_statement71_hrv_static"}:
        return gpu_concat_feature_blocks([static, hrv, deep, statement])
    if variant == "deep_pca64_statement71_hrv_static":
        return gpu_concat_feature_blocks([static, hrv, deep_pca64, statement])
    if variant == "deep_aux_projected":
        deep_aux_projected = blocks.get("deep_aux_projected") if isinstance(blocks, dict) else None
        if deep_aux_projected is None:
            raise ValueError("Missing deep_aux_projected for variant: deep_aux_projected")
        return deep_aux_projected

    raise ValueError(f"Unknown variant: {variant}")


def train_and_eval(variant: str, representation: str, seed: int, train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray, blocks: Dict[str, Dict[str, object]], pid_meta: Dict[str, dict]) -> Dict[str, object]:
    y_tr, y_val, y_te = blocks["train"]["y"], blocks["val"]["y"], blocks["test"]["y"]
    cpc_tr, cpc_val, cpc_te = blocks["train"]["cpc"], blocks["val"]["cpc"], blocks["test"]["cpc"]
    cls = CatBoostClassifier(iterations=CATBOOST_ITERATIONS, depth=CATBOOST_DEPTH, learning_rate=CATBOOST_LR, class_weights=CLASS_WEIGHTS, random_seed=seed, verbose=0, allow_writing_files=False, thread_count=CPU_WORKERS, **catboost_gpu_kwargs())
    cls.fit(train_X, y_tr, eval_set=(val_X, y_val), use_best_model=True)
    reg = CatBoostRegressor(iterations=CATBOOST_ITERATIONS, depth=CATBOOST_DEPTH, learning_rate=CATBOOST_LR, random_seed=seed, verbose=0, allow_writing_files=False, thread_count=CPU_WORKERS, **catboost_gpu_kwargs())
    reg.fit(train_X, cpc_tr, eval_set=(val_X, cpc_val), use_best_model=True)
    row = {"variant": variant, "representation": representation, "seed": int(seed), "feature_dim": int(train_X.shape[1]), "train_n": int(len(y_tr)), "val_n": int(len(y_val)), "test_n": int(len(y_te))}
    for split_name, payload in {"train": (train_X, y_tr, cpc_tr, blocks["train"]["patient_ids"]), "val": (val_X, y_val, cpc_val, blocks["val"]["patient_ids"]), "test": (test_X, y_te, cpc_te, blocks["test"]["patient_ids"])}.items():
        metrics = evaluate_model(cls, reg, *payload, pid_meta=pid_meta)
        for key, value in metrics.items():
            row[f"{split_name}_{key}"] = value
    print(f"  [Result] {variant:22s} rep={representation:24s} seed={seed} | dim={train_X.shape[1]:4d} | test AUROC={row['test_auroc']:.4f}, AUPRC={row['test_auprc']:.4f}, Acc={row['test_accuracy']:.4f}, Spec={row['test_specificity']:.4f}, CPC_MAE={row['test_cpc_mae']:.4f}")
    del cls, reg
    release_memory()
    return row


def train_eval_design_specs(
    design_specs: Sequence[Tuple[str, str]],
    blocks: Dict[str, Dict[str, object]],
    pid_meta: Dict[str, dict],
    log_prefix: str,
) -> List[Dict[str, object]]:
    rows = []
    for variant, rep in design_specs:
        train_X = val_X = test_X = None
        try:
            train_X = make_design_matrix(blocks["train"], variant)
            val_X = make_design_matrix(blocks["val"], variant)
            test_X = make_design_matrix(blocks["test"], variant)
            for run_idx in range(NUM_RUNS):
                seed = BASE_SEED + run_idx
                print(f"\n[{log_prefix}] variant={variant} | representation={rep} | seed={seed}")
                rows.append(train_and_eval(variant, rep, seed, train_X, val_X, test_X, blocks, pid_meta))
        finally:
            del train_X, val_X, test_X
            release_memory(f"{variant} design matrices")
    return rows
