# Runtime helpers for CPU/GPU selection and tensor utilities.
import gc
import os

import numpy as np
import torch

CPU_CORE_LIMIT = 4
_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

def _cap_thread_env(max_cores=CPU_CORE_LIMIT):
    for name in _CPU_THREAD_ENV_VARS:
        raw_value = os.environ.get(name)
        if raw_value is None:
            os.environ[name] = str(max_cores)
            continue
        try:
            value = int(raw_value)
        except ValueError:
            value = max_cores
        os.environ[name] = str(max(1, min(value, max_cores)))


def _cap_cpu_affinity(max_cores=CPU_CORE_LIMIT):
    if not (hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")):
        return
    try:
        allowed_cpus = sorted(os.sched_getaffinity(0))
        if len(allowed_cpus) > max_cores:
            os.sched_setaffinity(0, set(allowed_cpus[:max_cores]))
    except OSError:
        return


def get_cpu_worker_count(max_cores=CPU_CORE_LIMIT):
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, min(len(os.sched_getaffinity(0)), max_cores))
    except OSError:
        pass
    return max(1, min(os.cpu_count() or max_cores, max_cores))


_cap_thread_env()
_cap_cpu_affinity()
CPU_WORKERS = get_cpu_worker_count()


def release_memory(label=None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    if label:
        print(f"[Memory] Released unused objects after {label}.")


def get_gpu_device(context=""):
    if not torch.cuda.is_available():
        prefix = f"{context}: " if context else ""
        raise RuntimeError(
            prefix
            + "CUDA GPU is required for this experiment. "
            "CPU fallback is disabled by request."
        )
    device_index = int(torch.cuda.current_device())
    device = torch.device("cuda", device_index)
    torch.backends.cudnn.benchmark = True
    return device


def get_current_cuda_device_index():
    get_gpu_device("CUDA device selection")
    return int(torch.cuda.current_device())


def catboost_gpu_kwargs():
    devices = os.environ.get("CATBOOST_DEVICES")
    if devices is None or str(devices).strip() == "":
        devices = str(get_current_cuda_device_index())
    return {"task_type": "GPU", "devices": str(devices)}


def describe_selected_gpu(context="GPU"):
    device = get_gpu_device(context)
    device_index = int(device.index)
    name = torch.cuda.get_device_name(device_index)
    catboost_devices = catboost_gpu_kwargs()["devices"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(
        f"[{context}] Using torch device cuda:{device_index} ({name}) | "
        f"CUDA_VISIBLE_DEVICES={visible} | CatBoost devices={catboost_devices}"
    )
    return device


def _gpu_float_tensor(array_like, device=None):
    device = get_gpu_device("GPU tensor conversion") if device is None else device
    return torch.as_tensor(array_like, dtype=torch.float32, device=device)


def _gpu_clean_feature_tensor(array_like, device=None):
    x = _gpu_float_tensor(array_like, device=device)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _gpu_to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def gpu_concat_feature_blocks(blocks):
    if len(blocks) == 1:
        return blocks[0]
    device = get_gpu_device("feature concatenation")
    tensors = [_gpu_float_tensor(block, device=device) for block in blocks]
    out = torch.cat(tensors, dim=1)
    result = _gpu_to_numpy(out)
    del tensors, out
    release_memory("GPU feature concatenation")
    return result


def gpu_concat_feature_vectors(blocks):
    device = get_gpu_device("feature vector concatenation")
    tensors = [_gpu_float_tensor(block, device=device).reshape(-1) for block in blocks]
    out = torch.cat(tensors, dim=0)
    result = _gpu_to_numpy(out)
    del tensors, out
    return result


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


