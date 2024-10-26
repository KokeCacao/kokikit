import torch
import numpy as np
import random
import os
import glob
import shutil
import psutil

from datetime import datetime
from torch import Tensor
from pathlib import Path
from torch.autograd import Function
from typing import Any, Union, List, Tuple, Dict, Callable
from functools import partial

from .const import *


def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)


class _TruncExp(Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp: Callable[..., Any] = _TruncExp.apply


class SpecifyGradient(torch.autograd.Function):
    # artificial gradient creation for nerf (passed into latents_noised)
    # this line could be written as d(loss)/d(latents) = grad after detach gradient
    # [loss = (grad * latents_noised).sum()] works fine (credits to @Xallt)

    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed(seed: int):
    ### set random seed everywhere
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.manual_seed(seed)


def safe_normalize(x, dim=-1):
    if isinstance(x, Tensor):
        return x / torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=True), min=EPS_1E20))
    elif isinstance(x, np.ndarray):
        return x / np.sqrt(np.maximum(np.sum(x * x, axis=dim, keepdims=True), EPS_1E20))
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def safe_magnitude(x, dim=-1):
    if isinstance(x, Tensor):
        return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=True), min=EPS_1E20))
    elif isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=dim, keepdims=True), EPS_1E20))
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def is_normalized(x, dim=-1):
    x = x.to(dtype=torch.float32) # can't be float16, otherwise allclose will always fail
    return torch.allclose(
        torch.norm(x, dim=dim),
        torch.ones_like(x[..., 0], dtype=torch.float32), # so ones_like should follow the same dtype
        rtol=1e-3 if x.dtype == torch.float16 else 1e-3,
        atol=1e-5 if x.dtype == torch.float16 else 1e-5,
    )


def backward_vector_to_thetas_phis(backward_vector: Tensor) -> Tuple[Tensor, Tensor]:
    thetas = torch.acos(backward_vector[..., 1]) # [B,]
    phis = torch.atan2(backward_vector[..., 0], backward_vector[..., 2]) # [B,]
    phis[phis < 0] += 2 * torch.pi
    return thetas, phis # [B,], [B,]


def recursive_copy(src: Path, dest: Path, ext: str = 'py'):
    for file_path in glob.iglob(os.path.join(src, '**', f"*.{ext}"), recursive=True):
        new_path = os.path.join(dest, os.path.join(os.path.basename(src), os.path.relpath(file_path, src)))
        new_dir = os.path.dirname(new_path)
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(file_path, new_path)


def round_floats(o: Union[float, Dict, List, None], digits: int) -> Any:
    if isinstance(o, float):
        return round(o, digits)
    if isinstance(o, dict):
        return {k: round_floats(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x, digits) for x in o]
    if isinstance(o, None.__class__):
        return o
    if isinstance(o, (int, bool, str)):
        return o
    if isinstance(o, np.ndarray):
        return round_floats(o.tolist(), digits)
    if isinstance(o, Tensor):
        return round_floats(o.detach().cpu().numpy().tolist(), digits)
    else:
        return o.__class__.__name__


def copy_src(dir: Path):
    recursive_copy(Path("src"), dir, ext="py")


def get_now_str():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def get_mem_info():
    cpu_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    gpu_gbs = [torch.cuda.memory_allocated(i) / 1024**3 for i in range(torch.cuda.device_count())]
    gpu_gbs_max = [torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(torch.cuda.device_count())]
    gpu_gbs_total = [torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(torch.cuda.device_count())]

    ret = [
        f"CPU: {cpu_gb:.1f}G",
        f"GPU: {'|'.join(f'{gpu_gb:.1f}' for gpu_gb in gpu_gbs)}G",
        f"Peak: {'|'.join(f'{gpu_gb_max:.1f}' for gpu_gb_max in gpu_gbs_max)}G",
        f"Total: {'|'.join(f'{gpu_gb_total:.1f}' for gpu_gb_total in gpu_gbs_total)}G",
    ]
    return ' | '.join(ret)


def quaternion_to_3x3_rotation(quaternion: Tensor):
    norm = torch.sqrt(quaternion[:, 0] * quaternion[:, 0] + quaternion[:, 1] * quaternion[:, 1] + quaternion[:, 2] * quaternion[:, 2] + quaternion[:, 3] * quaternion[:, 3])

    quaternion = quaternion / norm[:, None] # [B, 4]

    R = torch.zeros((quaternion.size(0), 3, 3), device=quaternion.device, dtype=quaternion.dtype)

    quaternion = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - quaternion * z)
    R[:, 0, 2] = 2 * (x * z + quaternion * y)
    R[:, 1, 0] = 2 * (x * y + quaternion * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - quaternion * x)
    R[:, 2, 0] = 2 * (x * z - quaternion * y)
    R[:, 2, 1] = 2 * (y * z + quaternion * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R # [B, 3, 3]


def scale_quaternion_to_3x3_matrix(scale: Tensor, quaternion: Tensor):
    # scale: [B, 3]
    # quaternion: [B, 4]
    R = quaternion_to_3x3_rotation(quaternion)

    S = torch.eye(3, device=scale.device, dtype=scale.dtype).repeat(scale.shape[0], 1, 1)
    for i in range(3):
        S[:, i, i] = scale[:, i]

    combined_transform = torch.matmul(R, S)
    return combined_transform # [B, 3, 3]
