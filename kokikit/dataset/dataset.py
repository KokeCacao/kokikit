import torch
import numpy as np

from typing import Tuple, Optional
from torch import Tensor

from ..utils.utils import safe_normalize
from ..nerf.rays import RayBundle


class Dataset:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_train_ray_bundle(self, batch_size: int) -> RayBundle:
        raise NotImplementedError

    def get_eval_ray_bundle(self, h_latent: int, w_latent: int, c2w: Tensor, fov: float, near: float, far: float) -> RayBundle:
        raise NotImplementedError

    def get_test_ray_bundle(self, batch_size: int, idx: Optional[Tensor]) -> RayBundle:
        raise NotImplementedError

    @staticmethod
    def _get_round_rays_o(
        batch_size: int,
        radius_float: float,
        idx: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        radius = torch.ones(batch_size, device=device) * radius_float # [B,]
        thetas = torch.ones(batch_size, device=device) * 90 / 180 * np.pi # [B,]
        phis = torch.linspace(0, 360, batch_size, device=device) / 180 * np.pi # [B,]

        if idx is not None:
            radius = radius[idx]
            thetas = thetas[idx]
            phis = phis[idx]

        ray_o = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]
        return thetas, phis, ray_o # [B,], [B,], [B, 3]

    @staticmethod
    def _get_lookat(batch_size: int, rays_o: Tensor, targets: Tensor, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
        # Get camera lookat vectors.

        forward_vector = safe_normalize(targets - rays_o) # [B, 3]
        up_vector = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(batch_size, 1) # [B, 3]
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1)) # [B, 3]
        up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1)) # [B, 3]

        return up_vector, right_vector, forward_vector # [B, 3], [B, 3], [B, 3]

    @staticmethod
    def _get_sphere_rays_o(
        batch_size: int,
        radius_float: float,
        loops: int,
        idx: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        radius = torch.ones(batch_size, device=device) * radius_float # [B,]
        thetas = torch.linspace(0, 180, batch_size, device=device) / 180 * np.pi # [B,]
        phis = (torch.linspace(0, 360 * loops, batch_size, device=device) % 360) / 180 * np.pi # [B,]

        if idx is not None:
            radius = radius[idx]
            thetas = thetas[idx]
            phis = phis[idx]

        ray_o = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]
        return thetas, phis, ray_o # [B,], [B,], [B, 3]
