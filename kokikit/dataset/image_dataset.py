import torch

from typing import Optional
from torch import Tensor

from ..nerf.rays import RayBundle
from .dataset import Dataset


class ImageDataset(Dataset):

    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device

    def get_train_ray_bundle(
        self,
        h_latent_dataset: int,
        w_latent_dataset: int,
        batch_size: int,
    ) -> RayBundle:
        h_latent = h_latent_dataset
        w_latent = w_latent_dataset
        device = self.device

        # Generate an input grid coordinates in a range of [-1, 1]
        h = torch.linspace(start=-1, end=1, steps=h_latent, device=device)
        w = torch.linspace(start=-1, end=1, steps=w_latent, device=device)
        samples = torch.meshgrid(w, h, indexing='xy')
        samples = torch.stack(samples, dim=-1).unsqueeze(0) # [1, H, W, 2]
        return RayBundle(
            origins=samples.expand(batch_size, -1, -1, -1),
            collider=None,
            directions=None,
            nears=None,
            fars=None,
            forward_vector=torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).expand(batch_size, -1),
            mvp=None,
        ) # No positional encoding since we use SIREN

    def get_eval_ray_bundle(self, h_latent: int, w_latent: int, c2w: Tensor, fov: float, focal: float, near: float, far: float) -> RayBundle:
        device = self.device

        # Generate an input grid coordinates in a range of [-1, 1]
        h = torch.linspace(start=-1, end=1, steps=h_latent, device=device)
        w = torch.linspace(start=-1, end=1, steps=w_latent, device=device)
        samples = torch.meshgrid(w, h, indexing='xy')
        samples = torch.stack(samples, dim=-1).unsqueeze(0) # [1, H, W, 2]

        thetas, phis, rays_o = self._get_round_rays_o(
            batch_size=1, # Note that here is [batch_size] instead of [selected_batch_size]
            radius_float=1,
            idx=None,
            device=device,
        )
        up_vector, right_vector, forward_vector = self._get_lookat(
            batch_size=1,
            rays_o=rays_o,
            targets=torch.zeros_like(rays_o),
            device=device,
        ) # [B, 3], [B, 3], [B, 3]

        return RayBundle(
            origins=samples.expand(1, -1, -1, -1),
            collider=None,
            directions=None,
            nears=None,
            fars=None,
            forward_vector=forward_vector,
            mvp=None,
        ) # No positional encoding since we use SIREN

    def get_test_ray_bundle(
        self,
        h_latent_dataset: int,
        w_latent_dataset: int,
        batch_size: int,
        idx: Optional[Tensor],
    ) -> RayBundle:
        selected_batch_size = batch_size if idx is None else idx.shape[0]
        h_latent = h_latent_dataset
        w_latent = w_latent_dataset
        device = self.device

        # Generate an input grid coordinates in a range of [-1, 1]
        h = torch.linspace(start=-1, end=1, steps=h_latent, device=device)
        w = torch.linspace(start=-1, end=1, steps=w_latent, device=device)
        samples = torch.meshgrid(w, h, indexing='xy')
        samples = torch.stack(samples, dim=-1).unsqueeze(0) # [1, H, W, 2]

        thetas, phis, rays_o = self._get_round_rays_o(
            batch_size=batch_size, # Note that here is [batch_size] instead of [selected_batch_size]
            radius_float=1,
            idx=idx,
            device=device,
        )
        up_vector, right_vector, forward_vector = self._get_lookat(
            batch_size=selected_batch_size,
            rays_o=rays_o,
            targets=torch.zeros_like(rays_o),
            device=device,
        ) # [B, 3], [B, 3], [B, 3]

        return RayBundle(
            origins=samples.expand(selected_batch_size, -1, -1, -1),
            collider=None,
            directions=None,
            nears=None,
            fars=None,
            forward_vector=forward_vector,
            mvp=None,
        ) # No positional encoding since we use SIREN
