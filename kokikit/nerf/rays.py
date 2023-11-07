import warnings
import torch
import json

from typing import Optional, Tuple
from torch import Tensor

from ..utils.utils import round_floats, backward_vector_to_thetas_phis
from .colliders import Collider


class RayBundle:

    def __init__(
        self,
        near_plane: float,
        far_plane: float,
        fov_x: float,
        fov_y: float,
        origins: Tensor,
        collider: Optional[Collider],
        directions: Optional[Tensor],
        nears: Optional[Tensor],
        fars: Optional[Tensor],
        forward_vector: Tensor,
        mvp: Optional[Tensor],
        c2w: Optional[Tensor],
    ) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.fov_x = fov_x
        self.fov_y = fov_y
        
        self.origins = origins # [B, H, W, 2] or [B, H, W, 3] (or [B, H*W, 3] if sampled)
        if directions is not None:
            assert torch.allclose(torch.norm(directions, dim=-1), torch.ones_like(directions[..., 0])), f"torch.norm(directions, dim=-1)={torch.norm(directions, dim=-1)}"
        self.directions = directions # None or [B, H, W, 3] (or [B, H*W, 3] if sampled)
        self.nearsnear_plane = nears # None or [B, H, W, 1] (or [B, H*W, 1] if sampled)
        self.fars = fars # None or [B, H, W, 1] (or [B, H*W, 1] if sampled)
        assert self.nears is None or self.fars is None or torch.all(self.nears <= self.fars), f"self.fars - self.nears={(self.fars - self.nears).min()}"
        self.forward_vector = forward_vector # None or [B, 3], assume always look at origin
        self.collider = collider # None or Collider
        self.mvp = mvp # None or [B, 4, 4]
        self.c2w = c2w # None or [B, 4, 4]

    def get_thetas_phis(self) -> Tuple[Tensor, Tensor]:
        backward_vector = -self.forward_vector # [B, 3]
        return backward_vector_to_thetas_phis(backward_vector)

    def plot(self, save_path: str, subsampling_factor: int = 10):
        try:
            import matplotlib.pyplot as plt # type: ignore
        except ImportError:
            warnings.warn(f"Cannot import matplotlib.pyplot, skip plotting.", ImportWarning)
            return

        assert self.directions is not None, "directions is None"

        # Convert to numpy for plotting
        rays_o_np = self.origins.reshape(-1, 3).cpu().numpy()
        rays_d_np = self.directions.reshape(-1, 3).cpu().numpy()

        # Subsample the rays for plotting
        rays_o_np_subsampled = rays_o_np[::subsampling_factor]
        rays_d_np_subsampled = rays_d_np[::subsampling_factor]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(rays_o_np_subsampled[:, 0], rays_o_np_subsampled[:, 1], rays_o_np_subsampled[:, 2], rays_d_np_subsampled[:, 0], rays_d_np_subsampled[:, 1], rays_d_np_subsampled[:, 2], length=1.0, color='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Rays from cameras to origin')
        plt.savefig(save_path)

    def to_device(self, device, dtype: torch.dtype):
        # send rays_o, rays_d to device
        self.origins = self.origins.to(device=device, dtype=dtype)
        if self.directions is not None:
            self.directions = self.directions.to(device=device, dtype=dtype)

        # collide rays with the scene
        if self.nears is None and self.fars is None and self.collider is not None:
            assert self.origins.shape[-1] == 3, f"origins.shape[-1]={self.origins.shape[-1]}"
            assert self.directions is not None, "directions is None"
            nears, fars = self.collider.collide(rays_o=self.origins, rays_d=self.directions) # [B, H, W, 1], [B, H, W, 1]
            self.nears = nears
            self.fars = fars
        else:
            warnings.warn(f"Not colliding rays with the scene because [collider] is None.", RuntimeWarning)

        # send other tensors to device
        if self.nears is not None:
            self.nears = self.nears.to(device=device, dtype=dtype)
        if self.fars is not None:
            self.fars = self.fars.to(device=device, dtype=dtype)
        if self.c2w is not None:
            self.c2w = self.c2w.to(device=device, dtype=dtype)

        # tensors don't go to device:
        # self.forward_vector = self.forward_vector
        # self.mvp = self.mvp
        return self

    def get_sampled_rays(self, num_samples: int) -> "RayBundle":
        # sample num_samples rays from the ray bundle, return a new RayBundle object
        # however, sampled rays will have different shapes
        if self.origins.shape[1] * self.origins.shape[2] < num_samples:
            return RayBundle(
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                fov_x=self.fov_x,
                fov_y=self.fov_y,
                origins=self.origins,
                directions=self.directions,
                collider=self.collider,
                nears=self.nears,
                fars=self.fars,
                forward_vector=self.forward_vector,
                mvp=self.mvp,
                c2w=self.c2w,
            )

        idx = torch.randperm(self.origins.shape[1] * self.origins.shape[2], device=self.origins.device)[:num_samples] # [num_samples,]
        h = idx // self.origins.shape[2]
        w = idx % self.origins.shape[2]
        origins = self.origins[:, h, w, :] # [B, num_samples, 2] or [B, num_samples, 3]
        directions = None
        nears = None
        fars = None
        if self.directions is not None:
            directions = self.directions[:, h, w, :] # [B, num_samples, 3]
        if self.nears is not None:
            nears = self.nears[:, h, w, :] # [B, num_samples, 1]
        if self.fars is not None:
            fars = self.fars[:, h, w, :] # [B, num_samples, 1]
        return RayBundle(
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            fov_x=self.fov_x,
            fov_y=self.fov_y,
            origins=origins,
            directions=directions,
            collider=self.collider,
            nears=nears,
            fars=fars,
            forward_vector=self.forward_vector,
            mvp=self.mvp,
            c2w=self.c2w,
        ) # [B, num_samples, 2] or [B, num_samples, 3], [B, num_samples, 3], [B, num_samples], [B, num_samples], [B,]

    def to_dict(self) -> dict:
        d = self.get_sampled_rays(16).__dict__ # always sample 16 rays per batch
        # no need to copy because get_sampled_rays() returns a new object
        for k, v in d.items():
            if isinstance(v, Tensor):
                d[k] = v.detach().cpu().numpy().tolist()
        return round_floats(d, 8) # to avoid json serialization load

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)
