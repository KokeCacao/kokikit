import torch

from torch import Tensor
from typing import Tuple
from ..utils.const import *


class Collider(torch.nn.Module):

    def __init__(self, near_plane, far_plane) -> None:
        super().__init__()
        self.near_plane = near_plane
        self.far_plane = far_plane

    def collide(self, rays_o: Tensor, rays_d: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class AABBBoxCollider(Collider):

    def __init__(self, mins: Tensor, maxs: Tensor, near_plane: float, far_plane: float) -> None:
        super().__init__(near_plane=near_plane, far_plane=far_plane)
        self.mins = mins
        self.maxs = maxs

    def collide(self, rays_o: Tensor, rays_d: Tensor) -> Tuple[Tensor, Tensor]:
        # Adapted from relu_field/sample.py
        # Refer to: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
        # --------------------------------------------------------------------------------------
        # compute the near and far bounds for each ray based on its intersection with the aabb
        # --------------------------------------------------------------------------------------
        # Turns out there is a much more elegant form of this function in Ray-Tracing-Gems II :)
        # Here -> https://link.springer.com/content/pdf/10.1007/978-1-4842-7185-8.pdf
        # Page 88 / 884. The physical hardcover copy is not expensive btw :)
        aabb = torch.stack([self.mins, self.maxs], dim=0)
        dtype, device = rays_o.dtype, rays_o.device
        rays_o, rays_d = rays_o, rays_d
        num_rays = rays_o.shape[0]
        orig_ray_bounds = (torch.tensor([self.near_plane, self.far_plane], dtype=dtype, device=device).reshape(1, -1).repeat(num_rays, 1))
        intersecting = torch.tensor([[True]], dtype=torch.bool, device=device).repeat([num_rays, 1])
        non_intersecting = torch.tensor([[False]], dtype=torch.bool, device=device).repeat([num_rays, 1])

        # compute intersections with the X-planes:
        x_min = (aabb[0][0] - rays_o[:, 0]) / (rays_d[:, 0] + EPS_1E10)
        x_max = (aabb[1][0] - rays_o[:, 0]) / (rays_d[:, 0] + EPS_1E10)
        x_ray_bounds = torch.stack([x_min, x_max], dim=-1)
        # noinspection PyTypeChecker
        x_ray_bounds = torch.where(x_ray_bounds[:, :1] > x_ray_bounds[:, 1:], x_ray_bounds[:, [1, 0]], x_ray_bounds)
        final_ray_bounds = x_ray_bounds

        # compute intersections with the Y-planes:
        y_min = (aabb[0][1] - rays_o[:, 1]) / (rays_d[:, 1] + EPS_1E10)
        y_max = (aabb[1][1] - rays_o[:, 1]) / (rays_d[:, 1] + EPS_1E10)
        # noinspection DuplicatedCode
        y_ray_bounds = torch.stack([y_min, y_max], dim=-1)

        # noinspection PyTypeChecker
        y_ray_bounds = torch.where(y_ray_bounds[:, :1] > y_ray_bounds[:, 1:], y_ray_bounds[:, [1, 0]], y_ray_bounds)

        intersecting = torch.where(
            torch.logical_or(
                final_ray_bounds[:, :1] > y_ray_bounds[:, 1:],
                y_ray_bounds[:, :1] > final_ray_bounds[:, 1:],
            ),
            non_intersecting,
            intersecting,
        )

        final_ray_bounds[:, 0] = torch.where(
            y_ray_bounds[:, 0] > final_ray_bounds[:, 0],
            y_ray_bounds[:, 0],
            final_ray_bounds[:, 0],
        )

        final_ray_bounds[:, 1] = torch.where(
            y_ray_bounds[:, 1] < final_ray_bounds[:, 1],
            y_ray_bounds[:, 1],
            final_ray_bounds[:, 1],
        )

        # compute intersections with the Z-planes:
        z_min = (aabb[0][2] - rays_o[:, 2]) / (rays_d[:, 2] + EPS_1E10)
        z_max = (aabb[1][2] - rays_o[:, 2]) / (rays_d[:, 2] + EPS_1E10)
        # noinspection DuplicatedCode
        z_ray_bounds = torch.stack([z_min, z_max], dim=-1)
        # noinspection PyTypeChecker
        z_ray_bounds = torch.where(z_ray_bounds[:, :1] > z_ray_bounds[:, 1:], z_ray_bounds[:, [1, 0]], z_ray_bounds)

        intersecting = torch.where(
            torch.logical_or(
                final_ray_bounds[:, :1] > z_ray_bounds[:, 1:],
                z_ray_bounds[:, :1] > final_ray_bounds[:, 1:],
            ),
            non_intersecting,
            intersecting,
        )

        final_ray_bounds[:, 0] = torch.where(
            z_ray_bounds[:, 0] > final_ray_bounds[:, 0],
            z_ray_bounds[:, 0],
            final_ray_bounds[:, 0],
        )

        final_ray_bounds[:, 1] = torch.where(
            z_ray_bounds[:, 1] < final_ray_bounds[:, 1],
            z_ray_bounds[:, 1],
            final_ray_bounds[:, 1],
        )

        # finally revert the non_intersecting rays to the original scene_bounds:
        final_ray_bounds = torch.where(torch.logical_not(intersecting), orig_ray_bounds, final_ray_bounds)

        # We don't consider the intersections behind the camera
        final_ray_bounds = torch.clip(final_ray_bounds, min=0.0)

        # final_ray_bouunds is a tensor of shape (num_rays, 2) where each row is [near, far]
        # intersecting is a boolean tensor of shape (num_rays, 1) denoting whether the ray intersected the aabb or not.
        final_ray_bounds = torch.where(intersecting, final_ray_bounds, torch.zeros_like(final_ray_bounds))
        nears = final_ray_bounds[:, 0:1]
        fars = final_ray_bounds[:, 1:2]
        return nears, fars


class NearFarCollider(Collider):

    def __init__(self, near_plane: float, far_plane: float) -> None:
        super().__init__(near_plane=near_plane, far_plane=far_plane)

    def collide(self, rays_o: Tensor, rays_d: Tensor) -> Tuple[Tensor, Tensor]:
        ones = torch.ones_like(rays_o[..., 0:1], dtype=rays_o.dtype, device=rays_o.device)
        nears = ones * self.near_plane
        fars = ones * self.far_plane
        return nears, fars
