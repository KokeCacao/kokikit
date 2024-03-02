import random
import torch
import io
import base64
import numpy as np

from typing import Tuple, Optional, List, Dict, Any
from torch import Tensor
from PIL import Image

from ..utils.utils import safe_normalize
from ..nerf.rays import RayBundle
from ..nerf.colliders import Collider
from .dataset import Dataset


class NeRFDataset(Dataset):

    def __init__(
        self,
        collider: Collider,
        radius_min: float,
        radius_max: float,
        theta_min: float,
        theta_max: float,
        phi_min: float,
        phi_max: float,
        uniform_sphere_rate: float,
        fov_y_min: float,
        fov_y_max: float,
        device: torch.device,
    ) -> None:
        self.collider: Collider = collider
        self.radius_min: float = radius_min
        self.radius_max: float = radius_max
        self.theta_min: float = theta_min
        self.theta_max: float = theta_max
        self.phi_min: float = phi_min
        self.phi_max: float = phi_max
        self.uniform_sphere_rate: float = uniform_sphere_rate
        self.fov_y_min: float = fov_y_min
        self.fov_y_max: float = fov_y_max
        self.near_plane: float = collider.near_plane
        self.far_plane: float = collider.far_plane
        self.device: torch.device = device

    def get_train_ray_bundle(
        self,
        cx_latent_dataset: float,
        cy_latent_dataset: float,
        h_latent_dataset: int,
        w_latent_dataset: int,
        batch_size: int,
        mv_dream_views: Optional[int] = None,
    ) -> RayBundle:
        radius_min = self.radius_min
        radius_max = self.radius_max
        theta_min = self.theta_min
        theta_max = self.theta_max
        phi_min = self.phi_min
        phi_max = self.phi_max
        uniform_sphere_rate = self.uniform_sphere_rate

        fov_y_min = self.fov_y_min
        fov_y_max = self.fov_y_max
        cx_latent = cx_latent_dataset
        cy_latent = cy_latent_dataset
        h_latent = h_latent_dataset
        w_latent = w_latent_dataset
        near_plane = self.near_plane
        far_plane = self.far_plane
        device = self.device

        # extrinsic
        if mv_dream_views is not None:
            thetas, phis, rays_o = self._get_rays_o_mvdream(
                batch_size=batch_size,
                radius_min=radius_min,
                radius_max=radius_max,
                n_views=mv_dream_views,
                device=device,
            )
        else:
            thetas, phis, rays_o = self._get_rays_o(
                batch_size=batch_size,
                radius_min=radius_min,
                radius_max=radius_max,
                theta_min=theta_min,
                theta_max=theta_max,
                phi_min=phi_min,
                phi_max=phi_max,
                uniform_sphere_rate=uniform_sphere_rate,
                device=device,
            ) # [B,], [B,], [B, 3]
        up_vector, right_vector, forward_vector = self._get_lookat(
            batch_size=batch_size,
            rays_o=rays_o,
            targets=torch.zeros_like(rays_o),
            device=device,
        ) # [B, 3], [B, 3], [B, 3]

        c2w = self._get_c2w(
            batch_size=batch_size,
            up_vector=up_vector,
            right_vector=right_vector,
            forward_vector=forward_vector,
            rays_o=rays_o,
            device=device,
        ) # [B, 4, 4], [:, :3, 3] is rays_o

        # intrinsic
        fov_y: float = np.random.rand() * (fov_y_max - fov_y_min) + fov_y_min # scalar cuz we want projection matrix to be the same for a batch
        focal: float = h_latent / (2 * np.tan(fov_y / 2)) # [1,]
        fov_x: float = 2 * np.arctan(w_latent / (2 * focal)) # [1,]
        # assert focal == w_latent / (2 * np.tan(fov_x / 2))

        rays_d = self._get_rays(
            batch_size=batch_size,
            w_latent=w_latent,
            h_latent=h_latent,
            c2w=c2w,
            focal=focal,
            cx=cx_latent,
            cy=cy_latent,
            device=device,
        ) # [B, H, W, 3]
        rays_o = rays_o[:, None, None, :].expand(-1, h_latent, w_latent, -1) # [B, H, W, 3]

        projection = self._get_projection(
            focal=focal,
            h=h_latent, # TODO: don't know whether its h_img or h_latent
            w=w_latent,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device,
        )

        mvp = self._get_mvp(
            projection=projection,
            c2w=c2w,
        ) # [B, 4, 4]

        ray_bundle = RayBundle(
            near_plane=near_plane,
            far_plane=far_plane,
            fov_x=fov_x,
            fov_y=fov_y,
            origins=rays_o,
            directions=rays_d,
            forward_vector=forward_vector,
            collider=self.collider,
            nears=None,
            fars=None,
            mvp=mvp,
            c2w=c2w,
        ) # [B, H, W, 3], [B, H, W, 3], [B,]

        return ray_bundle

    def get_eval_ray_bundle(self, h_latent: int, w_latent: int, c2w: Tensor, fov_y: float) -> RayBundle:
        near_plane = self.near_plane
        far_plane = self.far_plane
        device = self.device

        c2w = c2w.unsqueeze(0).to(device=device) # [1, 4, 4]
        focal: float = h_latent / (2 * np.tan(fov_y / 2)) # focal length isn't very reliable due to units ambiguity
        fov_x: float = 2 * np.arctan(w_latent / (2 * focal)) # [1,]
        # assert focal == w_latent / (2 * np.tan(fov_x / 2))
        rays_d = self._get_rays(
            batch_size=1,
            w_latent=w_latent, # note that we don't use config value
            h_latent=h_latent, # note that we don't use config value
            c2w=c2w,
            focal=focal,
            cx=w_latent / 2,
            cy=h_latent / 2,
            device=device,
        ) # [1, H, W, 3]
        rays_o = c2w[:, :3, 3] # [1, 3]
        up_vector, right_vector, forward_vector = self._get_lookat(
            batch_size=1,
            rays_o=rays_o,
            targets=torch.zeros_like(rays_o),
            device=device,
        ) # [1, 3], [1, 3], [1, 3]

        projection = self._get_projection(
            focal=focal,
            h=h_latent, # TODO: don't know whether its h_img or h_latent
            w=w_latent,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device,
        )

        mvp = self._get_mvp(
            projection=projection,
            c2w=c2w,
        ) # [B, 4, 4]

        rays_o = rays_o[:, None, None, :].expand(-1, h_latent, w_latent, -1) # [1, H, W, 3]
        return RayBundle(
            near_plane=near_plane,
            far_plane=far_plane,
            fov_x=fov_x,
            fov_y=fov_y,
            origins=rays_o,
            directions=rays_d,
            collider=self.collider,
            nears=None,
            fars=None,
            forward_vector=forward_vector,
            mvp=mvp,
            c2w=c2w,
        ) # [1, H, W, 3], [1, H, W, 3], [1,]

    def get_test_ray_bundle(self, cx_latent_dataset: float, cy_latent_dataset: float, h_latent_dataset: int, w_latent_dataset: int, batch_size: int, idx: Optional[Tensor], super_resolution: int = 1) -> RayBundle:
        radius = (self.radius_min + self.radius_max) / 2
        fov_y = (self.fov_y_min + self.fov_y_max) / 2

        cx_latent = cx_latent_dataset * super_resolution
        cy_latent = cy_latent_dataset * super_resolution
        h_latent = h_latent_dataset * super_resolution
        w_latent = w_latent_dataset * super_resolution
        near_plane = self.near_plane
        far_plane = self.far_plane
        device = self.device

        selected_batch_size = len(idx) if idx is not None else batch_size

        # extrinsic
        # thetas, phis, rays_o = self._get_sphere_rays_o(
        #     batch_size=batch_size, # Note that here is [batch_size] instead of [selected_batch_size]
        #     radius_float=radius,
        #     loops=2,
        #     idx=idx,
        #     device=device,
        # )
        thetas, phis, rays_o = self._get_round_rays_o(
            batch_size=batch_size, # Note that here is [batch_size] instead of [selected_batch_size]
            radius_float=radius,
            idx=idx,
            device=device,
        )
        up_vector, right_vector, forward_vector = self._get_lookat(
            batch_size=selected_batch_size,
            rays_o=rays_o,
            targets=torch.zeros_like(rays_o),
            device=device,
        ) # [B, 3], [B, 3], [B, 3]
        c2w = self._get_c2w(
            batch_size=selected_batch_size,
            up_vector=up_vector,
            right_vector=right_vector,
            forward_vector=forward_vector,
            rays_o=rays_o,
            device=device,
        ) # [B, 4, 4], [:, :3, 3] is rays_o

        # intrinsic
        focal: float = h_latent / (2 * np.tan(fov_y / 2)) # [1,]
        fov_x: float = 2 * np.arctan(w_latent / (2 * focal)) # [1,]
        # assert focal == w_latent / (2 * np.tan(fov_x / 2))

        rays_d = self._get_rays(
            batch_size=selected_batch_size,
            w_latent=w_latent,
            h_latent=h_latent,
            c2w=c2w,
            focal=focal,
            cx=cx_latent,
            cy=cy_latent,
            device=device,
        ) # [B, H, W, 3]
        rays_o = rays_o[:, None, None, :].expand(-1, h_latent, w_latent, -1) # [B, H, W, 3]

        projection = self._get_projection(
            focal=focal,
            h=h_latent, # TODO: don't know whether its h_img or h_latent
            w=w_latent,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device,
        )

        mvp = self._get_mvp(
            projection=projection,
            c2w=c2w,
        ) # [B, 4, 4]

        ray_bundle = RayBundle(
            near_plane=near_plane,
            far_plane=far_plane,
            fov_x=fov_x,
            fov_y=fov_y,
            origins=rays_o,
            directions=rays_d,
            forward_vector=forward_vector,
            collider=self.collider,
            nears=None,
            fars=None,
            mvp=mvp,
            c2w=c2w,
        ) # [B, H, W, 3], [B, H, W, 3], [B,]
        return ray_bundle

    @staticmethod
    def _get_projection(focal: float, h: int, w: int, near_plane: float, far_plane: float, device: torch.device) -> Tensor:
        # WARNING: this function is not tested
        return torch.tensor([
            [2 * focal / w, 0, 0, 0],
            [0, -2 * focal / h, 0, 0],
            [0, 0, -(far_plane + near_plane) / (far_plane - near_plane), -(2 * far_plane * near_plane) / (far_plane - near_plane)],
            [0, 0, -1, 0],
        ], device=device).unsqueeze(0) # [1, 4, 4]

    @staticmethod
    def _get_mvp(projection: Tensor, c2w: Tensor) -> Tensor:
        return projection.to(dtype=c2w.dtype) @ torch.inverse(c2w) # [1, 4, 4]

    @staticmethod
    def _get_rays(batch_size: int, w_latent: int, h_latent: int, c2w: Tensor, focal: float, cx: float, cy: float, device: torch.device) -> Tensor:
        # get un-normalized ray direction

        W = w_latent
        H = h_latent
        B = batch_size

        xs, ys = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(H - 1, 0, H, device=device), # small y value in image space means high value y in camera space
            indexing='xy',
        )
        xs = xs.reshape([1, H, W]).expand([B, H, W]) + 0.5 # [B, H, W], shift right
        ys = ys.reshape([1, H, W]).expand([B, H, W]) + 0.5 # [B, H, W], shift up

        zs = -torch.ones_like(xs) # pointing to -z axis
        xs = (xs - cx) / focal # = (xs - cx) / h_latent * (2 * np.tan(fov_y / 2))
        ys = (ys - cy) / focal # so that direction isn't affected by image size
        directions = torch.stack((xs, ys, zs), dim=-1) # [B, H, W, 3]
        rays_d = torch.einsum('bjk,bhwk->bhwj', c2w[:, :3, :3], directions) # [B, H, W, 3], don't translate since it is vector
        rays_d = safe_normalize(rays_d)

        return rays_d # [B, H, W, 3]

    @staticmethod
    def _get_c2w(batch_size: int, right_vector: Tensor, up_vector: Tensor, forward_vector: Tensor, rays_o: Tensor, device: torch.device) -> Tensor:
        c2w = torch.eye(4, dtype=forward_vector.dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1) # [B, 4, 4]
        c2w[:, :3, :3] = torch.stack((right_vector, up_vector, -forward_vector), dim=-1) # -z direction is camera's forward
        c2w[:, :3, 3] = rays_o
        # c2w matrix after execution:
        # [[r_x, u_x, -f_x, o_x]
        #  [r_y, u_y, -f_y, o_y]
        #  [r_z, u_z, -f_z, o_z]
        #  [  0,   0,    0,   1]]
        # we negate front vector because camera looks at -z
        
        # sanity check
        # _thetas = torch.acos(-forward_vector[:, 1])
        # _phis = torch.atan2(-forward_vector[:, 0], -forward_vector[:, 2])
        # _phis[_phis < 0] += 2 * np.pi
        # assert torch.allclose(thetas, _thetas)
        # assert torch.allclose(phis, _phis)
        return c2w # [B, 4, 4]
    
    @staticmethod
    def _get_thetas_phis_r(c2w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        forward_vector = c2w[:, :3, 2]
        thetas = torch.acos(-forward_vector[:, 1])
        phis = torch.atan2(-forward_vector[:, 0], -forward_vector[:, 2])
        phis[phis < 0] += 2 * np.pi
        r = torch.norm(c2w[:, :3, 3], dim=-1)
        return thetas, phis, r

    @staticmethod
    def _get_rays_o(
        batch_size: int,
        radius_min: float,
        radius_max: float,
        theta_min: float,
        theta_max: float,
        phi_min: float,
        phi_max: float,
        uniform_sphere_rate: float,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        radius = torch.rand(batch_size, 1, device=device) * (radius_max - radius_min) + radius_min
        if random.random() < uniform_sphere_rate:
            # WARNING: not tested
            unit_centers = torch.nn.functional.normalize(torch.stack([
                torch.randn(batch_size, device=device),
                torch.abs(torch.randn(batch_size, device=device)),
                torch.randn(batch_size, device=device),
            ], dim=-1), p=2, dim=1) # [B, 3]
            thetas = torch.acos(unit_centers[:, 1]) # [B,]
            phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2]) # [B,]
            phis[phis < 0] += 2 * np.pi
            ray_o = unit_centers * radius
            return thetas, phis, ray_o # [B,], [B,], [B, 3]
        else:
            # TODO: this sampling method is biased towards polar
            thetas = torch.rand(batch_size, device=device) * (theta_max - theta_min) + theta_min
            phis = torch.rand(batch_size, device=device) * (phi_max - phi_min) + phi_min
            phis[phis < 0] += 2 * np.pi

            # default view is [sin(90) * sin(0), cos(90), sin(90) * cos(0)] = [0, 0, 1]
            # rotate to [sin(90) * sin(90), cos(90), sin(90) * cos(90)] = [1, 0, 0]
            # up is [sin(0), cos(0), sin(0)] = [0, 1, 0]
            ray_o = radius * torch.stack([
                torch.sin(thetas) * torch.sin(phis),
                torch.cos(thetas),
                torch.sin(thetas) * torch.cos(phis),
            ], dim=-1) # [B, 3]
            return thetas, phis, ray_o # [B,], [B,], [B, 3]

    @staticmethod
    def _get_rays_o_mvdream(
        batch_size: int,
        radius_min: float,
        radius_max: float,
        device: torch.device,
        theta_min: float = (90 - 30) / 180 * np.pi,
        theta_max: float = (90 + 0) / 180 * np.pi,
        phi_min: float = -180 / 180 * np.pi,
        phi_max: float = 180 / 180 * np.pi,
        n_views: int = 4,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        real_batch_size = batch_size // n_views

        # [b1v1, b1v2, b1v3, b1v4, b2v1, b2v2, b2v3, b2v4, ...] b=batch, v=view
        radius = (torch.rand(real_batch_size, device=device) * (radius_max - radius_min) + radius_min).repeat_interleave(n_views, dim=0)
        thetas = (torch.rand(real_batch_size, device=device) * (theta_max - theta_min) + theta_min).repeat_interleave(n_views, dim=0)
        rad = torch.rand(real_batch_size, device=device).reshape(-1, 1) # 0~1 -> 0~1, 1~2, 2~3, ...
        phis = (rad + torch.arange(n_views, device=device).reshape(1, -1)).reshape(-1) / n_views * (phi_max - phi_min) + phi_min
        phis[phis < 0] += 2 * np.pi

        # default view is [sin(90) * sin(0), cos(90), sin(90) * cos(0)] = [0, 0, 1]
        # rotate to [sin(90) * sin(90), cos(90), sin(90) * cos(90)] = [1, 0, 0]
        # up is [sin(0), cos(0), sin(0)] = [0, 1, 0]
        ray_o = radius.unsqueeze(-1) * torch.stack([
            torch.sin(thetas) * torch.sin(phis),
            torch.cos(thetas),
            torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]
        return thetas, phis, ray_o # [B,], [B,], [B, 3]

class NeRFSingleImageDataset(NeRFDataset):
    def __init__(
        self,
        image: Tensor,
        collider: Collider,
        radius: float,
        theta: float,
        phi: float,
        fov_y: float,
        device: torch.device,
    ) -> None:
        super().__init__(
            collider=collider,
            radius_min=radius,
            radius_max=radius,
            theta_min=theta,
            theta_max=theta,
            phi_min=phi,
            phi_max=phi,
            uniform_sphere_rate=0.0,
            fov_y_min=fov_y,
            fov_y_max=fov_y,
            device=device,
        )
        self.fov_y = fov_y
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        self.frames = image.to(device=device) # [1, 3, H, W]
        self.H = image.shape[-2]
        self.W = image.shape[-1]
        self.ray_bundle = super().get_train_ray_bundle(
            cx_latent_dataset=self.W / 2,
            cy_latent_dataset=self.H / 2,
            h_latent_dataset=self.H,
            w_latent_dataset=self.W,
            batch_size=1,
            mv_dream_views=None,
        )

    def get_train_images_and_ray(self, batch_size: int) -> Tuple[Tensor, RayBundle]:
        assert batch_size == 1, f"batch_size must be 1 for NeRFSingleImageDataset, but got {batch_size}"
        images = self.get_train_images() # [1, 3, H, W]
        ray_bundle = self.get_train_ray_bundle()
        return images, ray_bundle
    
    def get_train_images(self) -> Tensor:
        return self.frames # [1, 3, H, W]
        
    def get_train_ray_bundle(
        self
    ) -> RayBundle:
        return self.ray_bundle
        
    def get_nerf_panel_info(self, scale_factor=0.01) -> List[Dict[str, Any]]:
        # Used for:
        # self.set_output("render", {
        #     "dataset": [
        #         {
        #             "c2w": nerf_dataset.c2w,
        #             "image": nerf_dataset.image,
        #             "focal": nerf_dataset.focal,
        #         },
        #        ...
        #     ]
        # })
        focal = self.H / (2 * np.tan(self.fov_y / 2)) * scale_factor
        w = int(self.W * scale_factor)
        h = int(self.H * scale_factor)

        # fov_x = 2 * np.arctan(self.json["w"] * scale_factor / (2 * focal))
        # fov_y = 2 * np.arctan(self.json["h"] * scale_factor / (2 * focal))
        # print(f"FOV remote: {math.degrees(fov_x)}, {math.degrees(fov_y)}")

        # convert self.frames[0] to PIL from gpu tensor in range [-1, 1]
        image = ((self.frames[0].cpu().detach().numpy() + 1) / 2 * 255).astype(np.uint8)
        image = Image.fromarray(image.transpose(1, 2, 0)).resize((w, h)) # rescale

        # Convert to base64
        byte = io.BytesIO()
        image.save(byte, format="PNG")
        image = "data:image/png;base64," + base64.b64encode(byte.getvalue()).decode("utf-8")

        assert self.ray_bundle.c2w is not None
        c2w = self.ray_bundle.c2w[0].cpu().detach().numpy().tolist()
        return [{
                "c2w": c2w,
                "image": image,
                "focal": focal,
            }]