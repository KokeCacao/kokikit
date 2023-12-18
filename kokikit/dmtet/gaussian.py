# modified from DreamGaussian by kiui
import torch
import math
import numpy as np
import warnings

from typing import Optional, Any, Sequence, List, Dict
from torch import Tensor

GAUSSIAN=False
try:
    GAUSSIAN=True
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as e:
    warnings.warn(f"WARNING: {e}\nNeRF Acceleration (gs) is disabled because you don't have diff_gaussian_rasterization package.", ImportWarning)
    
from simple_knn._C import distCUDA2 # You need to import torch before importing this

from ..utils.utils import quaternion_to_3x3_rotation, scale_quaternion_to_3x3_matrix
from ..nerf.rays import RayBundle
from ..nerf.field_base import FieldBase

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    #  Copyright 2021 The PlenOctree Authors.
    #  Redistribution and use in source and binary forms, with or without
    #  modification, are permitted provided that the following conditions are met:
    #
    #  1. Redistributions of source code must retain the above copyright notice,
    #  this list of conditions and the following disclaimer.
    #
    #  2. Redistributions in binary form must reproduce the above copyright notice,
    #  this list of conditions and the following disclaimer in the documentation
    #  and/or other materials provided with the distribution.
    #
    #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    #  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    #  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    #  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    #  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    #  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    #  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    #  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    #  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    #  POSSIBILITY OF SUCH DAMAGE.

    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1)**2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result + C2[0] * xy * sh[..., 4] + C2[1] * yz * sh[..., 5] + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] + C2[3] * xz * sh[..., 7] + C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result + C3[0] * y * (3 * xx - yy) * sh[..., 9] + C3[1] * xy * z * sh[..., 10] + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] + C3[5] * z * (xx - yy) * sh[..., 14] + C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] + C4[1] * yz * (3 * xx - yy) * sh[..., 17] + C4[2] * xy * (7 * zz - 1) * sh[..., 18] + C4[3] * yz * (7 * zz - 3) * sh[..., 19] + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] + C4[5] * xz * (7 * zz - 3) * sh[..., 21] + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] + C4[7] * xz * (xx - 3 * yy) * sh[..., 23] + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


# TODO: make this into my system
def getProjectionMatrix(znear, zfar, fovX, fovY, dtype, device):
    return torch.tensor([
        [1 / math.tan((fovX / 2)), 0, 0, 0],
        [0, 1 / math.tan((fovY / 2)), 0, 0],
        [0, 0, 1.0 * zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
        [0, 0, 1.0, 0],
    ], dtype=dtype, device=device)


# TODO: make this into my system
class MiniCam:

    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.
        c2w = c2w[0].cuda() # [B, 4, 4] -> [4, 4]

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        # w2c = np.linalg.inv(c2w)
        w2c = torch.inverse(c2w)

        # rectify...
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = w2c.transpose(0, 1) # R is stored transposed due to 'glm' in CUDA code
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
            dtype=self.world_view_transform.dtype,
            device=self.world_view_transform.device,
        ).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -c2w[:3, 3]


class InverseSigmoid(torch.nn.Module):

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(x / (1 - x + self.eps))


class GaussianModel:

    def __init__(
        self,
        sh_degree: int,
        percent_dense: float,
        points: Optional[np.ndarray],
        colors: Optional[np.ndarray],
        num_pts: int,
        radius: float,
        position_lr_init: float,
        feature_lr: float,
        opacity_lr: float,
        scaling_lr: float,
        rotation_lr: float,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        def build_covariance_from_scaling_rotation(scaling: Tensor, scaling_modifier: float, rotation: Tensor):
            L = scale_quaternion_to_3x3_matrix(scale=scaling_modifier * scaling, quaternion=rotation) # L = R S # [B, 3, 3]
            covariance = L @ L.transpose(1, 2) # \sigma = R S S^T R^T = L L^T # [B, 3, 3]

            # Given a symmetric matrix, we can extract the lower triangular part
            # \[
            # \begin{bmatrix}
            # \textbf{a} & - & - \\
            # \textbf{d} & \textbf{e} & - \\
            # \textbf{g} & \textbf{h} & \textbf{i} \\
            # \end{bmatrix}
            # \]
            # The following function extracts the elements in this order: \(a, d, g, e, h, i\).
            uncertainty = torch.zeros((covariance.shape[0], 6), device=covariance.device, dtype=covariance.dtype)
            uncertainty[:, 0] = covariance[:, 0, 0]
            uncertainty[:, 1] = covariance[:, 0, 1]
            uncertainty[:, 2] = covariance[:, 0, 2]
            uncertainty[:, 3] = covariance[:, 1, 1]
            uncertainty[:, 4] = covariance[:, 1, 2]
            uncertainty[:, 5] = covariance[:, 2, 2]
            return uncertainty

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = InverseSigmoid()

        self.rotation_activation = torch.nn.functional.normalize

        # Initialization
        if points is None or colors is None:
            # init from random point cloud

            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            self.create_from_pcd(points=xyz, colors=SH2RGB(shs), spatial_lr_scale=10)
        elif points is not None and colors is not None:
            # load from a provided pcd
            self.create_from_pcd(points=points, colors=colors, spatial_lr_scale=1)
        else:
            # load from saved ply
            raise NotImplementedError

        # Training Setup
        self.percent_dense = percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                'params': [self._xyz],
                'lr': position_lr_init * self.spatial_lr_scale,
                "name": "xyz"
            },
            {
                'params': [self._features_dc],
                'lr': feature_lr,
                "name": "f_dc"
            },
            {
                'params': [self._features_rest],
                'lr': feature_lr / 20.0,
                "name": "f_rest"
            },
            {
                'params': [self._opacity],
                'lr': opacity_lr,
                "name": "opacity"
            },
            {
                'params': [self._scaling],
                'lr': scaling_lr,
                "name": "scaling"
            },
            {
                'params': [self._rotation],
                'lr': rotation_lr,
                "name": "rotation"
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def training_setup(self,):
        return self.optimizer

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.covariance_activation(scaling=self.get_scaling, scaling_modifier=scaling_modifier, rotation=self._rotation)

    def create_from_pcd(self, points: np.ndarray, colors: np.ndarray, spatial_lr_scale: float = 1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1)**2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest, "opacity": new_opacities, "scaling": new_scaling, "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_3x3_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat([selected_pts_mask, torch.zeros(size=(int(N * selected_pts_mask.sum().item()),), device="cuda", dtype=torch.bool)])
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


class Gaussian(FieldBase):

    def __init__(
        self,
        latent_dreamfusion: bool,
        gaussian_model: GaussianModel,
        degree_latent: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            dtype=dtype,
            device=device,
        )

        self.gaussians = gaussian_model

        self.bg_color = torch.tensor(
            [1.0, 1.0, 1.0],
            dtype=dtype,
            device=device,
        )

    # TODO: tune these values and add to configuration
    def before_loop(
        self,
        density_start_iter: int = 0,
        density_end_iter: int = 3000,
        densification_interval: int = 30,
        opacity_reset_interval: int = 700,
        densify_grad_threshold: float = 0.01,
    ):
        self.step: int = getattr(self, "steps", 0)
        if self.step < 1:
            return

        assert self.viewspace_points is not None and self.visibility_filter is not None and self.radii is not None
        if self.step >= density_start_iter and self.step <= density_end_iter:
            self.gaussians.max_radii2D[self.visibility_filter] = torch.max(self.gaussians.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
            self.gaussians.add_densification_stats(self.viewspace_points, self.visibility_filter)

            if self.step % densification_interval == 0:
                # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                self.gaussians.densify_and_prune(max_grad=densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)

            if self.step % opacity_reset_interval == 0:
                self.gaussians.reset_opacity()

        self.viewspace_points = None
        self.visibility_filter = None
        self.radii = None

    def after_loop(self, viewspace_points, visibility_filter, radii):
        self.step += 1
        self.viewspace_points = viewspace_points
        self.visibility_filter = visibility_filter
        self.radii = radii

    def to_device(self, device: torch.device):
        raise NotImplementedError
    
    def parameter_groups(self, lr: float) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    # TODO: properly do renderers
    def _forward(
        self,
        ray_bundle: RayBundle,
        renderers: Sequence[Any],
        scaling_modifier: float = 1.0,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        # mvp: [B, 4, 4]
        self.before_loop()

        H = ray_bundle.origins.shape[1]
        W = ray_bundle.origins.shape[2]
        # WARNING: projection will work differently than ray tracer, expect differences in render results
        mvp: Optional[Tensor] = ray_bundle.mvp
        c2w: Optional[Tensor] = ray_bundle.c2w
        assert c2w is not None and mvp is not None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (torch.zeros_like(
            self.gaussians.get_xyz,
            dtype=self.gaussians.get_xyz.dtype,
            requires_grad=True,
            device=self.gaussians.get_xyz.device,
        ) + 0)
        try:
            screenspace_points.retain_grad()
        except:
            pass

        viewpoint_camera = MiniCam(
            c2w=c2w,
            width=W,
            height=H,
            fovy=ray_bundle.fov_y,
            fovx=ray_bundle.fov_x,
            znear=ray_bundle.near_plane,
            zfar=ray_bundle.far_plane,
        )
        # TODO: shitcode below
        try:
            rasterizer = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
                image_height=H,
                image_width=W,
                tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
                tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
                bg=self.bg_color.to(dtype=torch.float32),
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform.to(dtype=torch.float32),
                projmatrix=viewpoint_camera.full_proj_transform.to(dtype=torch.float32),
                sh_degree=self.gaussians.active_sh_degree,
                campos=viewpoint_camera.camera_center.to(dtype=torch.float32),
                prefiltered=False,
                debug=True,
            ))
        except:
            # TODO: add support for users without gs (espacially MPS users)
            raise NotImplementedError

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if convert_SHs_python:
            shs_view = self.gaussians.get_features.transpose(1, 2).view(-1, 3, (self.gaussians.max_sh_degree + 1)**2)
            dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(self.gaussians.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = self.gaussians.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D, # parameter
            means2D=means2D, # tensor
            shs=shs, # tensor
            colors_precomp=colors_precomp, # None
            opacities=opacity, # tensor
            scales=scales, # tensor
            rotations=rotations, # tensor
            cov3D_precomp=cov3D_precomp, # None
        ) # [C, H, W]

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        self.after_loop(viewspace_points=means2D, visibility_filter=radii > 0, radii=radii)
        # return {
        #     "image": rendered_image,
        #     "depth": rendered_depth,
        #     "alpha": rendered_alpha,
        # }

        rendered_image = rendered_image.permute(1, 2, 0).unsqueeze(0) # [1, H, W, C]

        # TODO: regularization loss, add depth and alpha output
        # calculate [images]
        images: List[Tensor] = [] # List[B, H, W, ?]
        losses: List[Tensor] = [] # List[?]
        for renderer in renderers:
            images.append(rendered_image.clamp(0, 1))
            losses.append(torch.zeros(1, device='cuda'))

        return images, losses
