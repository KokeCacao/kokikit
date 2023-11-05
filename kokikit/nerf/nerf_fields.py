import numpy as np
import torch
import warnings
try:
    import tinycudann as tcnn
except EnvironmentError as e:
    warnings.warn(f"WARNING: {e}\nThis error is fine for CPU-only mode.", ImportWarning)

from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from torch import Tensor
from typing_extensions import Literal
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd # type: ignore[attr-defined]
from ..utils.const import *
from .ray_samplers import RaySamples
from .renderers import SHRenderer
from .density_inits import DensityInit


class FARTNeRFContraction(Enum):
    NO_CONTRACTION = auto()
    L2_CONTRACTION = auto()
    LINF_CONTRACTION = auto()


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


class MLP(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(torch.nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = torch.nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = torch.nn.functional.relu(x, inplace=True)
        return x


class NeRFField(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def to_device(self, device: torch.device):
        raise NotImplementedError

    def get_density(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_color(self, directions: Tensor, color_feature: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class TCNNWithMLP(torch.nn.Module):

    def __init__(self, dim, n_levels, n_features_per_level, base_res, target_res, grid_output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # When using 2nd derivative, FullyFusedMLP will cause the following error. We avoid it using MLP instead.
        # File ".../lib/python3.8/site-packages/tinycudann/modules.py", line 145, in backward
        # doutput_grad, params_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
        # RuntimeError: DifferentiableObject::backward_backward_input_impl: not implemented error
        self.encoder = tcnn.Encoding(
            n_input_dims=dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": 19,
                "base_resolution": base_res,
                "interpolation": "Smoothstep",
                "per_level_scale": np.exp2(np.log2(target_res / base_res) / (n_levels - 1)),
            },
            dtype=torch.float32, # ENHANCE: default float16 seems unstable...
        )
        self.mlp = MLP(self.encoder.n_output_dims, grid_output_dim, ((grid_output_dim + 15) // 16) * 16, 2 + 1, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x


class FARTNeRFField(NeRFField):

    def __init__(
        self,
        aabb_scale: float,
        grid_dims: Tuple[int, ...], # if only cube supported, will use 1st element
        max_sh_order: int,
        color_activation: Union[torch.nn.Module, Callable[..., Any]],
        density_activation: Union[torch.nn.Module, Callable[..., Any]],
        contraction: FARTNeRFContraction,
        density_init: DensityInit,
        background_density_init: float,
        interpolation: Literal["Nearest", "Linear", "Smoothstep"],
        color_degree: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()

        # PARAMETERS
        self.dim: int = len(grid_dims)
        self.aabb_scale: float = aabb_scale
        self.density_init: DensityInit = density_init
        self.density_activation = density_activation
        self.color_activation = color_activation

        self.grid_output_dim = color_degree * ((max_sh_order + 1)**2) + 1

        # STORAGE
        base_res = 16
        n_levels = 16
        target_res = grid_dims[0]
        n_features_per_level = 2

        try:
            # self.composed = tcnn.NetworkWithInputEncoding(
            #     n_input_dims=self.dim,
            #     n_output_dims=self.grid_output_dim,
            #     encoding_config={
            #         "otype": "HashGrid", # HashGrid has detailed texture, but has "white border" artifacts compared to DenseGrid
            #         "n_levels": n_levels,
            #         "n_features_per_level": n_features_per_level,
            #         "log2_hashmap_size": 19,
            #         "base_resolution": base_res,
            #         "interpolation": interpolation,
            #         "per_level_scale": np.exp2(np.log2(target_res / base_res) / (n_levels - 1)),
            #     },
            #     network_config={
            #         "otype": "FullyFusedMLP",
            #         "activation": "ReLU",
            #         "output_activation": "None",
            #         "n_neurons": ((self.grid_output_dim + 15) // 16) * 16, # May only be 16, 32, 64, or 128.
            #         "n_hidden_layers": 1,
            #     },
            # )
            self.composed = TCNNWithMLP(
                dim=self.dim,
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                base_res=base_res,
                target_res=target_res,
                grid_output_dim=self.grid_output_dim,
            )
        except NameError as e:
            self.composed = torch.nn.Linear(1, 1) # dummy
            warnings.warn(f"WARNING: {e}\nThis error is fine for CPU-only mode.", ImportWarning)

        # outside of [-1, 1]^3
        background_rgb = torch.zeros(self.grid_output_dim - 1, dtype=dtype, device=device, requires_grad=False)
        background_density = torch.tensor([background_density_init], dtype=dtype, device=device, requires_grad=False)
        self.background = torch.nn.Parameter(torch.cat([background_rgb, background_density], dim=0), requires_grad=False) # [SH + 1]

        # renderers
        self.sh_renderer: SHRenderer = SHRenderer(
            color_degree=color_degree,
            max_sh_order=max_sh_order,
        )

    def to_device(self, device: torch.device):
        self.composed = self.composed.to(device)
        self.background = self.background.to(device)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return super().parameters(recurse)

    # def contract(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
    #     # Assuming center of the scene is at the origin, and the scene is scaled to [-1, 1]^3
    #     # Note that the function should only be applied at the end of sampling pipeline
    #     # It only transform point positions, not a Frustum
    #     mag_raw: Tensor = torch.linalg.norm(positions, ord=float("inf"), dim=-1)
    #     mag = mag_raw[..., None]

    #     positions = torch.where(mag > 1.0, (2 - (1 / mag)) * (positions / mag), positions)
    #     # At this line, uncontracted positions are in [-1, 1]^3, contracted positions are in [-2, 2]^3
    #     positions = positions / 2.0
    #     # At this line, uncontracted positions are in [-0.5, 0.5]^3, contracted positions are in [-1, 1]^3
    #     contracted_mask = torch.where(mag_raw > 1.0, True, False)
    #     return positions, contracted_mask

    def get_density(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        original_positions = (
            # everything in [-aabb_scale, aabb_scale]^3 (depending on SceneBox from DataParser) if NO_CONTRACTION
            # positions are from raw ray's origin, then clamped by Collider by SceneBox
            # then sampled along ray's segment by Sampler
            # Note that SceneBox only change box in viewport, and maybe sampling region (depends on Collider)
            # It does not scale the scene or ray in any way by default
            positions) # [N_ray, N_sample, 3]

        n_ray: int = positions.shape[-3]
        n_sample: int = positions.shape[-2]
        assert self.dim == positions.shape[-1]

        # CONTRACTION and GRID INTERPOLATION
        positions = positions.reshape(-1, 3) # [N_ray * N_sample, 3]
        # positions, contracted_mask = self.contract(positions) # [N_ray * N_sample, 3], [N_ray * N_sample]

        # background: everything outside of [-1, 1]^3
        background_positions_selector = torch.linalg.norm(positions, ord=float("inf"), dim=-1) > 1.0
        foreground_positions = positions[~background_positions_selector] # [N_ray * N_sample, 3]

        features_densities: Tensor
        if foreground_positions.shape[0] > 0:
            # foreground is not empty -> go through the inference

            foreground_positions = (foreground_positions + 1.0) / 2.0 # [N_ray * N_sample, 3]
            assert torch.all(torch.logical_and(
                foreground_positions >= 0.0,
                foreground_positions <= 1.0,
            ))
            features_densities_foreground = self.composed(foreground_positions).view(-1, self.grid_output_dim) # [N_ray * N_sample, (SH + 1)]

            # WARNING: for some reason, nvidia-tcnn will always give [torch.float16] for features_densities_foreground
            features_densities_foreground = features_densities_foreground.to(dtype=foreground_positions.dtype) # [N_ray * N_sample, (SH + 1)]

            features_densities = self.background.expand(n_ray * n_sample, -1).clone() # [N_ray * N_sample, (SH + 1)]
            features_densities[~background_positions_selector, :] = features_densities_foreground

        else:
            # foreground is empty -> everything is background
            features_densities = self.background.expand(n_ray * n_sample, -1) # [N_ray * N_sample, (SH + 1)]

        features, densities = torch.split(features_densities, [features_densities.shape[-1] - 1, 1], dim=-1) # [N, SH], [N, 1]

        # edit density
        densities = self.density_init.edit_density(densities, original_positions) # [N, 1]

        features = features.reshape(-1, n_sample, features.shape[-1]) # [N_ray, N_sample, SH]
        densities = densities.reshape(-1, n_sample, densities.shape[-1]) # [N_ray, N_sample, 1]
        return self.density_activation(densities), features # [N_ray, N_sample, 1], [N_ray, N_sample, SH]

    def get_color(self, directions: Tensor, color_feature: Tensor) -> Tensor:
        # SPHERICAL HARMONICS
        # directions [N_rays, N_sample, 3]

        sh = self.sh_renderer(sh=color_feature, directions=directions) # [N_ray, N_sample, C]
        return self.color_activation(sh) # [N_ray, N_sample, C]

    def forward(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        # accept both collated or non-collated ray_samples
        BHW = ray_samples.center.shape[:-2]
        N = ray_samples.center.shape[-2]
        D = ray_samples.center.shape[-1]
        positions = ray_samples.center.reshape(-1, N, D) # [B, H, W, num_samples, D] -> [B*H*W, num_samples, D]
        directions = ray_samples.directions.reshape(-1, 1, D) # [B, H, W, D] -> [B*H*W, 1, D]
        directions = directions.expand(-1, N, -1) # [B*H*W, 1, D] -> [B*H*W, N_sample, D]

        densities, color_feature = self.get_density(positions=positions) # [B*H*W, N_sample, 1], [B*H*W, N_sample, SH]

        colors = self.get_color(directions=directions, color_feature=color_feature) # [B*H*W, N_sample, C]

        assert torch.isfinite(colors).all(), f"colors is not finite: {colors}"
        # density is allowed to contain inf
        return colors.reshape(*BHW, N, -1), densities.reshape(*BHW, N, -1) # [B, H, W, num_samples, C], [B, H, W, num_samples, 1]
