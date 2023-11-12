import nerfacc
import torch
import warnings

from typing import Iterator, Optional, Union, Callable, Tuple, List, Dict, Any
from torch import Tensor
from typing_extensions import Literal
from torch.nn.parameter import Parameter

from ..utils.const import *
from .ray_samplers import RaySamples
from .regularizations import Regularization


class Renderer(torch.nn.Module):

    def __init__(self, need_decode: bool, reg: Optional[Regularization] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # For visualizing the latent, we don't need to decode it
        self.need_decode = need_decode
        self.reg = reg

    def to_device(self, device: torch.device):
        # Sampler will usually read device from input
        return self

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward_loss(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        x = self.forward(*args, **kwargs)
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.reg is not None:
            loss = self.reg.forward(x)
        return x, loss

    def parameter_groups(self, lr: float) -> List[Dict[str, Any]]:
        groups = []
        for name, module in self.named_children():
            if hasattr(module, 'parameter_groups'):
                groups.extend(module.parameter_groups(lr=lr))
            else:
                groups.append({'params': module.parameters(), 'lr': lr})
        return groups

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)


class ImageRenderer(Renderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class NeRFRenderer(Renderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class IdentityRenderer(ImageRenderer, NeRFRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, color: Tensor, B: int, H: int, W: int, C: int) -> Tensor:
        return color.view(B, H, W, C)


class RGBRenderer(NeRFRenderer):

    def __init__(self, background_color: Union[Literal["random", "last_sample", "black"], Tensor, torch.nn.Parameter], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Note that background color here is after activation (whether it's sigmoid or tanh or no activation)
        self.background_color = background_color

    def parameter_groups(self, lr: float) -> List[Dict[str, Any]]:
        return [{'params': self.parameters(), 'lr': lr * 0.1}]

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if type(self.background_color) == torch.nn.Parameter:
            return iter([self.background_color])
        return iter([])

    def combine_colors(
        self,
        rgb: Tensor,
        weights: Tensor,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        device = rgb.device
        dtype = rgb.dtype

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if self.background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_rgb = nerfacc.accumulate_along_rays(weights, ray_indices, rgb, num_rays)
            accumulated_weight = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
        else:
            comp_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        background_color: Tensor
        if self.background_color == "last_sample":
            background_color = rgb[..., -1, :]
        elif self.background_color == "random":
            background_color = torch.rand_like(comp_rgb, device=device, dtype=dtype)
        elif self.background_color == "black":
            background_color = torch.zeros_like(comp_rgb, device=device, dtype=dtype)
        elif type(self.background_color) == Tensor:
            background_color = self.background_color
        elif type(self.background_color) == torch.nn.Parameter:
            background_color = self.background_color
        else:
            raise ValueError(f"Background color {self.background_color} not implemented")

        return comp_rgb + background_color * (1.0 - accumulated_weight)

    def forward(
        self,
        colors: Tensor,
        weights: Tensor,
        min: float,
        max: float,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        rgb = self.combine_colors(colors, weights, ray_indices=ray_indices, num_rays=num_rays)
        if torch.any(rgb > max) or torch.any(rgb < min):
            warnings.warn(f"Clipping rgb values to [{min}, {max}], current range is [{rgb.min()}, {rgb.max()}]")
            rgb = torch.clamp(rgb, min=min, max=max)
        return rgb


class RGBMaxRenderer(RGBRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def combine_colors(
        self,
        rgb: Tensor,
        weights: Tensor,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        device = rgb.device
        dtype = rgb.dtype

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if self.background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_rgb = nerfacc.accumulate_along_rays(weights, ray_indices, rgb, num_rays)
            accumulated_weight = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
        else:
            # comp_rgb = torch.sum(weights * rgb, dim=-2)
            max_weight, max_indices = torch.max(weights, dim=-2, keepdim=True)
            comp_rgb = torch.gather(rgb, -2, max_indices.expand(*([-1] * (max_indices.dim() - 1)), 3)).squeeze(-2)
            accumulated_weight = max_weight.squeeze(-2)
            # accumulated_weight = 1.0

        background_color: Tensor
        if self.background_color == "last_sample":
            background_color = rgb[..., -1, :]
        elif self.background_color == "random":
            background_color = torch.rand_like(comp_rgb, device=device, dtype=dtype)
        elif self.background_color == "black":
            background_color = torch.zeros_like(comp_rgb, device=device, dtype=dtype)
        elif type(self.background_color) == Tensor:
            background_color = self.background_color
        elif type(self.background_color) == torch.nn.Parameter:
            background_color = self.background_color
        else:
            raise ValueError(f"Background color {self.background_color} not implemented")

        return comp_rgb + background_color * (1.0 - accumulated_weight)


class NormalAlignmentRenderer(NeRFRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
            self,
            normals, # [B, H, W, S, 3]
            weights: Tensor, # [B, H, W, S, 1]
            directions: Tensor, # [B, H, W, 3]
    ) -> Tensor:
        directions = directions.unsqueeze(-2) # [B, H, W, 1, 3]
        alignment = (normals * directions).sum(-1).clamp(min=0)**2 # [B, H, W, S]
        alignment = alignment * weights.detach().squeeze(-1) # [B, H, W, S]
        alignment = torch.sum(alignment, dim=-1, keepdim=True) # [B, H, W, 1]
        return alignment


class NormalRenderer(RGBRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(background_color="black", *args, **kwargs)

    def forward(
        self,
        normals,
        weights: Tensor,
        min: float,
        max: float,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        colors = (normals + 1) / 2 # [-1, 1] -> [0, 1]
        return super().forward(colors=colors, weights=weights, min=min, max=max, ray_indices=ray_indices, num_rays=num_rays)


class LambertianRenderer(RGBRenderer):

    def __init__(self, ambient: float, background_color: Union[Literal["random", "last_sample", "black"], Tensor], *args, **kwargs) -> None:
        super().__init__(background_color=background_color, *args, **kwargs)
        self.ambient = ambient

    def forward(
        self,
        colors: Tensor,
        lambertian: Tensor,
        weights: Tensor,
        min: float,
        max: float,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        colors = colors * (self.ambient + (1 - self.ambient) * lambertian).unsqueeze(-1)
        return super().forward(colors=colors, weights=weights, min=min, max=max, ray_indices=ray_indices, num_rays=num_rays)


class TexturelessRenderer(LambertianRenderer):

    def __init__(self, const_color: Tensor, ambient: float, background_color: Union[Literal["random", "last_sample", "black"], Tensor], *args, **kwargs) -> None:
        super().__init__(ambient=ambient, background_color=background_color, *args, **kwargs)
        self.const_color = const_color

    def to_device(self, device: torch.device):
        self.const_color = self.const_color.to(device)
        return super().to_device(device)

    def forward(
        self,
        lambertian: Tensor,
        weights: Tensor,
        min: float,
        max: float,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        return super().forward(colors=self.const_color, lambertian=lambertian, weights=weights, min=min, max=max, ray_indices=ray_indices, num_rays=num_rays)


class SHRenderer(torch.nn.Module):

    def __init__(
        self,
        color_degree: int,
        max_sh_order: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.color_degree = color_degree
        self.level = max_sh_order + 1
        self.n_components = self.level**2
        self.input_dim = self.color_degree * self.n_components

    def forward(self, sh: Tensor, directions: Tensor) -> Tensor:
        # sh: [N_ray, N_sample, SH]
        # directions: [N_ray, N_sample, 3]
        sh = sh.view(*sh.shape[:-1], self.color_degree, self.n_components) # [N_ray, N_sample, color_degree, sh_components]
        components = self.components_from_spherical_harmonics(levels=self.level, directions=directions) # [N_ray, N_sample, sh_components]
        rgb = sh * components[..., None, :] # [N_ray, N_sample, color_degree, sh_components]
        rgb = torch.sum(sh, dim=-1) + 0.5 # [N_ray, N_sample, color_degree]
        return rgb

    @staticmethod
    def components_from_spherical_harmonics(levels: int, directions: Tensor) -> Tensor:
        num_components = levels**2
        components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

        assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
        assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

        x = directions[..., 0]
        y = directions[..., 1]
        z = directions[..., 2]

        xx = x**2
        yy = y**2
        zz = z**2

        # l0
        components[..., 0] = 0.28209479177387814

        # l1
        if levels > 1:
            components[..., 1] = 0.4886025119029199 * y
            components[..., 2] = 0.4886025119029199 * z
            components[..., 3] = 0.4886025119029199 * x

        # l2
        if levels > 2:
            components[..., 4] = 1.0925484305920792 * x * y
            components[..., 5] = 1.0925484305920792 * y * z
            components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
            components[..., 7] = 1.0925484305920792 * x * z
            components[..., 8] = 0.5462742152960396 * (xx - yy)

        # l3
        if levels > 3:
            components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
            components[..., 10] = 2.890611442640554 * x * y * z
            components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
            components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
            components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
            components[..., 14] = 1.445305721320277 * z * (xx - yy)
            components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

        # l4
        if levels > 4:
            components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
            components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
            components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
            components[..., 19] = 0.6690465435572892 * y * (7 * zz - 3)
            components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
            components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
            components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
            components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
            components[..., 24] = 0.4425326924449826 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return components


class AccumulationRenderer(NeRFRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        weights: Tensor,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            accumulation = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
        else:
            accumulation = torch.sum(weights, dim=-2)

        if torch.any(accumulation > 1.0) or torch.any(accumulation < 0.0):
            warnings.warn(f"Clipping accumulation values to [{0.0}, {1.0}], current range is [{accumulation.min()}, {accumulation.max()}]")
            accumulation = torch.clamp(accumulation, 0.0, 1.0)
        return accumulation


class DepthRenderer(NeRFRenderer):
    # Depth Method:
    #     - median: Depth is set to the distance where the accumulated weight reaches 0.5.
    #     - expected: Expected depth along ray. Same procedure as rendering rgb, but with depth.

    def __init__(self, method: Literal["median", "expected"], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method = method

    def forward(
        self,
        weights: Tensor,
        ray_samples: RaySamples,
        normalize: bool,
        nears: Optional[Tensor] = None,
        fars: Optional[Tensor] = None,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        assert ray_samples.euclidean_starts is not None
        assert ray_samples.euclidean_ends is not None

        steps = ((ray_samples.euclidean_starts + ray_samples.euclidean_ends) / 2).unsqueeze(-1) # [B, H, W, num_samples] -> [B, H, W, num_samples, 1]
        if self.method == "median":
            if ray_indices is not None and num_rays is not None:
                raise NotImplementedError("Median depth calculation is not implemented for packed samples.")
            cumulative_weights = torch.cumsum(weights[..., 0], dim=-1) # [..., num_samples]
            split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5 # [..., 1]
            median_index = torch.searchsorted(cumulative_weights, split, side="left") # [..., 1]
            median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1) # [..., 1]

            depth = torch.gather(steps[..., 0], dim=-1, index=median_index) # [..., 1]
        elif self.method == "expected":
            if ray_indices is not None and num_rays is not None:
                # Necessary for packed samples from volumetric ray sampler
                depth = nerfacc.accumulate_along_rays(weights, ray_indices, steps, num_rays)
                accumulation = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
                depth = depth / (accumulation + EPS_1E10)
            else:
                depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + EPS_1E10)

            depth = torch.clip(depth, steps.min(), steps.max())
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

        if normalize:
            if nears is None:
                nears = torch.min(depth)
            if fars is None:
                fars = torch.max(depth)
            depth = 1 - ((depth - nears) / (fars - nears + EPS_1E10)) # [B, H, W, 1]

            if torch.any(depth > 1.0) or torch.any(depth < 0.0):
                warnings.warn(f"Clipping depth values to [{0.0}, {1.0}], current range is [{depth.min()}, {depth.max()}]")
                depth = torch.clamp(depth, 0.0, 1.0)
        return depth


class DeltaDepthRenderer(DepthRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(method="median", *args, **kwargs)

    def forward(
        self,
        weights: Tensor,
        ray_samples: RaySamples,
        nears: Optional[Tensor] = None,
        fars: Optional[Tensor] = None,
        ray_indices: Optional[Tensor] = None,
        num_rays: Optional[int] = None,
    ) -> Tensor:
        depth = super().forward(
            weights=weights,
            ray_samples=ray_samples,
            normalize=True,
            nears=nears,
            fars=fars,
            ray_indices=ray_indices,
            num_rays=num_rays,
        ).detach() # [B, H, W, 1]

        # PyTorch expects the filter passed to conv2d to have a shape of [out_channels, in_channels, height, width]
        depth = depth.permute(0, 3, 1, 2) # [B, 1, H, W]

        # Define Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)

        # Compute the derivatives
        derivative_x = torch.nn.functional.conv2d(depth, sobel_x, padding=1) # [B, 1, H, W]
        derivative_y = torch.nn.functional.conv2d(depth, sobel_y, padding=1) # [B, 1, H, W]

        # value ranges from [-4, 4] to [-0.5, 0.5]
        derivative_x = derivative_x / 8.0
        derivative_y = derivative_y / 8.0

        delta = torch.sqrt(derivative_x**2 + derivative_y**2) # [B, 1, H, W]
        delta = delta.permute(0, 2, 3, 1) # [B, H, W, 1]
        assert torch.all(delta >= 0.0) and torch.all(delta <= 1.0), f"Delta depth should be in [0, 1], but is in [{delta.min()}, {delta.max()}]"
        return delta


class GeometryRenderer(NeRFRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        fn: Optional[Callable[[Tensor, Tensor], Tensor]],
        rays_o: Tensor,
        rays_d: Tensor,
    ) -> Tensor:

        def sphere_hit(rays_o: Tensor, rays_d: Tensor) -> Tensor:
            # rays_o: [B, H, W, 3]
            # rays_d: [B, H, W, 3]
            # Assuming the radius of the sphere is 1
            radius = 1.0
            # Compute the coefficients of the quadratic equation
            a = torch.sum(rays_d * rays_d, dim=-1) # a: [B, H, W]
            b = 2.0 * torch.sum(rays_o * rays_d, dim=-1) # b: [B, H, W]
            c = torch.sum(rays_o * rays_o, dim=-1) - radius**2 # c: [B, H, W]
            # Compute the discriminant
            discriminant = b**2 - 4 * a * c # discriminant: [B, H, W]
            # Check if the discriminant is greater than zero, which indicates that the ray intersects the sphere
            hit = discriminant > 0 # hit: [B, H, W]
            return hit

        device = rays_o.device
        dtype = rays_o.dtype
        B = rays_o.shape[0]
        H = rays_o.shape[1]
        W = rays_o.shape[2]
        if fn is None:
            fn = sphere_hit

        hit = fn(rays_o, rays_d) # [B, H, W]
        out = hit

        out = torch.zeros((B * H * W, 1), dtype=dtype, device=device) # black
        out[hit.reshape(-1), :] = torch.tensor([1.0], dtype=dtype, device=device) # white
        out = out.reshape(B, H, W, 1) # [B, H, W, 1]
        return out


class DebugImageRenderer(NeRFRenderer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        B: int,
        H: int,
        W: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        # Create a meshgrid
        x = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
        y = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
        # Create the image
        img = torch.zeros(H, W, 3, device=device, dtype=dtype)
        img[:, :, 0] = x # red channel
        img[:, :, 1] = y # green channel
        img = img.unsqueeze(0).repeat(B, 1, 1, 1)
        return img
