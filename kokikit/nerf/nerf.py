import warnings
import torch
import itertools
import nerfacc

from torch import Tensor
from typing import Optional, Literal, Sequence, Tuple, Iterator, List
from torch.cuda.amp.autocast_mode import autocast

from ..utils.utils import safe_normalize
from .field_base import FieldBase
from .rays import RayBundle
from .ray_samplers import Sampler, NeRFAccSampler, RaySamples, NeRFAccSamples, UniformSampler, PDFSampler, ProposalNetworkSampler
from .renderers import NeRFRenderer, RGBRenderer, RGBMaxRenderer, DepthRenderer, DeltaDepthRenderer, AccumulationRenderer, NormalRenderer, TexturelessRenderer, LambertianRenderer, NormalAlignmentRenderer
from .nerf_fields import NeRFField, NeRFAccField, SHMLPBackground


class TemporaryGrad():

    def __init__(self, variable: Tensor):
        self.variable = variable
        self.original_requires_grad = variable.requires_grad

    def __enter__(self):
        self.variable.requires_grad = True

    def __exit__(self, type, value, traceback):
        self.variable.requires_grad = self.original_requires_grad


class AccNeRF(FieldBase):

    def __init__(
        self,
        latent_dreamfusion: bool,
        degree_latent: int,
        sampler: NeRFAccSampler,
        nerf_field: NeRFAccField,
        background_field: SHMLPBackground,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            dtype=dtype,
            device=device,
        )
        self.sampler: NeRFAccSampler = sampler
        self.nerf_field: NeRFAccField = nerf_field
        self.background_field: SHMLPBackground = background_field

    def to_device(self, device: torch.device):
        self.sampler.to_device(device)
        self.nerf_field.to_device(device)
        self.background_field.to_device(device)
        return self

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return itertools.chain(self.nerf_field.parameters(), self.sampler.parameters(), self.background_field.parameters())

    def _forward(self, ray_bundle: RayBundle, renderers: Sequence[NeRFRenderer]) -> Tuple[List[Tensor], List[Tensor]]:
        # translate ray_bundle to ray_samples
        assert ray_bundle.origins.shape[-1] == 3, f"ray_bundle.origins.shape[-1]={ray_bundle.origins.shape[-1]}"
        assert ray_bundle.directions is not None
        B, H, W = ray_bundle.origins.shape[:3]

        ray_samples: NeRFAccSamples
        n_rays: int
        if isinstance(self.sampler, NeRFAccSampler):
            ray_samples = self.sampler.get_ray_samples(ray_bundle=ray_bundle, density_fn=self.nerf_field.get_density)
            n_rays = ray_bundle.origins.shape[0] * ray_bundle.origins.shape[1] * ray_bundle.origins.shape[2]
        else:
            raise ValueError(f"Unknown sampler type: {type(self.sampler)}")
        
        rgb_fg_all, density = self.nerf_field.forward(ray_samples=ray_samples)
        comp_rgb_bg = self.background_field(dirs=ray_bundle.directions) # [B, H, W, 3]
        
        weights_, _, _ = nerfacc.render_weight_from_density(
            ray_samples.t_starts[..., 0],
            ray_samples.t_ends[..., 0],
            density[..., 0],
            ray_indices=ray_samples.ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
        opacity = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_samples.ray_indices, n_rays=n_rays
        )
        depth = nerfacc.accumulate_along_rays(
            weights[..., 0], values=((ray_samples.t_starts + ray_samples.t_ends) / 2.0), ray_indices=ray_samples.ray_indices, n_rays=n_rays
        )
        comp_rgb_fg = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_samples.ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        t_depth = depth[ray_samples.ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(((ray_samples.t_starts + ray_samples.t_ends) / 2.0) - t_depth) ** 2,
            ray_indices=ray_samples.ray_indices,
            n_rays=n_rays,
        )

        bg_color = comp_rgb_bg
        if bg_color.shape[:-1] == (B, H, W):
            bg_color = bg_color.reshape(B * H * W, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)
        
        # out = {
        #     "comp_rgb": comp_rgb.view(B, H, W, -1),
        #     "comp_rgb_fg": comp_rgb_fg.view(B, H, W, -1),
        #     "comp_rgb_bg": comp_rgb_bg.view(B, H, W, -1),
        #     "opacity": opacity.view(B, H, W, 1),
        #     "depth": depth.view(B, H, W, 1),
        #     "z_variance": z_variance.view(B, H, W, 1),
        #     # "weights": weights,
        #     # "t_points": ((ray_samples.t_starts + ray_samples.t_ends) / 2.0),
        #     # "t_dirs": ray_samples.t_dirs,
        #     # "ray_indices": ray_samples.ray_indices,
        #     # "points": ray_samples.positions,
        # }
        
        # calculate [images]
        images: List[Tensor] = [] # List[B, H, W, ?]
        losses: List[Tensor] = [] # List[?]
        for i, renderer in enumerate(renderers):
            if i == 1:
                images.append(opacity.clamp(0, 1).view(B, H, W, 1))
                losses.append(torch.zeros(1, device='cuda'))
            else:
                images.append(comp_rgb.view(B, H, W, -1))
                losses.append(torch.zeros(1, device='cuda'))
        return images, losses
    

class NeRF(FieldBase):

    def __init__(
        self,
        latent_dreamfusion: bool,
        degree_latent: int,
        num_samples: int,
        sampler: Sampler,
        nerf_field: NeRFField,
        light_delta: Optional[float],
        accumulator: Literal["cumsum", "cumprod"],
        ray_samples_chunk_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            dtype=dtype,
            device=device,
        )
        self.num_samples: int = num_samples
        self.sampler: Sampler = sampler
        self.accumulator: Literal['cumsum', 'cumprod'] = accumulator
        self.ray_samples_chunk_size: int = ray_samples_chunk_size
        self.nerf_field: NeRFField = nerf_field

        # Auxiliary Variables
        self.light_delta: Optional[float] = light_delta

    def to_device(self, device: torch.device):
        self.sampler.to_device(device)
        self.nerf_field.to_device(device)
        return self

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return itertools.chain(self.nerf_field.parameters(), self.sampler.parameters())

    def _forward(self, ray_bundle: RayBundle, renderers: Sequence[NeRFRenderer]) -> Tuple[List[Tensor], List[Tensor]]:
        # translate ray_bundle to ray_samples
        assert ray_bundle.origins.shape[-1] == 3, f"ray_bundle.origins.shape[-1]={ray_bundle.origins.shape[-1]}"

        ray_samples: RaySamples
        if type(self.sampler) == UniformSampler:
            ray_samples = self.sampler.get_ray_samples(ray_bundle=ray_bundle, num_samples=self.num_samples)
        elif type(self.sampler) == ProposalNetworkSampler:
            ray_samples = self.sampler.get_ray_samples(ray_bundle=ray_bundle, num_samples=self.num_samples)
        else:
            raise ValueError(f"Unknown sampler type: {type(self.sampler)}")

        return self._forward_ray_samples(ray_samples=ray_samples, renderers=renderers)

    def _forward_colors_densities_normals(self, ray_samples: RaySamples, requires_normals: bool) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        # calculate [colors], [densities], [normal]
        colors: Tensor
        densities: Tensor
        normals: Optional[Tensor] = None # precalculate normal if needed
        if requires_normals:
            with torch.enable_grad(), autocast(enabled=False), TemporaryGrad(ray_samples.center):
                colors, densities = self.nerf_field.forward(ray_samples) # [B, H, W, num_samples, C], [B, H, W, num_samples, 1]
                normals = -torch.autograd.grad(torch.sum(input=densities), ray_samples.center, create_graph=True)[0] # [B, H, W, num_samples, 3]
            normals = safe_normalize(normals, dim=-1)
            if not torch.isfinite(normals).all():
                if torch.isnan(normals).any():
                    warnings.warn(f"normals contains NaN", RuntimeWarning)
                else:
                    warnings.warn(f"normals contains infinity", RuntimeWarning)
            normals = normals.nan_to_num()
        else:
            colors, densities = self.nerf_field.forward(ray_samples) # [B, H, W, num_samples, C], [B, H, W, num_samples, 1]
        return colors, densities, normals # [B, H, W, num_samples, C], [B, H, W, num_samples, 1], [B, H, W, num_samples, 3]

    def _forward_ray_samples(self, ray_samples: RaySamples, renderers: Sequence[NeRFRenderer]) -> Tuple[List[Tensor], List[Tensor]]:
        B = ray_samples.center.shape[0]
        H = ray_samples.center.shape[1]
        W = ray_samples.center.shape[2]
        # Determine whether to pre-calculate normals and lambertian
        requires_normals: bool = False
        requires_lambertian: bool = False
        for renderer in renderers:
            if type(renderer) == NormalRenderer or type(renderer) == NormalAlignmentRenderer:
                requires_normals = True
            elif type(renderer) == TexturelessRenderer or type(renderer) == LambertianRenderer:
                requires_lambertian = True
                requires_normals = True

        # calculate [colors], [densities], [normal]
        colors: Tensor
        densities: Tensor
        normals: Optional[Tensor] = None

        ray_samples = ray_samples.collate()
        ray_samples_sequence: List[RaySamples] = ray_samples.chunking(chunk_size=self.ray_samples_chunk_size)
        if len(ray_samples_sequence) > 1:
            warnings.warn(f"Chunking ray_samples_sequence into {len(ray_samples_sequence)} chunks, may not be ideal for maximum efficiency.", RuntimeWarning)
            colors_sequence, densities_sequence, normal_sequence = tuple(zip(*[self._forward_colors_densities_normals(ray_samples=ray_samples_chunk, requires_normals=requires_normals) for ray_samples_chunk in ray_samples_sequence]))

            colors = torch.cat(colors_sequence, dim=0) # [?, num_samples, C] # type: ignore
            densities = torch.cat(densities_sequence, dim=0) # [?, num_samples, 1] # type: ignore
            normals: Optional[Tensor] = None
            if requires_normals:
                normals = torch.cat(normal_sequence, dim=0) # [?, num_samples, 3] # type: ignore
        else:
            colors, densities, normals = self._forward_colors_densities_normals(ray_samples=ray_samples, requires_normals=requires_normals)

        ray_samples = ray_samples.uncollate()
        colors = colors.reshape(B, H, W, self.num_samples, -1) # [?, num_samples, C] -> [B, H, W, num_samples, C]
        densities = densities.reshape(B, H, W, self.num_samples, 1) # [?, num_samples, 1] -> [B, H, W, num_samples, 1]
        if normals is not None:
            normals = normals.reshape(B, H, W, self.num_samples, 3)

        # precalculate lambertian if needed
        lambertian: Optional[Tensor] = None
        if requires_lambertian:
            assert normals is not None
            assert self.light_delta is not None
            light = safe_normalize(ray_samples.origins + self.light_delta * torch.randn(3, device=ray_samples.origins.device)) # [B, H, W, 3]
            light = light.unsqueeze(-2) # [B, H, W, 1, 3]
            lambertian = (normals * light).sum(-1).clamp(min=0) # [B, H, W, num_samples, 3]

        # calculate [color_weights]
        densities = densities.reshape(B, H * W, self.num_samples) # [B, H, W, num_samples, 1] -> [B, H*W, num_samples]
        densities = torch.nan_to_num(densities) # densities may contain inf
        color_weights: Tensor
        if self.accumulator == "cumsum":
            color_weights = ray_samples.get_weights_cumsum(densities=densities) # [B, R, num_samples]
        elif self.accumulator == "cumprod":
            color_weights = ray_samples.get_weights_cumprod(densities=densities) # [B, R, num_samples]
        else:
            raise ValueError(f"Unknown accumulator type: {self.accumulator}")
        color_weights = color_weights.reshape(B, H, W, self.num_samples, 1) # [B, R, num_samples] -> [B, H, W, num_samples, 1]
        assert torch.isfinite(color_weights).all(), f"color_weights contains NaN or infinity, color_weights.min()={color_weights.min()}, color_weights.max()={color_weights.max()}"

        # calculate [images]
        images: List[Tensor] = [] # List[B, H, W, ?]
        losses: List[Tensor] = [] # List[?]
        for renderer in renderers:
            if type(renderer) == RGBRenderer:
                image, loss = renderer.forward_loss(colors=colors, weights=color_weights, min=0.0, max=1.0)
            elif type(renderer) == RGBMaxRenderer:
                image, loss = renderer.forward_loss(colors=colors, weights=color_weights, min=0.0, max=1.0)
            elif type(renderer) == DepthRenderer:
                image, loss = renderer.forward_loss(weights=color_weights, ray_samples=ray_samples, normalize=True, nears=None, fars=None)
            elif type(renderer) == DeltaDepthRenderer:
                image, loss = renderer.forward_loss(weights=color_weights, ray_samples=ray_samples, nears=None, fars=None)
            elif type(renderer) == AccumulationRenderer:
                image, loss = renderer.forward_loss(weights=color_weights)
            elif type(renderer) == NormalRenderer:
                assert normals is not None
                image, loss = renderer.forward_loss(normals=normals, weights=color_weights, min=0.0, max=1.0)
            elif type(renderer) == TexturelessRenderer:
                assert lambertian is not None
                image, loss = renderer.forward_loss(lambertian=lambertian, weights=color_weights, min=0.0, max=1.0)
            elif type(renderer) == LambertianRenderer:
                assert lambertian is not None
                image, loss = renderer.forward_loss(colors=colors, lambertian=lambertian, weights=color_weights, min=0.0, max=1.0)
            elif type(renderer) == NormalAlignmentRenderer:
                assert normals is not None
                image, loss = renderer.forward_loss(normals=normals, weights=color_weights, directions=ray_samples.directions)
            else:
                raise ValueError(f"Unknown renderer type: {type(renderer)}")
            assert torch.isfinite(image).all(), f"image contains NaN or infinity, image.min()={image.min()}, image.max()={image.max()}, renderer={type(renderer)}"
            assert torch.isfinite(loss).all(), f"loss contains NaN or infinity, loss.min()={loss.min()}, loss.max()={loss.max()}, renderer={type(renderer)}"
            images.append(image)
            losses.append(loss)
        return images, losses
