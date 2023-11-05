from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, List, Literal, Iterator
if TYPE_CHECKING:
    from torch import Tensor
    from .nerf_fields import NeRFField
    from .rays import RayBundle

import torch
import math
import numpy as np
import itertools

from ..utils.const import *
from ..utils.utils import is_normalized


class RaySamples:

    def __init__(
        self,
        center: Tensor,
        directions: Tensor,
        euclidean_starts: Optional[Tensor],
        euclidean_ends: Optional[Tensor],
        spacing_starts: Optional[Tensor],
        spacing_ends: Optional[Tensor],
        origins: Tensor,
        deltas: Tensor,
        spacing_to_euclidean_fn: Optional[Callable[..., Tensor]],
        collated: bool,
    ) -> None:
        self.center = center # [B, H, W, num_samples, 3] or [?, num_samples, 3]
        assert is_normalized(directions, dim=-1), f"norm of directions: {torch.norm(directions, dim=-1)}"
        self.directions = directions # [B, H, W, 3] or [?, 3]
        self.euclidean_starts = euclidean_starts # [B, H, W, num_samples] or [?, num_samples], used for DepthRendering
        self.euclidean_ends = euclidean_ends # [B, H, W, num_samples] or [?, num_samples], used for DepthRendering
        self.spacing_starts = spacing_starts # [B, H, W, num_samples] or [?, num_samples], used for PDFSampling
        self.spacing_ends = spacing_ends # [B, H, W, num_samples] or [?, num_samples], used for PDFSampling
        self.origins = origins # [B, H, W, 3] or [?, 3]
        self.deltas = deltas # [B, H, W, num_samples] or [?, num_samples]
        assert torch.all(self.deltas >= 0), f"min(self.deltas): {self.deltas.min()}"
        self.spacing_to_euclidean_fn = spacing_to_euclidean_fn

        self.collated = collated
        self.B: Optional[int] = None
        self.H: Optional[int] = None
        self.W: Optional[int] = None
        self.num_samples: int
        if not collated:
            self.B = self.center.shape[0]
            self.H = self.center.shape[1]
            self.W = self.center.shape[2]
            self.num_samples = self.center.shape[3]
        else:
            self.num_samples = self.center.shape[-2]

    def collate(self):
        assert not self.collated
        self.center = self.center.reshape(-1, self.num_samples, 3)
        self.directions = self.directions.reshape(-1, 3)
        if self.euclidean_starts is not None:
            self.euclidean_starts = self.euclidean_starts.reshape(-1, self.num_samples)
        if self.euclidean_ends is not None:
            self.euclidean_ends = self.euclidean_ends.reshape(-1, self.num_samples)
        if self.spacing_starts is not None:
            self.spacing_starts = self.spacing_starts.reshape(-1, self.num_samples)
        if self.spacing_ends is not None:
            self.spacing_ends = self.spacing_ends.reshape(-1, self.num_samples)
        self.origins = self.origins.reshape(-1, 3)
        self.deltas = self.deltas.reshape(-1, self.num_samples)

        self.collated = True
        return self

    def uncollate(self):
        assert self.collated
        assert self.B is not None and self.H is not None and self.W is not None

        self.center = self.center.reshape(self.B, self.H, self.W, self.num_samples, 3)
        self.directions = self.directions.reshape(self.B, self.H, self.W, 3)
        if self.euclidean_starts is not None:
            self.euclidean_starts = self.euclidean_starts.reshape(self.B, self.H, self.W, self.num_samples)
        if self.euclidean_ends is not None:
            self.euclidean_ends = self.euclidean_ends.reshape(self.B, self.H, self.W, self.num_samples)
        if self.spacing_starts is not None:
            self.spacing_starts = self.spacing_starts.reshape(self.B, self.H, self.W, self.num_samples)
        if self.spacing_ends is not None:
            self.spacing_ends = self.spacing_ends.reshape(self.B, self.H, self.W, self.num_samples)
        self.origins = self.origins.reshape(self.B, self.H, self.W, 3)
        self.deltas = self.deltas.reshape(self.B, self.H, self.W, self.num_samples)
        self.collated = False
        return self

    def get_weights_cumsum(self, densities: Tensor) -> Tensor:
        assert not self.collated
        # densities [B, ..., num_samples] must not contain inf
        delta_density = self.deltas.reshape_as(densities) * densities # [B, ..., num_samples]
        alphas = 1 - torch.exp(-delta_density) # [B, ..., num_samples]

        transmittance = torch.cumsum(delta_density[..., :-1], dim=-1) # [B, ..., num_samples-1]
        zeros = torch.zeros((*transmittance.shape[:-1], 1), dtype=densities.dtype, device=densities.device) # [B, ..., 1]
        transmittance = torch.cat([zeros, transmittance], dim=-1) # [B, ..., num_samples]
        transmittance = torch.exp(-transmittance) # [B, ..., num_samples]
        weights = alphas * transmittance # [B, ..., num_samples]
        return weights

    def get_weights_cumprod(self, densities: Tensor) -> Tensor:
        assert not self.collated
        # densities: [B, ..., num_samples] must not contain inf
        delta_density = self.deltas.reshape_as(densities) * densities
        alphas = 1 - torch.exp(-delta_density) # [B, ..., num_samples]

        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + EPS_1E15], dim=-1) # [B, ..., num_samples+1] # QUESTION: why shift
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [B, ..., num_samples]
        return weights

    def chunking(self, chunk_size: int) -> List["RaySamples"]:
        assert self.collated
        BHW = self.center.shape[0]
        if BHW <= chunk_size:
            return [self]

        num_chunks = math.ceil(BHW / chunk_size)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, BHW)

            center = self.center[start:end]
            directions = self.directions[start:end]
            euclidean_starts: Optional[Tensor] = None
            if self.euclidean_starts is not None:
                euclidean_starts = self.euclidean_starts[start:end]
            euclidean_ends: Optional[Tensor] = None
            if self.euclidean_ends is not None:
                euclidean_ends = self.euclidean_ends[start:end]
            spacing_starts: Optional[Tensor] = None
            if self.spacing_starts is not None:
                spacing_starts = self.spacing_starts[start:end]
            spacing_ends: Optional[Tensor] = None
            if self.spacing_ends is not None:
                spacing_ends = self.spacing_ends[start:end]
            origins = self.origins[start:end]
            deltas = self.deltas[start:end]

            chunks.append(RaySamples(
                center=center,
                directions=directions,
                euclidean_starts=euclidean_starts,
                euclidean_ends=euclidean_ends,
                spacing_starts=spacing_starts,
                spacing_ends=spacing_ends,
                origins=origins,
                deltas=deltas,
                collated=True,
                spacing_to_euclidean_fn=self.spacing_to_euclidean_fn,
            ))
        return chunks

    @staticmethod
    def unchunking(ray_sample_sequence: Sequence["RaySamples"]) -> "RaySamples":
        assert all(ray_sample.collated for ray_sample in ray_sample_sequence)
        center = torch.cat([ray_sample.center for ray_sample in ray_sample_sequence], dim=0)
        directions = torch.cat([ray_sample.directions for ray_sample in ray_sample_sequence], dim=0)
        euclidean_starts: Optional[Tensor] = None
        if ray_sample_sequence[0].euclidean_starts is not None:
            assert all(ray_sample.euclidean_starts is not None for ray_sample in ray_sample_sequence)
            euclidean_starts = torch.cat([ray_sample.euclidean_starts for ray_sample in ray_sample_sequence], dim=0) # type: ignore
        else:
            assert all(ray_sample.euclidean_starts is None for ray_sample in ray_sample_sequence)
        euclidean_ends: Optional[Tensor] = None
        if ray_sample_sequence[0].euclidean_ends is not None:
            assert all(ray_sample.euclidean_ends is not None for ray_sample in ray_sample_sequence)
            euclidean_ends = torch.cat([ray_sample.euclidean_ends for ray_sample in ray_sample_sequence], dim=0) # type: ignore
        else:
            assert all(ray_sample.euclidean_ends is None for ray_sample in ray_sample_sequence)
        spacing_starts: Optional[Tensor] = None
        if ray_sample_sequence[0].spacing_starts is not None:
            assert all(ray_sample.spacing_starts is not None for ray_sample in ray_sample_sequence)
            spacing_starts = torch.cat([ray_sample.spacing_starts for ray_sample in ray_sample_sequence], dim=0) # type: ignore
        else:
            assert all(ray_sample.spacing_starts is None for ray_sample in ray_sample_sequence)
        spacing_ends: Optional[Tensor] = None
        if ray_sample_sequence[0].spacing_ends is not None:
            assert all(ray_sample.spacing_ends is not None for ray_sample in ray_sample_sequence)
            spacing_ends = torch.cat([ray_sample.spacing_ends for ray_sample in ray_sample_sequence], dim=0) # type: ignore
        else:
            assert all(ray_sample.spacing_ends is None for ray_sample in ray_sample_sequence)
        origins = torch.cat([ray_sample.origins for ray_sample in ray_sample_sequence], dim=0)
        deltas = torch.cat([ray_sample.deltas for ray_sample in ray_sample_sequence], dim=0)
        return RaySamples(
            center=center,
            directions=directions,
            euclidean_starts=euclidean_starts,
            euclidean_ends=euclidean_ends,
            spacing_starts=spacing_starts,
            spacing_ends=spacing_ends,
            origins=origins,
            deltas=deltas,
            collated=True,
            spacing_to_euclidean_fn=ray_sample_sequence[0].spacing_to_euclidean_fn,
        )


class Sampler(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def to_device(self, device: torch.device):
        # Sampler will read device from ray_bundle
        return self

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([])

    def get_ray_samples(
        self,
        ray_bundle: RayBundle,
        num_samples: int,
    ) -> RaySamples:
        raise NotImplementedError


class DreamfusionSampler(Sampler):

    def __init__(self) -> None:
        super().__init__()

    def get_ray_samples(self, ray_bundle: RayBundle, num_samples: int) -> RaySamples:
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None
        assert ray_bundle.directions is not None

        device = ray_bundle.origins.device
        z_vals = torch.linspace(0.0, 1.0, num_samples, device=device) # [T]
        z_vals = z_vals[None, None, None, :, None] # [1, 1, 1, T, 1]
        nears = ray_bundle.nears.unsqueeze(-2) # [B, H, W, 1, 1]
        fars = ray_bundle.fars.unsqueeze(-2) # [B, H, W, 1, 1]
        z_vals = torch.lerp(nears, end=fars, weight=z_vals) # [B, H, W, T, 1]
        center = ray_bundle.origins.unsqueeze(-2) + ray_bundle.directions.unsqueeze(-2) * z_vals # [B, H, W, T, 3]

        # calculate deltas
        deltas = z_vals[..., 1:, :] - z_vals[..., :-1, :] # [B, H, W, T-1, 1]
        avg_dist = (fars - nears) / num_samples # [B, H, W, 1, 1]
        deltas = torch.cat([deltas, avg_dist * torch.ones_like(deltas[..., :1, :])], dim=-2) # [B, H, W, T, 1]
        return RaySamples(
            center=center, # [B, H, W, T, 3]
            directions=ray_bundle.directions, # [B, H, W, 3]
            euclidean_starts=None,
            euclidean_ends=None,
            spacing_starts=None,
            spacing_ends=None,
            origins=ray_bundle.origins, # [B, H, W, 3]
            deltas=deltas.squeeze(-1), # [B, H, W, T]
            collated=False,
            spacing_to_euclidean_fn=None,
        )


class SpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__()
        self.stratified = stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def get_ray_samples(
        self,
        ray_bundle: RayBundle,
        num_samples: int,
    ) -> RaySamples:
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None
        assert ray_bundle.directions is not None
        device = ray_bundle.origins.device
        dtype = ray_bundle.origins.dtype

        B = ray_bundle.origins.shape[0]
        H = ray_bundle.origins.shape[1]
        W = ray_bundle.origins.shape[2]
        assert ray_bundle.origins.shape[3] == 3
        num_rays = B * H * W

        bins = torch.linspace(0.0, 1.0, num_samples + 1, device=device, dtype=dtype)[None, None, None, ...] # [1, 1, 1, num_samples+1]

        if self.stratified:
            if self.single_jitter:
                t_rand = torch.rand((B, H, W, 1), dtype=dtype, device=device)
            else:
                t_rand = torch.rand((B, H, W, num_samples + 1), dtype=dtype, device=device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand # [B, H, W, num_samples+1]

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars))
        spacing_to_euclidean_fn: Callable[..., Tensor] = lambda x: self.spacing_fn_inv(torch.lerp(s_near, end=s_far, weight=x)) # don't use bins * s_far + (1 - bins) * s_near due to floating point error
        euclidean_bins: Tensor = spacing_to_euclidean_fn(bins) # [B, H, W, num_samples+1]

        euclidean_bins_starts: Tensor = euclidean_bins[..., :-1] # [B, H, W, num_samples]
        euclidean_bins_ends: Tensor = euclidean_bins[..., 1:] # [B, H, W, num_samples]
        center: Tensor = ray_bundle.origins.unsqueeze(-2) + ray_bundle.directions.unsqueeze(-2) * (euclidean_bins_starts.unsqueeze(-1) + euclidean_bins_ends.unsqueeze(-1)) / 2.0 # [B, H, W, num_samples, 3]

        ray_samples = RaySamples(
            center=center, # [B, H, W, num_samples, 3]
            directions=ray_bundle.directions, # [B, H, W, 3]
            euclidean_starts=euclidean_bins_starts, # [B, H, W, num_samples]
            euclidean_ends=euclidean_bins_ends, # [B, H, W, num_samples]
            spacing_starts=bins[..., :-1], # [B, H, W, num_samples]
            spacing_ends=bins[..., 1:], # [B, H, W, num_samples]
            origins=ray_bundle.origins, # [B, H, W, 3]
            deltas=euclidean_bins_ends - euclidean_bins_starts, # [B, H, W, num_samples]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            collated=False,
        )

        return ray_samples


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        stratified: Use stratified sampling during training
        single_jitter: Use a same random jitter for all samples along a ray
    """

    def __init__(
        self,
        stratified: bool,
        single_jitter: bool,
    ) -> None:
        super().__init__(
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            stratified=stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(Sampler):

    def __init__(
        self,
        stratified: bool,
        single_jitter: bool,
        include_original: bool,
        histogram_padding: float,
    ) -> None:
        super().__init__()
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.stratified = stratified
        self.single_jitter = single_jitter

    def get_ray_samples(
        self,
        ray_bundle: RayBundle,
        ray_samples: RaySamples,
        weights: Tensor, # [..., num_samples]
        num_samples: int,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle.directions is not None
        assert ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        assert ray_samples.spacing_to_euclidean_fn is not None
        device = ray_bundle.origins.device
        dtype = ray_bundle.origins.dtype

        num_bins = num_samples + 1

        weights = weights.reshape(*ray_samples.spacing_starts.shape) # [B, H, W, num_samples]
        weights = weights + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(EPS_1E6 - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.stratified:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, dtype=dtype, device=device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), dtype=dtype, device=device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), dtype=dtype, device=device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, dtype=dtype, device=device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts,
                ray_samples.spacing_ends[..., -1:],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = torch.lerp(bins_g0, bins_g1, t)
        bins, _ = torch.cummax(bins, dim=-1) # TODO: replace this hack to force increasing bin, since lerp before is complicated and I don't want to touch that

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()
        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)
        euclidean_bins_starts = euclidean_bins[..., :-1]
        euclidean_bins_ends = euclidean_bins[..., 1:]

        center: Tensor = ray_bundle.origins.unsqueeze(-2) + ray_bundle.directions.unsqueeze(-2) * (euclidean_bins_starts.unsqueeze(-1) + euclidean_bins_ends.unsqueeze(-1)) / 2.0 # [B, H, W, num_samples, 3]

        ray_samples = RaySamples(
            center=center, # [B, H, W, num_samples, 3]
            directions=ray_bundle.directions, # [B, H, W, 3]
            euclidean_starts=euclidean_bins_starts, # [B, H, W, num_samples]
            euclidean_ends=euclidean_bins_ends, # [B, H, W, num_samples]
            spacing_starts=bins[..., :-1], # [B, H, W, num_samples]
            spacing_ends=bins[..., 1:], # [B, H, W, num_samples]
            origins=ray_bundle.origins, # [B, H, W, 3]
            deltas=euclidean_bins_ends - euclidean_bins_starts, # [B, H, W, num_samples]
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
            collated=False,
        )

        return ray_samples


class ProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples."""

    def __init__(
        self,
        stratified: bool,
        accumulator: Literal["cumsum", "cumprod"],
        num_proposal_samples_per_ray: Tuple,
        proposal_networks: List[NeRFField],
        proposal_weights_anneal_max_num_iters: int,
        proposal_weights_anneal_slope: float,
        single_jitter: bool,
        update_sched: Callable[[int], int],
    ) -> None:
        super().__init__()

        # configs
        self.accumulator: Literal['cumsum', 'cumprod'] = accumulator
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.update_sched = update_sched
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = proposal_networks

        assert len(self.num_proposal_samples_per_ray) > 0

        # samplers
        self.initial_sampler = UniformSampler(stratified=stratified, single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(stratified=stratified, single_jitter=single_jitter, include_original=False, histogram_padding=0.01)

        # internal tracking
        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def to_device(self, device: torch.device):
        self.proposal_networks = [proposal_network.to(device) for proposal_network in self.proposal_networks]
        return self

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return itertools.chain(*[proposal_network.parameters() for proposal_network in self.proposal_networks])

    def callback_before(self) -> None:
        # https://arxiv.org/pdf/2111.12077.pdf eq. 18
        train_frac = np.clip(self._step / self.proposal_weights_anneal_max_num_iters, 0, 1)
        bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
        anneal = bias(train_frac, self.proposal_weights_anneal_slope)
        self._anneal = anneal

    def callback_after(self) -> None:
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step += 1
        self._steps_since_update += 1

    def get_ray_samples(
        self,
        ray_bundle: RayBundle,
        num_samples: int,
    ) -> RaySamples:
        self.callback_before() # WARNING: assuming 1 sampling per step

        weights: Optional[Tensor] = None
        ray_samples: Optional[RaySamples] = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10

        for i_level in range(len(self.num_proposal_samples_per_ray) + 1):
            is_proposal_net_level = i_level != len(self.num_proposal_samples_per_ray)
            current_samples = self.num_proposal_samples_per_ray[i_level] if is_proposal_net_level else num_samples

            # get samples
            if i_level == 0:
                ray_samples = self.initial_sampler.get_ray_samples(
                    ray_bundle=ray_bundle,
                    num_samples=current_samples,
                )
            else:
                assert weights is not None
                assert ray_samples is not None
                ray_samples = self.pdf_sampler.get_ray_samples(
                    ray_bundle=ray_bundle,
                    ray_samples=ray_samples,
                    weights=torch.pow(weights, self._anneal), # Annealing weights. This will be a no-op if self._anneal is 1.0.
                    num_samples=current_samples,
                )
            assert ray_samples is not None

            # get weights by evaluating density
            if is_proposal_net_level:
                B = ray_samples.center.shape[0]
                H = ray_samples.center.shape[1]
                W = ray_samples.center.shape[2]
                T = ray_samples.center.shape[3]
                C = ray_samples.center.shape[4]
                center = ray_samples.center.reshape(B * H * W, T, C) # [B*H*W, num_samples, 3]

                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    densities, _ = self.proposal_networks[i_level].get_density(center) # [B*H*W, num_samples, 1]
                else:
                    with torch.no_grad():
                        densities, _ = self.proposal_networks[i_level].get_density(center) # [B*H*W, num_samples, 1]
                densities = densities.reshape(B, H * W, T) # [B, H, W, num_samples, 1] -> [B, H*W, num_samples]

                if self.accumulator == "cumsum":
                    weights = ray_samples.get_weights_cumsum(densities) # [B, H*W, num_samples]
                elif self.accumulator == "cumprod":
                    weights = ray_samples.get_weights_cumprod(densities) # [B, H*W, num_samples]
                else:
                    raise ValueError(f"Unknown accumulator {self.accumulator}")
        if updated:
            self._steps_since_update = 0

        self.callback_after() # WARNING: assuming 1 sampling per step
        assert ray_samples is not None
        return ray_samples
