from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, Union, Tuple
import warnings
if TYPE_CHECKING:
    from torch import Tensor
    from .nerf_fields import FARTNeRFContraction
    from .ray_samplers import RaySamples

import numpy as np
import torch
try:
    import tinycudann as tcnn
except EnvironmentError as e:
    warnings.warn(f"WARNING: {e}\nThis error is fine for CPU-only mode.", ImportWarning)

from .nerf_fields import NeRFField


class HashMLPDensityField(NeRFField):

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        contraction: FARTNeRFContraction,
        density_activation: Union[torch.nn.Module, Callable[..., Any]],
        use_linear: bool,
        num_levels: int,
        max_res: int,
        base_res: int,
        log2_hashmap_size: int,
        features_per_level: int,
        background_density_init: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.use_linear = use_linear
        self.density_activation = density_activation

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": np.exp2((np.log2(max_res / base_res)) / (num_levels - 1)),
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        try:
            if self.use_linear:
                self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config["encoding"])
                self.linear = torch.nn.Linear(self.encoding.n_output_dims, 1)
            else:
                self.mlp_base = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3,
                    n_output_dims=1,
                    encoding_config=config["encoding"],
                    network_config=config["network"],
                )
        except NameError as e:
            if self.use_linear:
                self.encoding = torch.nn.Linear(1, 1) # dummy
                self.linear = torch.nn.Linear(1, 1) # dummy
            else:
                self.mlp_base = torch.nn.Linear(1, 1) # dummy
            warnings.warn(f"WARNING: {e}\nThis error is fine for CPU-only mode.", ImportWarning)

        # outside of [-1, 1]^3
        background_density = torch.tensor([background_density_init], dtype=dtype, device=device, requires_grad=False)
        self.background = torch.nn.Parameter(background_density, requires_grad=False) # [SH + 1]

    def to_device(self, device: torch.device):
        if self.use_linear:
            self.encoding = self.encoding.to(device)
            self.linear = self.linear.to(device)
        else:
            self.mlp_base = self.mlp_base.to(device)

        self.background = self.background.to(device)

    def get_density(self, positions: Tensor):
        original_positions = positions

        n_ray: int = positions.shape[-3]
        n_sample: int = positions.shape[-2]
        assert positions.shape[-1] == 3
        positions = positions.reshape(-1, 3) # [N_ray * N_sample, 3]
        # TODO: contraction

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

            if self.use_linear:
                features_densities_foreground = self.linear(self.encoding(foreground_positions)).view(-1, 1)
            else:
                features_densities_foreground = self.mlp_base(foreground_positions).view(-1, 1)

            # WARNING: for some reason, nvidia-tcnn will always give [torch.float16] for features_densities_foreground
            features_densities_foreground = features_densities_foreground.to(dtype=foreground_positions.dtype) # [N_ray * N_sample, 1]

            features_densities = self.background.expand(n_ray * n_sample, -1).clone() # [N_ray * N_sample, 1]
            features_densities[~background_positions_selector, :] = features_densities_foreground

        else:
            # foreground is empty -> everything is background
            features_densities = self.background.expand(n_ray * n_sample, -1) # [N_ray * N_sample, 1]

        densities = features_densities
        densities = densities.reshape(-1, n_sample, densities.shape[-1]) # [N_ray, N_sample, 1]
        return self.density_activation(densities), None # [N_ray, N_sample, 1], [N_ray, N_sample, SH]

    def get_color(self, directions: Tensor, color_feature: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
