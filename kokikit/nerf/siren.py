import torch
import numpy as np

from torch import Tensor
from typing import List, Sequence, Tuple, Iterator

from .field_base import FieldBase
from .rays import RayBundle
from .renderers import ImageRenderer, IdentityRenderer


### siren from https://github.com/vsitzmann/siren/
class SineLayer(torch.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: torch.device,
        bias: bool = True,
        is_first: bool = False,
        omega_0: int = 30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(FieldBase):

    def __init__(
        self,
        latent_dreamfusion: bool,
        degree_latent: int,
        hidden_features: int,
        hidden_layers: int,
        color_activation: torch.nn.Module,
        outermost_linear: bool,
        first_omega_0: int,
        hidden_omega_0: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            dtype=dtype,
            device=device,
        )
        in_features = 2 # 2 dimensional image

        net: List[torch.nn.Module] = []
        net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, dtype=dtype, device=device))
        for i in range(hidden_layers):
            net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, dtype=dtype, device=device))
        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, self.degree_latent, dtype=dtype)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            net.append(final_linear)
        else:
            net.append(SineLayer(hidden_features, self.degree_latent, is_first=False, omega_0=hidden_omega_0, dtype=dtype, device=device))
        self.net: torch.nn.Sequential = torch.nn.Sequential(*net)
        self.color_activation: torch.nn.Module = color_activation

    def to_device(self, device: torch.device):
        self.net = self.net.to(device)
        self.net.requires_grad_(True)
        return self

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.net.parameters()

    def _forward(self, ray_bundle: RayBundle, renderers: Sequence[ImageRenderer]) -> Tuple[List[Tensor], List[Tensor]]:
        # translate ray_bungle to coordinate
        assert ray_bundle.origins.shape[-1] == 2, f"ray_bundle.origins.shape[-1]={ray_bundle.origins.shape[-1]}"
        coords = ray_bundle.origins

        shape = coords.shape # [B, H, W, C]
        coords = coords.reshape(-1, shape[-1]) # [B*H*W, C]

        output = self.net(coords) # [B*H*W, C]
        output = self.color_activation(output) # [B*H*W, C]

        images: List[Tensor] = [] # List[B, H, W, C]
        losses: List[Tensor] = [] # List[?]
        for renderer in renderers:
            if type(renderer) == IdentityRenderer:
                image, loss = renderer.forward_loss(color=output, B=shape[0], H=shape[1], W=shape[2], C=self.degree_latent)
            else:
                raise ValueError(f"Unknown renderer type: {type(renderer)}")
            images.append(image)
            losses.append(loss)
        return images, losses
