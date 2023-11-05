import torch

from torch import Tensor
from typing import Sequence, Tuple, Iterator, List

from .field_base import FieldBase
from .rays import RayBundle
from .renderers import ImageRenderer, IdentityRenderer


class Embed(FieldBase):

    def __init__(
        self,
        latent_dreamfusion: bool,
        embedding_h: int,
        embedding_w: int,
        degree_latent: int,
        color_activation: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            dtype=dtype,
            device=device,
        )
        self.embedding = torch.nn.Parameter(torch.randn(self.degree_latent, embedding_h, embedding_w, dtype=dtype, device=device))
        self.color_activation: torch.nn.Module = color_activation

    def to_device(self, device: torch.device):
        self.embedding = self.embedding.to(device)
        self.embedding.requires_grad_(True)
        return self

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return super().parameters(recurse)

    def _forward(self, ray_bundle: RayBundle, renderers: Sequence[ImageRenderer]) -> Tuple[List[Tensor], List[Tensor]]:
        # translate ray_bungle to coordinate
        assert ray_bundle.origins.shape[-1] == 2, f"ray_bundle.origins.shape[-1]={ray_bundle.origins.shape[-1]}"
        coords = ray_bundle.origins

        # [-1, 1] -> [0, W-1] or [0, H-1]
        coords = (coords + 1) / 2
        coords[..., 0] *= (self.embedding.shape[2] - 1)
        coords[..., 1] *= (self.embedding.shape[1] - 1)
        shape = coords.shape # [B, H, W, C]
        coords = coords.reshape(-1, shape[-1]).round().long() # [B*H*W, C]

        xs = coords[:, 0]
        ys = coords[:, 1]

        output = self.embedding[:, ys, xs] # [B*H*W, C]
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
