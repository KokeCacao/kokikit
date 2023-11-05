import torch

from torch import Tensor
from diffusers import AutoencoderKL
from typing import Union, Any, Sequence, Tuple, Iterator, List

from .rays import RayBundle
from .renderers import Renderer


class FieldBase(torch.nn.Module):

    def __init__(
        self,
        latent_dreamfusion: bool,
        degree_latent: int,
        dtype: torch.dtype,
        device: torch.device,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.latent_dreamfusion = latent_dreamfusion
        self.degree_latent = degree_latent

    def to_device(self, device: torch.device):
        raise NotImplementedError

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return super().parameters(recurse)

    def _forward(self, ray_bundle: RayBundle, renderers: Sequence[Renderer]) -> Tuple[List[Tensor], List[Tensor]]:
        # return value should be [B, H, W, C]
        # ranging from 0~1 if not latent
        # usually ranging from -1~1 if latent
        raise NotImplementedError

    def get_latent(self, ray_bundle: RayBundle, vae: AutoencoderKL, renderers: Sequence[Renderer], nerf_scale: float) -> Tuple[Tensor, List[Tensor]]:

        def fn(output: Union[Tensor, Any]) -> Tensor:
            output = output.permute(0, 3, 1, 2) # [B, C, H, W], in [-1, 1] or [-inf, inf]
            assert torch.all(output >= 0.0) and torch.all(output <= 1.0), f"output.min()={output.min()}, output.max()={output.max()}, output.shape={output.shape}"
            if self.latent_dreamfusion: # assume in range [-1, 1] or [-inf, inf]
                assert nerf_scale == 1, f"nerf_scale={nerf_scale} is not supported for latent_dreamfusion"
                out_channels = output.shape[1]
                if out_channels == 4:
                    output = output
                elif out_channels == 1:
                    output = output.expand(-1, 4, -1, -1)
                else:
                    raise ValueError(f"Invalid out_channels={out_channels}, expected 4 or 1")
                return output # [B, 4, 64, 64]
            else: # assume in range [0, 1]
                if nerf_scale != 1:
                    output = torch.nn.functional.interpolate(output, size=(int(output.shape[-2] * nerf_scale), int(output.shape[-1] * nerf_scale)), mode='bilinear', align_corners=False) # [B, C, 64*nerf_scale, 64*nerf_scale]
                out_channels = output.shape[1]
                if out_channels == 3:
                    output = output
                elif out_channels == 1:
                    output = output.expand(-1, 3, -1, -1)
                elif out_channels > 3:
                    output = output[:, :3, :, :]
                else:
                    raise ValueError(f"Invalid out_channels={out_channels}, expected 3 or 1 or >3")
                # assume all renderer need encode when not latent, since [get_latent] isn't meant to provide visualization
                output = vae.config['scaling_factor'] * vae.encode(output * 2 - 1 # type: ignore
                                                                  ).latent_dist.sample() # https://github.com/huggingface/diffusers/issues/437
                return output # [B, 4, 64, 64]

        outputs, reg_losses = self._forward(ray_bundle=ray_bundle, renderers=renderers) # Sequence[B, H, W, C], Sequence[?]
        return torch.stack([fn(output) for output in outputs], dim=0), reg_losses # [R, B, 4, 64, 64]

    def get_image(self, ray_bundle: RayBundle, vae: AutoencoderKL, renderers: Sequence[Renderer], nerf_scale: float) -> Tensor:
        # if not self.latent_dreamfusion: # QUESTION: why need this to produce correct image?
        #     return self._get_image_resample(ray_bundle=ray_bundle, vae=vae, renderers=renderers)

        def fn(output: Union[Tensor, Any], renderer: Renderer) -> Tensor:
            output = output.permute(0, 3, 1, 2) # [B, C, H, W], in [0, 1]
            if self.latent_dreamfusion: # assumes in range [-1, 1] or [-inf, inf]
                assert nerf_scale == 1, f"nerf_scale={nerf_scale} is not supported for latent_dreamfusion"
                if renderer.need_decode:
                    output = vae.decode(1 / vae.config['scaling_factor'] * output).sample # type: ignore
                    output = (output / 2 + 0.5).clamp(0.0, 1.0) # [-1, 1] -> [0, 1]
                else:
                    # only take first 3 channels, and resize to vae_in
                    out_channels = output.shape[1]
                    if out_channels == 4:
                        # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204
                        output = output.permute(0, 2, 3, 1) # [B, H, W, C]
                        output = torch.matmul(
                            output,
                            torch.tensor(
                                [
                                    #   R       G       B
                                    [0.298, 0.207, 0.208], # L1
                                    [0.187, 0.286, 0.173], # L2
                                    [-0.158, 0.189, 0.264], # L3
                                    [-0.184, -0.271, -0.473], # L4
                                ],
                                device=output.device,
                                dtype=output.dtype,
                            ),
                        )
                        output = output.permute(0, 3, 1, 2) # [B, C, H, W]
                        output = output / 2 + 0.5 # [-1, 1] -> [0, 1]
                    elif out_channels == 3:
                        output = output
                    elif out_channels == 1:
                        output = output.expand(-1, 3, -1, -1)
                    elif out_channels > 4:
                        output = output[:, :3, :, :]
                        output = output / 2 + 0.5 # [-1, 1] -> [0, 1]
                    else:
                        raise ValueError(f"Invalid out_channels={out_channels}, expected 4 or 3 or 1 or >4")
                    output = torch.nn.functional.interpolate(output, size=(512, 512), mode='nearest') # TODO: properly interpolate here for x8 vae for viewing
                assert torch.all(output >= 0.0) and torch.all(output <= 1.0), f"output.min()={output.min()}, output.max()={output.max()}, output.shape={output.shape}"
                return output # [B, 3, 512, 512]
            else: # assume in range [0, 1]
                if nerf_scale != 1:
                    output = torch.nn.functional.interpolate(output, size=(int(output.shape[-2] * nerf_scale), int(output.shape[-1] * nerf_scale)), mode='bilinear', align_corners=False)
                out_channels = output.shape[1]
                if out_channels == 3:
                    output = output
                elif out_channels == 1:
                    output = output.expand(-1, 3, -1, -1)
                else:
                    raise ValueError(f"Invalid out_channels={out_channels}, expected 3 or 1")
                assert torch.all(output >= 0.0) and torch.all(output <= 1.0), f"output.min()={output.min()}, output.max()={output.max()}, output.shape={output.shape}"
                return output # [B, 3, 512, 512]

        outputs, _ = self._forward(ray_bundle=ray_bundle, renderers=renderers) # Sequence[B, H, W, C]
        return torch.stack([fn(output, renderer) for (output, renderer) in zip(outputs, renderers)], dim=0) # [R, B, 3, 512, 512]

    def _get_image_resample(self, ray_bundle: RayBundle, vae: AutoencoderKL, renderers: Sequence[Renderer], nerf_scale) -> Tensor:
        outputs, _ = self.get_latent(ray_bundle=ray_bundle, vae=vae, renderers=renderers, nerf_scale=nerf_scale) # [R, B, 4, 64, 64]
        R = outputs.shape[0]
        B = outputs.shape[1]
        outputs = outputs.reshape(R * B, *outputs.shape[2:]) # [R*B, 4, 64, 64]
        outputs = vae.decode(1 / vae.config['scaling_factor'] * outputs).sample # type: ignore
        outputs = (outputs / 2 + 0.5).clamp(0.0, 1.0)
        outputs = outputs.reshape(R, B, *outputs.shape[1:]) # [R, B, 3, 512, 512]
        return outputs # [R, B, 3, 512, 512]
