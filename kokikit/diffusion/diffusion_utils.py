import torch

from torch import Tensor
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers import DDIMScheduler
from typing import Dict, Union, List, Optional, Any, Callable


def extract_lora_diffusers(unet: UNet2DConditionModel) -> Dict[str, Union[LoRAAttnAddedKVProcessor, LoRAAttnProcessor]]:
    # https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers
    unet_lora_attn_procs: Dict[str, Union[LoRAAttnAddedKVProcessor, LoRAAttnProcessor]] = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim # type: ignore

        hidden_size: int
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1] # type: ignore
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id] # type: ignore
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id] # type: ignore
        else:
            raise ValueError(f"Unknown block name: {name}")

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor

        unet_lora_attn_procs[name] = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    return unet_lora_attn_procs


def noise_to_velocity(noise: Tensor, t: Tensor, latents_noised: Tensor, scheduler):
    latents_clean = scheduler.step(noise, t, latents_noised).pred_original_sample # type: ignore
    velocity = scheduler.get_velocity(latents_clean, noise, t) # type: ignore
    return velocity


def velocity_to_noise(velocity: Tensor, t: Tensor, latents_noised: Tensor, scheduler):
    alphas_cumprod = scheduler.alphas_cumprod
    alpha_t = alphas_cumprod[t]**0.5
    sigma_t = (1 - alphas_cumprod[t])**0.5

    noise = latents_noised * sigma_t.view(-1, 1, 1, 1) + velocity * alpha_t.view(-1, 1, 1, 1)
    return noise


def get_noise_pred(scheduler: DDIMScheduler, pred: Tensor, t: Tensor, latents_noised: Tensor):
    if scheduler.prediction_type == 'v_prediction':
        noise_pred = velocity_to_noise(pred, t, latents_noised, scheduler)
    elif scheduler.prediction_type == 'epsilon':
        noise_pred = pred
    else:
        raise ValueError(f"Unknown prediction type: {scheduler.prediction_type}")
    return noise_pred


def get_velocity_pred(scheduler: DDIMScheduler, pred: Tensor, t: Tensor, latents_noised: Tensor):
    if scheduler.prediction_type == 'v_prediction':
        velocity_pred = pred
    elif scheduler.prediction_type == 'epsilon':
        velocity_pred = noise_to_velocity(pred, t, latents_noised, scheduler)
    else:
        raise ValueError(f"Unknown prediction type: {scheduler.prediction_type}")
    return velocity_pred


def guidance(unconditional: Tensor, conditional: Tensor, cfg: float):
    return unconditional + cfg * (conditional - unconditional)


def get_prependicualr_component(x: Tensor, y: Tensor) -> Tensor:
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum()) / (torch.norm(y)**2)) * y


def weighted_prependicualr_aggricator(conditional: Tensor, pred_perpneg: List[Tensor], weights_perpneg: List[float]) -> Tensor:
    accumulated_output = 0
    for i, w in enumerate(weights_perpneg):
        accumulated_output += w * get_prependicualr_component(pred_perpneg[i].unsqueeze(0), conditional)

    return accumulated_output + conditional


def guidance_perpneg(unconditional: Tensor, conditional: Tensor, cfg: float, pred_perpneg: List[Tensor], weights_perpneg: List[float]) -> Tensor:
    return unconditional + cfg * weighted_prependicualr_aggricator(conditional=conditional, pred_perpneg=pred_perpneg, weights_perpneg=weights_perpneg)


def predict_noise_sd( # for any stable-diffusion-based models
        unet_sd: UNet2DConditionModel, # either unet_base or unet_lora
        latents_noised: Tensor,
        text_embeddings_conditional: Tensor,
        text_embeddings_unconditional: Tensor,
        text_embeddings_perpneg: Optional[List[Tensor]], # List[77, 1024]
        weights_perpneg: Optional[List[float]], # List[float]
        cfg: float,
        lora_scale: float, # > 0 to enable lora
        t: Tensor,
        scheduler: DDIMScheduler,
) -> Tensor:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore

    pred = unet_sd(
        torch.cat([latents_noised] * 2, dim=0),
        t,
        encoder_hidden_states=torch.cat([text_embeddings_conditional, text_embeddings_unconditional], dim=0),
        cross_attention_kwargs={
            'scale': lora_scale
        } if lora_scale > 0 else {},
    ).sample
    noise_pred = get_noise_pred(scheduler, pred, t, latents_noised)

    # cfg
    noise_pred_conditional, noise_pred_unconditional = noise_pred.chunk(2)

    if text_embeddings_perpneg is not None and weights_perpneg is not None and len(text_embeddings_perpneg) > 0:
        pred_perpneg = unet_sd(
            torch.cat([latents_noised] * len(text_embeddings_perpneg), dim=0),
            t,
            encoder_hidden_states=torch.cat(text_embeddings_perpneg, dim=0),
            cross_attention_kwargs={
                'scale': lora_scale
            } if lora_scale > 0 else {},
        ).sample

        pred_perpneg = pred_perpneg - noise_pred_unconditional
        noise_pred_conditional = noise_pred_conditional - noise_pred_unconditional

        noise_pred = guidance_perpneg(
            unconditional=noise_pred_unconditional,
            conditional=noise_pred_conditional,
            cfg=cfg,
            pred_perpneg=pred_perpneg,
            weights_perpneg=weights_perpneg,
        )
    else:
        noise_pred = guidance(unconditional=noise_pred_unconditional, conditional=noise_pred_conditional, cfg=cfg)

    return noise_pred


def predict_noise_z123( # for any z123-based models
        unet_z123: UNet2DConditionModel,
        latents_noised: Tensor, # [B, C, H, W] e.g. [1, 4, 32, 32]
        latents_image: Tensor, # [B, C, H, W] e.g. [1, 4, 32, 32]
        angle_embeddings_conditional: Tensor, # [B, 1, 768]
        angle_embeddings_unconditional: Tensor, # [B, 1, 768]
        cfg: float,
        lora_scale: float, # > 0 to enable lora
        t: Tensor,
        scheduler: DDIMScheduler,
) -> Tensor:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore

    pred = unet_z123(
        torch.cat([torch.cat([latents_noised] * 2, dim=0), torch.cat([latents_image] * 2, dim=0)], dim=1),
        t,
        encoder_hidden_states=torch.cat([angle_embeddings_conditional, angle_embeddings_unconditional], dim=0),
        cross_attention_kwargs={
            'scale': lora_scale
        } if lora_scale > 0 else {},
    ).sample
    noise_pred = get_noise_pred(scheduler, pred, t, latents_noised)

    # cfg
    noise_pred_conditional, noise_pred_unconditional = noise_pred.chunk(2)
    noise_pred = guidance(unconditional=noise_pred_unconditional, conditional=noise_pred_conditional, cfg=cfg)

    return noise_pred


def predict_noise_mvdream(
    unet_mvdream: Callable,
    latents_noised: Tensor,
    text_embeddings_conditional: List[Tensor],
    text_embeddings_unconditional: List[Tensor],
    camera_embeddings: Tensor,
    cfg: float,
    t: Tensor,
    scheduler: DDIMScheduler,
) -> Tensor:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore

    batch_size = len(text_embeddings_conditional)
    assert batch_size == len(text_embeddings_unconditional)

    pred = unet_mvdream(
        x=torch.cat([latents_noised] * 2),
        timesteps=torch.tensor([t] * batch_size * 2, device=t.device),
        context=torch.stack(text_embeddings_unconditional + text_embeddings_conditional),
        num_frames=batch_size,
        camera=torch.cat([camera_embeddings] * 2),
    )
    noise_pred = get_noise_pred(scheduler, pred, t, latents_noised)

    # cfg
    noise_pred_conditional, noise_pred_unconditional = noise_pred.chunk(2)
    noise_pred = guidance(unconditional=noise_pred_unconditional, conditional=noise_pred_conditional, cfg=cfg)

    return noise_pred


def predict_velocity_lora(
    unet_lora: UNet2DConditionModel,
    latents_noised: Tensor,
    text_embeddings_conditional: Tensor,
    text_embeddings_unconditional: Tensor,
    text_embeddings_perpneg: Optional[List[Tensor]], # List[77, 1024]
    weights_perpneg: Optional[List[float]], # List[float]
    cfg: float,
    lora_scale: float,
    t: Tensor,
    scheduler: DDIMScheduler,
) -> Tensor:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore

    pred = unet_lora(
        torch.cat([latents_noised] * 2, dim=0),
        t,
        encoder_hidden_states=torch.cat([text_embeddings_conditional, text_embeddings_unconditional], dim=0),
        cross_attention_kwargs={
            'scale': lora_scale
        },
    ).sample
    velocity_pred = get_velocity_pred(scheduler, pred, t, latents_noised)

    # cfg
    velocity_pred_conditional, velocity_pred_unconditional = velocity_pred.chunk(2)

    if text_embeddings_perpneg is not None and weights_perpneg is not None and len(text_embeddings_perpneg) > 0:
        assert False, "The following code is not tested"
        pred_perpneg = unet_lora(
            torch.cat([latents_noised] * len(text_embeddings_perpneg), dim=0),
            t,
            encoder_hidden_states=torch.cat(text_embeddings_perpneg, dim=0),
            cross_attention_kwargs={
                'scale': lora_scale
            },
        ).sample

        pred_perpneg = pred_perpneg - velocity_pred_unconditional
        velocity_pred_conditional = velocity_pred_conditional - velocity_pred_unconditional

        velocity_pred = guidance_perpneg(
            unconditional=velocity_pred_unconditional,
            conditional=velocity_pred_conditional,
            cfg=cfg,
            pred_perpneg=pred_perpneg,
            weights_perpneg=weights_perpneg,
        )
    else:
        velocity_pred = guidance(unconditional=velocity_pred_unconditional, conditional=velocity_pred_conditional, cfg=cfg)

    return velocity_pred


def predict_velocity_base(
    unet_base: UNet2DConditionModel,
    latents_noised: Tensor,
    text_embeddings_conditional: Tensor,
    text_embeddings_unconditional: Tensor,
    cfg: float,
    t: Tensor,
    scheduler: DDIMScheduler,
) -> Tensor:
    assert False, "The following code is not tested"
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore

    pred = unet_base(
        torch.cat([latents_noised] * 2, dim=0),
        t,
        encoder_hidden_states=torch.cat([text_embeddings_conditional, text_embeddings_unconditional], dim=0),
        cross_attention_kwargs={},
    ).sample
    velocity_pred = get_velocity_pred(scheduler, pred, t, latents_noised)

    # cfg
    velocity_pred_conditional, velocity_pred_unconditional = velocity_pred.chunk(2)
    velocity_pred = guidance(unconditional=velocity_pred_unconditional, conditional=velocity_pred_conditional, cfg=cfg)

    return velocity_pred


def predict_velocity_z123( # for any z123-based models
        unet_z123: UNet2DConditionModel,
        latents_noised: Tensor, # [B, C, H, W] e.g. [1, 4, 32, 32]
        latents_image: Tensor, # [B, C, H, W] e.g. [1, 4, 32, 32]
        angle_embeddings_conditional: Tensor, # [B, 1, 768]
        angle_embeddings_unconditional: Tensor, # [B, 1, 768]
        cfg: float,
        lora_scale: float, # > 0 to enable lora
        t: Tensor,
        scheduler: DDIMScheduler,
) -> Tensor:
    raise NotImplementedError