import torch

from torch import Tensor
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from typing import Dict, Union, List, Optional, Tuple, Callable

def get_controlnet_embedding(
    time: Tensor,
    latents_noised: Tensor,
    encoder_hidden_states: Tensor,
    added_cond_kwargs: Optional[Dict[str, Tensor]],
    controlnet: Union[ControlNetModel, List[ControlNetModel]],
    image: Union[Tensor, List[Tensor]],
    conditioning_scale: Union[float, List[float]] = 1.0,
    guess_mode: Union[bool, List[bool]] = False, # scales *= torch.logspace(-1, 0, len(down_block_res_samples) + 1)  # 0.1 to 1.0
) -> Tuple[List[Tensor], Tensor]:
    if isinstance(controlnet, (list, tuple)):
        multi_controlnet = torch.nn.ModuleList(controlnet)
    else:
        multi_controlnet = torch.nn.ModuleList([controlnet])
    n_control = len(multi_controlnet)
    assert n_control != 0
    
    if not isinstance(image, (list, tuple)):
        image = [image] * n_control
    if not isinstance(conditioning_scale, (list, tuple)):
        conditioning_scale = [float(conditioning_scale)] * n_control
    if not isinstance(guess_mode, (list, tuple)):
        guess_mode = [guess_mode] * n_control

    down_block_res_samples = None
    mid_block_res_sample = None
    for i, (_multi_controlnet, _image, _conditioning_scale, _guess_mode) in enumerate(zip(multi_controlnet, image, conditioning_scale, guess_mode)):
        down_samples, mid_sample = _multi_controlnet(
            sample=latents_noised,
            timestep=time,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=_image,
            conditioning_scale=_conditioning_scale,
            class_labels=None, # TODO: not sure what it does
            timestep_cond=None, # TODO: not sure what it does
            attention_mask=None, # TODO: not sure what it does
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=None, # TODO: maybe allow it?
            guess_mode=_guess_mode,
            return_dict=False,
        )

        # merge samples
        if i == 0:
            down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
        else:
            assert down_block_res_samples is not None and mid_block_res_sample is not None
            down_block_res_samples = [samples_prev + samples_curr for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)]
            mid_block_res_sample += mid_sample

    assert down_block_res_samples is not None and mid_block_res_sample is not None
    return down_block_res_samples, mid_block_res_sample

def get_controlnet_embeddings(
    condition_side_control: bool,
    uncondition_side_control: bool,
    time: Tensor,
    latents_noised: Tensor,
    text_embeddings_conditional: Tensor,
    text_embeddings_unconditional: Tensor,
    added_cond_kwargs_conditional: Optional[Dict[str, Tensor]],
    added_cond_kwargs_unconditional: Optional[Dict[str, Tensor]],
    controlnet: Union[ControlNetModel, List[ControlNetModel]],
    image: Union[Tensor, List[Tensor]],
    conditioning_scale: Union[float, List[float]] = 1.0,
    guess_mode: Union[bool, List[bool]] = False,
) -> Tuple[Optional[List[Tensor]], Optional[Tensor]]:
    if condition_side_control and not uncondition_side_control:
        down_block_res_samples, mid_block_res_sample = get_controlnet_embedding(
            time=time,
            latents_noised=latents_noised,
            encoder_hidden_states=text_embeddings_conditional,
            added_cond_kwargs=added_cond_kwargs_conditional,
            controlnet=controlnet,
            image=image,
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
        )
        down_block_res_samples = [torch.cat([d, torch.zeros_like(d)]) for d in down_block_res_samples]
        mid_block_res_sample = torch.cat([mid_block_res_sample, torch.zeros_like(mid_block_res_sample)])
    elif not condition_side_control and uncondition_side_control:
        down_block_res_samples, mid_block_res_sample = get_controlnet_embedding(
            time=time,
            latents_noised=latents_noised,
            encoder_hidden_states=text_embeddings_unconditional,
            added_cond_kwargs=added_cond_kwargs_unconditional,
            controlnet=controlnet,
            image=image,
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
        )
        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
    elif condition_side_control and uncondition_side_control:
        added_cond_kwargs = None
        if added_cond_kwargs_conditional is not None and added_cond_kwargs_unconditional is not None:
            added_cond_kwargs = {
                k: torch.cat([v_cond, v_uncond], dim=0) for k, v_cond, v_uncond in zip(
                    added_cond_kwargs_conditional.keys(),
                    added_cond_kwargs_conditional.values(),
                    added_cond_kwargs_unconditional.values(),
                )
            }
        down_block_res_samples, mid_block_res_sample = get_controlnet_embedding(
            time=time,
            latents_noised=torch.cat([latents_noised, latents_noised], dim=0),
            encoder_hidden_states=torch.cat([text_embeddings_conditional, text_embeddings_unconditional], dim=0),
            added_cond_kwargs=added_cond_kwargs,
            controlnet=controlnet,
            image=torch.cat([image, image], dim=0) if isinstance(image, Tensor) else [torch.cat([i, i], dim=0) for i in image],
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
        )
    else:
        down_block_res_samples, mid_block_res_sample = None, None
    return down_block_res_samples, mid_block_res_sample

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
    alphas_cumprod = scheduler.alphas_cumprod.to(velocity.device)
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


def cfg_rescale_x0(cfg_rescale: float, noise_pred_x0: Tensor, noise_pred_conditional: Tensor, latents_noised: Tensor, scheduler: DDIMScheduler, t: Tensor, n_views: int = 1, eps: float = 1e-8) -> Tensor:
    if cfg_rescale > 0 and cfg_rescale < 1:
        noise_pred_x0 = noise_pred_x0.view(-1, n_views, *noise_pred_x0.shape[1:]) # [B, n_view, C, H, W]
        noise_pred_conditional_x0: Tensor = scheduler.step(noise_pred_conditional, t, latents_noised).pred_original_sample # type: ignore
        noise_pred_conditional_x0 = noise_pred_conditional_x0.view(-1, n_views, *noise_pred_conditional_x0.shape[1:]) # [B, n_view, C, H, W]
        conditional_vs_cfg = (noise_pred_conditional_x0.std([1, 2, 3, 4], keepdim=True) + eps) / (noise_pred_x0.std([1, 2, 3, 4], keepdim=True) + eps) # [B, 1, 1, 1, 1]
        conditional_vs_cfg = conditional_vs_cfg.squeeze(1).repeat_interleave(n_views, dim=0) # [B*n_view, 1, 1, 1]
        noise_pred_x0 = noise_pred_x0.view(-1, *noise_pred_x0.shape[2:]) # [B*n_view, C, H, W]
        noise_pred_x0 = cfg_rescale * (noise_pred_x0 * conditional_vs_cfg) + (1 - cfg_rescale) * noise_pred_x0 # [B*n_view, C, H, W]
    return noise_pred_x0


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
        cfg: float,
        lora_scale: float, # > 0 to enable lora
        t: Tensor,
        scheduler: DDIMScheduler,
        reconstruction_loss: bool,
        cfg_rescale: float,
        # inpainting
        extra_latents: Optional[Tensor] = None,
        # controlnet
        controlnet: Optional[Union[ControlNetModel, List[ControlNetModel]]] = None,
        image: Optional[Union[Tensor, List[Tensor]]] = None,
        conditioning_scale: Optional[Union[float, List[float]]] = None,
        guess_mode: Optional[Union[bool, List[bool]]] = None,
        condition_side_control: bool = True,
        uncondition_side_control: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore
        if extra_latents is not None:
            extra_latents = extra_latents.to(memory_format=torch.channels_last) # type: ignore
    latents_noised = scheduler.scale_model_input(sample=latents_noised, timestep=int(t.item())) # type: ignore

    down_block_res_samples, mid_block_res_sample = None, None
    if controlnet is not None:
        assert image is not None and conditioning_scale is not None and guess_mode is not None
        down_block_res_samples, mid_block_res_sample = get_controlnet_embeddings(
            condition_side_control=condition_side_control,
            uncondition_side_control=uncondition_side_control,
            time=t,
            latents_noised=latents_noised,
            text_embeddings_conditional=text_embeddings_conditional,
            text_embeddings_unconditional=text_embeddings_unconditional,
            added_cond_kwargs_conditional=None,
            added_cond_kwargs_unconditional=None,
            controlnet=controlnet,
            image=image,
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
        )

    pred = unet_sd(
        torch.cat([torch.cat([latents_noised, extra_latents], dim=1) if extra_latents is not None else latents_noised] * 2, dim=0), # type: ignore
        t,
        encoder_hidden_states=torch.cat([text_embeddings_conditional, text_embeddings_unconditional], dim=0),
        cross_attention_kwargs={
            'scale': lora_scale
        } if lora_scale > 0 else {},
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    ).sample
    noise_pred = get_noise_pred(scheduler, pred, t, latents_noised)

    # cfg
    noise_pred_conditional, noise_pred_unconditional = noise_pred.chunk(2)
    noise_pred = guidance(unconditional=noise_pred_unconditional, conditional=noise_pred_conditional, cfg=cfg)

    # reconstruction loss
    if not reconstruction_loss:
        return noise_pred, None

    noise_pred_x0: Tensor = scheduler.step(noise_pred, t, latents_noised).pred_original_sample # [B*n_view, C, H, W] # type: ignore
    noise_pred_x0 = cfg_rescale_x0(
        cfg_rescale=cfg_rescale,
        noise_pred_x0=noise_pred_x0,
        noise_pred_conditional=noise_pred_conditional,
        latents_noised=latents_noised,
        scheduler=scheduler,
        t=t,
    ) # cfg rescale

    return noise_pred, noise_pred_x0

def predict_noise_sdxl_turbo(
    unet_sdxl: UNet2DConditionModel,
    latents_noised: Tensor,
    text_embeddings_unconditional: Tensor,
    text_embeddings_unconditional_pooled: Tensor,
    text_embeddings_unconditional_micro: Tensor,
    lora_scale: float, # > 0 to enable lora
    t: Tensor,
    scheduler: DDIMScheduler,
    reconstruction_loss: bool,
    # inpainting
    extra_latents: Optional[Tensor] = None,
    # controlnet
    controlnet: Optional[Union[ControlNetModel, List[ControlNetModel]]] = None,
    image: Optional[Union[Tensor, List[Tensor]]] = None,
    conditioning_scale: Optional[Union[float, List[float]]] = None,
    guess_mode: Optional[Union[bool, List[bool]]] = None,
    # we don't care about the following arguments
    condition_side_control: bool = True,
    uncondition_side_control: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore
        if extra_latents is not None:
            extra_latents = extra_latents.to(memory_format=torch.channels_last) # type: ignore
    latents_noised = scheduler.scale_model_input(sample=latents_noised, timestep=int(t.item())) # type: ignore

    added_cond_kwargs = {
        "text_embeds": text_embeddings_unconditional_pooled,
        "time_ids": text_embeddings_unconditional_micro,
    }
    
    down_block_res_samples, mid_block_res_sample = None, None
    if controlnet is not None:
        assert image is not None and conditioning_scale is not None and guess_mode is not None
        down_block_res_samples, mid_block_res_sample = get_controlnet_embedding(
            time=t,
            latents_noised=latents_noised,
            encoder_hidden_states=text_embeddings_unconditional,
            added_cond_kwargs=added_cond_kwargs,
            controlnet=controlnet,
            image=image,
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
        )

    pred = unet_sdxl(
        torch.cat([latents_noised, extra_latents], dim=1) if extra_latents is not None else latents_noised,
        t,
        encoder_hidden_states=text_embeddings_unconditional,
        cross_attention_kwargs={
            'scale': lora_scale
        } if lora_scale > 0 else {},
        added_cond_kwargs=added_cond_kwargs,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    ).sample
    noise_pred = get_noise_pred(scheduler, pred, t, latents_noised)

    # reconstruction loss
    if not reconstruction_loss:
        return noise_pred, None

    noise_pred_x0: Tensor = scheduler.step(noise_pred, t, latents_noised).pred_original_sample # [B*n_view, C, H, W] # type: ignore

    return noise_pred, noise_pred_x0

def predict_noise_sdxl(
    unet_sdxl: UNet2DConditionModel,
    latents_noised: Tensor,
    text_embeddings_conditional: Tensor,
    text_embeddings_unconditional: Tensor,
    text_embeddings_conditional_pooled: Tensor,
    text_embeddings_unconditional_pooled: Tensor,
    text_embeddings_conditional_micro: Tensor,
    text_embeddings_unconditional_micro: Tensor,
    cfg: float,
    lora_scale: float, # > 0 to enable lora
    t: Tensor,
    scheduler: DDIMScheduler,
    reconstruction_loss: bool,
    cfg_rescale: float,
    # inpainting
    extra_latents: Optional[Tensor] = None,
    # controlnet
    controlnet: Optional[Union[ControlNetModel, List[ControlNetModel]]] = None,
    image: Optional[Union[Tensor, List[Tensor]]] = None,
    conditioning_scale: Optional[Union[float, List[float]]] = None,
    guess_mode: Optional[Union[bool, List[bool]]] = None,
    condition_side_control: bool = True,
    uncondition_side_control: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore
        if extra_latents is not None:
            extra_latents = extra_latents.to(memory_format=torch.channels_last) # type: ignore
    latents_noised = scheduler.scale_model_input(sample=latents_noised, timestep=int(t.item())) # type: ignore

    added_cond_kwargs = {
        "text_embeds": torch.cat([
            text_embeddings_conditional_pooled,
            text_embeddings_unconditional_pooled,
        ], dim=0),
        "time_ids": torch.cat([
            text_embeddings_conditional_micro,
            text_embeddings_unconditional_micro,
        ], dim=0),
    }
    
    down_block_res_samples, mid_block_res_sample = None, None
    if controlnet is not None:
        assert image is not None and conditioning_scale is not None and guess_mode is not None
        down_block_res_samples, mid_block_res_sample = get_controlnet_embedding(
            time=t,
            latents_noised=latents_noised,
            encoder_hidden_states=text_embeddings_conditional,
            added_cond_kwargs=added_cond_kwargs,
            controlnet=controlnet,
            image=image,
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
        )
        down_block_res_samples = [torch.cat([d if condition_side_control else torch.zeros_like(d), d if uncondition_side_control else torch.zeros_like(d)]) for d in down_block_res_samples]
        mid_block_res_sample = torch.cat([mid_block_res_sample if condition_side_control else torch.zeros_like(mid_block_res_sample), mid_block_res_sample if uncondition_side_control else torch.zeros_like(mid_block_res_sample)])

    pred = unet_sdxl(
        torch.cat([torch.cat([latents_noised, extra_latents], dim=1) if extra_latents is not None else latents_noised] * 2, dim=0), # type: ignore
        t,
        encoder_hidden_states=torch.cat([text_embeddings_conditional, text_embeddings_unconditional], dim=0),
        cross_attention_kwargs={
            'scale': lora_scale
        } if lora_scale > 0 else {},
        added_cond_kwargs=added_cond_kwargs,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    ).sample
    noise_pred = get_noise_pred(scheduler, pred, t, latents_noised)

    # cfg
    noise_pred_conditional, noise_pred_unconditional = noise_pred.chunk(2)
    noise_pred = guidance(unconditional=noise_pred_unconditional, conditional=noise_pred_conditional, cfg=cfg)

    # reconstruction loss
    if not reconstruction_loss:
        return noise_pred, None

    noise_pred_x0: Tensor = scheduler.step(noise_pred, t, latents_noised).pred_original_sample # [B*n_view, C, H, W] # type: ignore
    noise_pred_x0 = cfg_rescale_x0(
        cfg_rescale=cfg_rescale,
        noise_pred_x0=noise_pred_x0,
        noise_pred_conditional=noise_pred_conditional,
        latents_noised=latents_noised,
        scheduler=scheduler,
        t=t,
    ) # cfg rescale

    return noise_pred, noise_pred_x0


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
        reconstruction_loss: bool,
        cfg_rescale: float,
) -> Tuple[Tensor, Optional[Tensor]]:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore
    latents_noised = scheduler.scale_model_input(sample=latents_noised, timestep=int(t.item())) # type: ignore

    pred = unet_z123(
        torch.cat([torch.cat([latents_noised] * 2, dim=0), torch.cat([latents_image] * 2, dim=0)], dim=1), # type: ignore
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

    # reconstruction loss
    if not reconstruction_loss:
        return noise_pred, None

    noise_pred_x0: Tensor = scheduler.step(noise_pred, t, latents_noised).pred_original_sample # [B*n_view, C, H, W] # type: ignore
    noise_pred_x0 = cfg_rescale_x0(
        cfg_rescale=cfg_rescale,
        noise_pred_x0=noise_pred_x0,
        noise_pred_conditional=noise_pred_conditional,
        latents_noised=latents_noised,
        scheduler=scheduler,
        t=t,
    ) # cfg rescale

    return noise_pred, noise_pred_x0


def predict_noise_mvdream(
    unet_mvdream: Callable,
    latents_noised: Tensor,
    text_embeddings_conditional: List[Tensor],
    text_embeddings_unconditional: List[Tensor],
    camera_embeddings: Tensor,
    cfg: float,
    t: Tensor,
    scheduler: DDIMScheduler,
    n_views: int,
    reconstruction_loss: bool,
    cfg_rescale: float,
) -> Tuple[Tensor, Optional[Tensor]]:
    if torch.backends.cudnn.version() >= 7603: # type: ignore
        # memory format convertion (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        latents_noised = latents_noised.to(memory_format=torch.channels_last) # type: ignore
    latents_noised = scheduler.scale_model_input(sample=latents_noised, timestep=int(t.item())) # type: ignore

    batch_size = len(text_embeddings_conditional)
    assert batch_size == len(text_embeddings_unconditional)

    pred = unet_mvdream(
        x=torch.cat([latents_noised] * 2), # type: ignore
        timesteps=torch.tensor([t] * batch_size * 2, device=t.device),
        context=torch.stack(text_embeddings_conditional + text_embeddings_unconditional),
        num_frames=batch_size,
        camera=torch.cat([camera_embeddings] * 2),
    )
    noise_pred = get_noise_pred(scheduler, pred, t, latents_noised)

    # cfg
    noise_pred_conditional, noise_pred_unconditional = noise_pred.chunk(2)
    noise_pred = guidance(unconditional=noise_pred_unconditional, conditional=noise_pred_conditional, cfg=cfg)

    # reconstruction loss
    if not reconstruction_loss:
        return noise_pred, None

    noise_pred_x0: Tensor = scheduler.step(noise_pred, t, latents_noised).pred_original_sample # [B*n_view, C, H, W] # type: ignore
    noise_pred_x0 = cfg_rescale_x0(
        cfg_rescale=cfg_rescale,
        noise_pred_x0=noise_pred_x0,
        noise_pred_conditional=noise_pred_conditional,
        latents_noised=latents_noised,
        scheduler=scheduler,
        t=t,
        n_views=n_views,
    ) # cfg rescale

    return noise_pred, noise_pred_x0


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
    latents_noised = scheduler.scale_model_input(sample=latents_noised, timestep=int(t.item())) # type: ignore

    pred = unet_lora(
        torch.cat([latents_noised] * 2, dim=0), # type: ignore
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
    latents_noised = scheduler.scale_model_input(sample=latents_noised, timestep=int(t.item())) # type: ignore

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
