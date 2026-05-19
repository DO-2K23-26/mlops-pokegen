"""Chargement SD 1.5 + LoRA PEFT (partagé entre notebooks train / eval / registry)."""

from __future__ import annotations

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def build_sd_lora_stack(
    device: torch.device,
    model_id: str = DEFAULT_MODEL_ID,
    lora_r: int = 8,
    lora_alpha: int = 32,
):
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    return vae, unet, text_encoder, noise_scheduler
