"""Component for notebook 03_train_model.ipynb."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from shared.paths import STEP_03_CHECKPOINT
from shared.pokemon_dataset import build_loaders_from_manifest, pick_device
from shared.sd_lora_models import DEFAULT_MODEL_ID, build_sd_lora_stack
from shared.step_artifacts import state_path


from kfp.dsl import component as kfp_component, Input, Output, Artifact



def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


@kfp_component(packages_to_install=["torch", "transformers", "diffusers", "peft", "pillow", "tqdm"])
def train_model_component(
    checkpoint_output: Output[Artifact],
    state_output: Output[Artifact],
    manifest_path: Input[Artifact],
    model_id: str = DEFAULT_MODEL_ID,
    epochs: int = 1,
    lr: float = 1e-5,
    fraction: float = 1 / 1120,
    seed: int = 42,
    batch_size: int = 16,
    num_workers: int = 4,
) -> None:
    """Train the LoRA U-Net, persist the checkpoint, and write the step state."""

    train_dataset, val_dataset, _, _, _ = build_loaders_from_manifest(Path(manifest_path.path))

    device = pick_device()
    vae, unet, text_encoder, noise_scheduler = build_sd_lora_stack(device, model_id=model_id)

    optimizer = AdamW(unet.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_epoch = 0

    checkpoint_file = Path(checkpoint_output.path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    def make_subset(dataset, subset_fraction: float, subset_seed: int):
        size = len(dataset)
        rng = np.random.default_rng(subset_seed)
        subset_size = int(size * subset_fraction)
        indices = rng.permutation(size)[:subset_size]
        return Subset(dataset, indices)

    train_dataset_small = make_subset(train_dataset, fraction, seed)
    val_dataset_small = make_subset(val_dataset, fraction, seed + 1)

    train_loader = DataLoader(
        train_dataset_small,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset_small,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if checkpoint_file.exists():
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        print(f"Reprise epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch")

    def encode_prompt(text_encoder_model, input_ids, attention_mask):
        with torch.no_grad():
            return text_encoder_model(input_ids, attention_mask=attention_mask)[0]

    print(f"Starting training for {epochs} epochs...")

    if start_epoch < epochs:
        for epoch in range(start_epoch, epochs):
            unet.train()
            running = 0.0

            for batch in tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}"):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                encodings = encode_prompt(text_encoder, input_ids, attention_mask)

                noise = torch.randn_like(latents)
                batch_size_value = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size_value,),
                    device=device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prediction = unet(noisy_latents, timesteps, encodings).sample

                loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()

                running += loss.item() * batch_size_value

            train_losses.append(running / len(train_dataset_small))

            unet.eval()
            running = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    encodings = encode_prompt(text_encoder, input_ids, attention_mask)

                    noise = torch.randn_like(latents)
                    batch_size_value = latents.shape[0]

                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (batch_size_value,),
                        device=device,
                    ).long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    prediction = unet(noisy_latents, timesteps, encodings).sample

                    running += torch.nn.functional.mse_loss(
                        prediction.float(),
                        noise.float(),
                    ).item() * batch_size_value

            val_losses.append(running / len(val_dataset_small))

            print(
                f"Epoch {epoch + 1}: "
                f"train={train_losses[-1]:.4f} "
                f"val={val_losses[-1]:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "model_id": model_id,
                },
                checkpoint_file,
            )

    payload: dict[str, Any] = {
        "manifest_path": str(Path(manifest_path.path).resolve()),
        "checkpoint_path": str(checkpoint_file.resolve()),
        "model_id": model_id,
        "epochs_done": len(train_losses),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }
    _write_json(state_output.path, {"step": "03_train_model", **payload})
