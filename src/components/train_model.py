"""Component for notebook 03_train_model.ipynb."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact

_GIT_PKG = "pokegen-shared @ git+https://github.com/DO-2K23-26/mlops-pokegen.git"


@kfp_component(packages_to_install=["torch", "transformers", "diffusers", "peft", "pillow", "tqdm"],
               ressources={"cpu": "12", "memory": "32Gi", "nvidia.com/gpu": "1"})
def train_model_component(
    checkpoint_output: Output[Artifact],
    state_output: Output[Artifact],
    manifest_path: Input[Artifact],
    model_id: str = "runwayml/stable-diffusion-v1-5",
    epochs: int = 1,
    lr: float = 1e-5,
    fraction: float = 1.0,
    seed: int = 42,
    batch_size: int = 4,
    num_workers: int = 0,
) -> None:
    """Train the LoRA U-Net, persist the checkpoint, and write step state."""
    import json
    from pathlib import Path

    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from shared.pokemon_dataset import build_loaders_from_manifest, pick_device
    from shared.sd_lora_models import build_sd_lora_stack

    train_dataset, val_dataset, _, _, _ = build_loaders_from_manifest(Path(manifest_path.path))

    def make_subset(dataset, subset_fraction, subset_seed):
        rng = np.random.default_rng(subset_seed)
        size = len(dataset)
        indices = rng.permutation(size)[: max(1, int(size * subset_fraction))]
        return Subset(dataset, indices)

    train_small = make_subset(train_dataset, fraction, seed)
    val_small = make_subset(val_dataset, fraction, seed + 1)

    train_loader = DataLoader(train_small, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_small, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = pick_device()
    vae, unet, text_encoder, noise_scheduler = build_sd_lora_stack(device, model_id=model_id)
    optimizer = AdamW(unet.parameters(), lr=lr)

    checkpoint_file = Path(checkpoint_output.path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    train_losses: list = []
    val_losses: list = []
    start_epoch = 0

    if checkpoint_file.exists():
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        print(f"Resuming from epoch {start_epoch}")

    def encode_prompt(encoder, input_ids, attention_mask):
        with torch.no_grad():
            return encoder(input_ids, attention_mask=attention_mask)[0]

    for epoch in range(start_epoch, epochs):
        unet.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}"):
            pv = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                latents = vae.encode(pv).latent_dist.sample() * 0.18215
            enc = encode_prompt(text_encoder, ids, mask)
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, t)
            pred = unet(noisy, t, enc).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * bs
        train_losses.append(running / len(train_small))

        unet.train(False)
        running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pv = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                latents = vae.encode(pv).latent_dist.sample() * 0.18215
                enc = encode_prompt(text_encoder, ids, mask)
                noise = torch.randn_like(latents)
                bs = latents.shape[0]
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy = noise_scheduler.add_noise(latents, noise, t)
                pred = unet(noisy, t, enc).sample
                running += torch.nn.functional.mse_loss(pred.float(), noise.float()).item() * bs
        val_losses.append(running / len(val_small))
        print(f"Epoch {epoch + 1}: train={train_losses[-1]:.4f} val={val_losses[-1]:.4f}")

        torch.save({
            "epoch": epoch + 1,
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "model_id": model_id,
        }, checkpoint_file)

    out = Path(state_output.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "03_train_model",
        "manifest_path": str(Path(manifest_path.path).resolve()),
        "checkpoint_path": str(checkpoint_file.resolve()),
        "model_id": model_id,
        "epochs_done": len(train_losses),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
