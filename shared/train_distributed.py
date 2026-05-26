"""DDP training script for SD 1.5 LoRA — run via `python -m shared.train_distributed`."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm

    from shared.pokemon_dataset import build_loaders_from_manifest
    from shared.sd_lora_models import build_sd_lora_stack

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"world_size={world_size}  device={device}  backend={backend}")

    train_dataset, val_dataset, _, _, _ = build_loaders_from_manifest(Path(args.manifest_path))

    def make_subset(dataset, frac, seed):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(dataset))[: max(1, int(len(dataset) * frac))]
        return Subset(dataset, indices)

    train_small = make_subset(train_dataset, args.fraction, args.seed)
    val_small = make_subset(val_dataset, args.fraction, args.seed + 1)

    train_loader = DataLoader(
        train_small, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, sampler=DistributedSampler(train_small, shuffle=True),
    )
    val_loader = DataLoader(
        val_small, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, sampler=DistributedSampler(val_small, shuffle=False),
    )

    vae, unet, text_encoder, noise_scheduler = build_sd_lora_stack(device, model_id=args.model_id)
    unet = DDP(unet, device_ids=[local_rank])
    optimizer = AdamW(unet.parameters(), lr=args.lr)

    checkpoint_file = Path(args.checkpoint_path)
    if rank == 0:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    train_losses: list = []
    val_losses: list = []
    start_epoch = 0

    if checkpoint_file.exists():
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        unet.module.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")

    def encode_prompt(encoder, input_ids, attention_mask):
        with torch.no_grad():
            return encoder(input_ids, attention_mask=attention_mask)[0]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        unet.train()
        running, local_n = 0.0, 0
        it = tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}") if rank == 0 else train_loader
        for batch in it:
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
            local_n += bs

        t_agg = torch.tensor([running, float(local_n)], device=device)
        dist.all_reduce(t_agg, op=dist.ReduceOp.SUM)
        train_losses.append((t_agg[0] / t_agg[1]).item())

        # switch to inference mode for validation
        unet.train(False)
        running, local_n = 0.0, 0
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
                local_n += bs

        v_agg = torch.tensor([running, float(local_n)], device=device)
        dist.all_reduce(v_agg, op=dist.ReduceOp.SUM)
        val_losses.append((v_agg[0] / v_agg[1]).item())

        if rank == 0:
            print(f"Epoch {epoch + 1}: train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")
            torch.save({
                "epoch": epoch + 1,
                "unet_state_dict": unet.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "model_id": args.model_id,
            }, checkpoint_file)
            (checkpoint_file.parent / "train_results.json").write_text(
                json.dumps({
                    "epochs_done": len(train_losses),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }, indent=2),
                encoding="utf-8",
            )

    dist.destroy_process_group()
    if rank == 0:
        print(f"Training complete — checkpoint: {checkpoint_file}")


if __name__ == "__main__":
    main()
