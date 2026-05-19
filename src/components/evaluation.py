"""Component for notebook 04_evaluation.ipynb."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact

_GIT_PKG = "pokegen-shared @ git+https://github.com/DO-2K23-26/mlops-pokegen.git"


@kfp_component(packages_to_install=[
    "torch", "diffusers", "pillow", "matplotlib", "transformers", _GIT_PKG,
])
def evaluation_component(
    metrics_output: Output[Artifact],
    loss_curves_output: Output[Artifact],
    samples_output: Output[Artifact],
    state_output: Output[Artifact],
    manifest_path: Input[Artifact],
    checkpoint_path: Input[Artifact],
    model_id: str = "runwayml/stable-diffusion-v1-5",
    num_inference_steps: int = 100,
    guidance_scale: float = 7.5,
    sample_count: int = 4,
) -> None:
    """Plot training curves and generate qualitative samples for evaluation."""
    import copy
    import json
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image

    from shared.pokemon_dataset import (
        build_loaders_from_manifest,
        get_metadata_from_png,
        metadata_to_conditioning,
        pick_device,
    )
    from shared.sd_lora_models import build_sd_lora_stack

    checkpoint = torch.load(checkpoint_path.path, map_location="cpu", weights_only=False)
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = Path(loss_curves_output.path)
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(loss_path, dpi=120)
    plt.close()

    metrics = {"train_losses": train_losses, "val_losses": val_losses, "epochs": len(train_losses)}
    metrics_file = Path(metrics_output.path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    device = pick_device()
    _, val_dataset, _, _, tokenizer = build_loaders_from_manifest(Path(manifest_path.path))
    vae, unet, text_encoder, _ = build_sd_lora_stack(device, model_id=model_id)
    unet.load_state_dict(
        torch.load(checkpoint_path.path, map_location=device, weights_only=False)["unet_state_dict"]
    )
    unet.train(False)

    unet_copy = copy.deepcopy(unet)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=vae,
        unet=unet_copy.merge_and_unload(),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
    ).to(device)
    del unet_copy

    def get_val_path(index):
        return val_dataset.dataset.image_paths[int(val_dataset.indices[index])]

    n_samples = min(sample_count, len(val_dataset))
    indices = random.sample(range(len(val_dataset)), n_samples)
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        path = get_val_path(idx)
        cond = metadata_to_conditioning(get_metadata_from_png(path))
        axes[0, col].imshow(pipe(cond, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0])
        axes[0, col].axis("off")
        axes[1, col].imshow(Image.open(path).convert("RGB").resize((512, 512)))
        axes[1, col].axis("off")

    plt.tight_layout()
    samples_file = Path(samples_output.path)
    samples_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(samples_file, dpi=120)
    plt.close()

    out = Path(state_output.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "04_evaluation",
        "checkpoint_path": str(Path(checkpoint_path.path).resolve()),
        "manifest_path": str(Path(manifest_path.path).resolve()),
        "model_id": model_id,
        "metrics_path": str(metrics_file.resolve()),
        "loss_curves_path": str(loss_path.resolve()),
        "eval_samples_path": str(samples_file.resolve()),
        "final_val_loss": val_losses[-1] if val_losses else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
