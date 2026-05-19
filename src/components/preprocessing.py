"""Component for notebook 02_preprocessing.ipynb."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact

_GIT_PKG = "pokegen-shared @ git+https://github.com/DO-2K23-26/mlops-pokegen.git"


@kfp_component(packages_to_install=["torch", "torchvision", "pillow", "transformers", _GIT_PKG])
def preprocessing_component(
    manifest_output: Output[Artifact],
    state_output: Output[Artifact],
    image_dir: Input[Artifact],
    img_size: int = 512,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 0,
) -> None:
    """Validate card images, build the preprocessing manifest, and write step state."""
    import json
    from pathlib import Path

    from shared.pokemon_dataset import (
        build_tokenizer_and_loaders,
        collect_valid_image_paths,
        pick_device,
        save_preprocessing_manifest,
    )

    # The pull_data state JSON contains the actual cards_dir path
    state = json.loads(Path(image_dir.path).read_text(encoding="utf-8"))
    source_dir = Path(state["cards_dir"])

    device = pick_device()
    print(f"Using device: {device}")

    image_paths = collect_valid_image_paths(source_dir)
    print(f"Valid images with pokemon_metadata: {len(image_paths)}")

    train_dataset, val_dataset, _, _, _ = build_tokenizer_and_loaders(
        image_dir=source_dir,
        batch_size=batch_size,
        img_size=img_size,
        val_fraction=val_fraction,
        split_seed=split_seed,
        num_workers=num_workers,
    )

    manifest_file = Path(manifest_output.path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    save_preprocessing_manifest(
        manifest_file,
        image_dir=source_dir,
        image_paths=image_paths,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        img_size=img_size,
        batch_size=batch_size,
        split_seed=split_seed,
    )

    out = Path(state_output.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "02_preprocessing",
        "cards_dir": str(source_dir.resolve()),
        "manifest_path": str(manifest_file.resolve()),
        "n_images": len(image_paths),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "img_size": img_size,
        "batch_size": batch_size,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Step output saved for 02_preprocessing")
