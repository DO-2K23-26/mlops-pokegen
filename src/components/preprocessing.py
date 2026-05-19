"""Component for notebook 02_preprocessing.ipynb."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.paths import CARDS_DIR, RAW_CARDS_DIR, STEP_02_MANIFEST
from shared.pokemon_dataset import (
    build_tokenizer_and_loaders,
    collect_valid_image_paths,
    pick_device,
    save_preprocessing_manifest,
)
from shared.step_artifacts import state_path


from kfp.dsl import component as kfp_component, Input, Output, Artifact


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


@kfp_component(packages_to_install=["torch", "torchvision", "pillow", "transformers"])
def preprocessing_component(
    manifest_output: Output[Artifact],
    state_output: Output[Artifact],
    image_dir: Input[Artifact],
    img_size: int = 512,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
) -> None:
    """Validate image metadata, build the manifest, and persist the preprocessing state."""

    source_dir = Path(image_dir.path)
    if not source_dir.exists() or not any(source_dir.rglob("*.png")):
        source_dir = CARDS_DIR if CARDS_DIR.exists() else RAW_CARDS_DIR

    device = pick_device()
    print(f"Using device: {device}")
    image_paths = collect_valid_image_paths(source_dir)
    print(f"Nombre d'images valides trouvees : {len(image_paths)}")

    train_dataset, val_dataset, train_loader, val_loader, tokenizer = build_tokenizer_and_loaders(
        image_dir=source_dir,
        batch_size=batch_size,
        img_size=img_size,
        val_fraction=val_fraction,
        split_seed=split_seed,
        num_workers=num_workers,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

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

    payload: dict[str, Any] = {
        "cards_dir": str(source_dir.resolve()),
        "manifest_path": str(manifest_file.resolve()),
        "n_images": len(image_paths),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "img_size": img_size,
        "batch_size": batch_size,
    }
    _write_json(state_output_path, {"step": "02_preprocessing", **payload})
    print("Step output saved for 02_preprocessing")
    return payload
