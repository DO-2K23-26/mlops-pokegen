"""Dataset cartes Pokémon + lecture du chunk PNG pokemon_metadata (notebooks MLOps)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import CLIPTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    stats = [(i, *torch.cuda.mem_get_info(i)) for i in range(torch.cuda.device_count())]
    for i, free, total in stats:
        print(f"GPU {i}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    best = max(stats, key=lambda x: x[1])[0]
    d = torch.device(f"cuda:{best}")
    print(f"→ Using {d}")
    return d


def get_metadata_from_png(path: Path) -> Optional[Dict]:
    """Lit le chunk pokemon_metadata depuis les métadonnées PNG (tEXt)."""
    with Image.open(path) as img:
        meta_str = img.info.get("pokemon_metadata")
    if meta_str is None:
        return None
    try:
        return json.loads(meta_str)
    except json.JSONDecodeError:
        return None


def metadata_to_conditioning(meta: Dict) -> str:
    """Sérialise les metadata (dict pokemon_metadata) en JSON pour le conditionnement CLIP."""
    return json.dumps(meta, sort_keys=True, ensure_ascii=False)


def collect_valid_image_paths(image_dir: Path) -> List[Path]:
    all_paths = sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    image_paths: List[Path] = []
    for p in all_paths:
        try:
            meta = get_metadata_from_png(p)
            if meta is None:
                continue
            with Image.open(p) as img:
                img.verify()
            image_paths.append(p)
        except Exception:
            continue
    return image_paths


class PokemonCardDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        transform,
        tokenizer: CLIPTokenizer,
        max_length: int = 77,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        meta = get_metadata_from_png(path)
        conditioning = metadata_to_conditioning(meta)
        img = Image.open(path).convert("RGB")
        pixel_values = self.transform(img)
        tokens = self.tokenizer(
            conditioning,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }


def build_tokenizer_and_loaders(
    image_dir: Path,
    batch_size: int = 4,
    img_size: int = 512,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    split_seed: int = 42,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader, CLIPTokenizer]:
    """Construit dataset / loaders CLIP à partir d'un répertoire d'images (métadonnées PNG)."""
    image_paths = collect_valid_image_paths(image_dir)
    print(f"Found {len(image_paths)} valid image(s) with pokemon_metadata under {image_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = PokemonCardDataset(image_paths, transform, tokenizer)
    val_size = max(1, int(val_fraction * len(dataset)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Pas assez d'images pour train/val.")
    g = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset, train_loader, val_loader, tokenizer


def save_preprocessing_manifest(
    manifest_path: Path,
    image_dir: Path,
    image_paths: List[Path],
    train_size: int,
    val_size: int,
    img_size: int,
    batch_size: int,
    split_seed: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_dir": str(image_dir.resolve()),
        "image_paths": [str(p.resolve()) for p in image_paths],
        "train_size": train_size,
        "val_size": val_size,
        "img_size": img_size,
        "batch_size": batch_size,
        "split_seed": split_seed,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Manifest enregistré: {manifest_path}")


def load_preprocessing_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_loaders_from_manifest(manifest_path: Path):
    """Reconstruit train/val loaders à partir du manifest (étapes train/eval séparées)."""
    m = load_preprocessing_manifest(manifest_path)
    image_paths = [Path(p) for p in m["image_paths"]]
    img_size = m["img_size"]
    batch_size = m["batch_size"]
    split_seed = m["split_seed"]
    train_size = m["train_size"]
    val_size = m["val_size"]

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = PokemonCardDataset(image_paths, transform, tokenizer)
    g = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataset, val_dataset, train_loader, val_loader, tokenizer
