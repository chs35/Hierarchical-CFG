"""Image I/O and the fixed training-image set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hcfg.config import DataConfig, resolve_path

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images under {directory}")
    return paths


def training_image_paths(config: DataConfig) -> list[Path]:
    """The fixed training set; the count must equal ``num_images`` (same 10 for every fine-tuner)."""
    if not config.image_dir:
        raise ValueError(
            "data.image_dir is not set. Point it at the directory holding the fixed "
            "Cityscapes training images, e.g. --set data.image_dir=/path/to/cityscapes_train10"
        )
    root = resolve_path(config.image_dir)
    if config.image_list:
        lines = resolve_path(config.image_list).read_text(encoding="utf-8").splitlines()
        paths = [root / line.strip() for line in lines if line.strip() and not line.startswith("#")]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} listed images missing, e.g. {missing[:3]}")
    else:
        paths = list_images(root)
    if len(paths) != config.num_images:
        raise ValueError(f"Expected {config.num_images} training images, found {len(paths)} under {root}")
    return paths


def resize_short_side(img: Image.Image, size: int) -> Image.Image:
    """Aspect-preserving resize; both sides rounded to multiples of 8 (VAE grid)."""
    width, height = img.size
    scale = size / min(width, height)
    new_w = max(size, int(round(width * scale / 8)) * 8)
    new_h = max(size, int(round(height * scale / 8)) * 8)
    return img.resize((new_w, new_h), Image.BICUBIC)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> ``[3,H,W]`` in ``[-1,1]``."""
    array = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_image(path: str | Path, short_side: int) -> torch.Tensor:
    with Image.open(path) as img:
        return to_tensor(resize_short_side(img.convert("RGB"), short_side))


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().float().cpu().permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
    return Image.fromarray(array.numpy().astype(np.uint8))
