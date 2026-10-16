"""The q(z_t|x0) source: held-out Cityscapes images, forward-noised at the sampler's t.

The 10 training images are excluded on purpose (10k steps on 10 images: the
fine-tuners have likely memorised them). The held-out images are a seeded random
draw from ``data.image_dir`` minus the training list, saved next to the results.
Noise depends only on (seed, image), so every fine-tuner sees identical z_t.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from hcfg.config import DataConfig, resolve_path
from hcfg.data.images import list_images, training_image_paths
from hcfg.data.latent_cache import LatentCache
from hcfg.models.sdxl import SDXL
from hcfg.utils.seed import derive_seed, make_generator


def select_heldout_images(data: DataConfig, *, num: int, seed: int, pinned: str | None = None) -> list[Path]:
    """``num`` images of ``data.image_dir`` outside the training list (sorted, deterministic)."""
    if not data.image_list:
        raise ValueError(
            "Held-out selection needs data.image_list (the training images to exclude), "
            "e.g. --set data.image_list=configs/data/train10.txt"
        )
    root = resolve_path(data.image_dir)
    train = {p.resolve() for p in training_image_paths(data)}
    if pinned:
        lines = resolve_path(pinned).read_text(encoding="utf-8").splitlines()
        paths = [root / line.strip() for line in lines if line.strip() and not line.startswith("#")]
        overlap = [str(p) for p in paths if p.resolve() in train]
        if overlap:
            raise ValueError(f"Pinned held-out list contains training images: {overlap[:3]}")
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} pinned held-out images missing, e.g. {missing[:3]}")
        return paths
    candidates = [p for p in list_images(root) if p.resolve() not in train]
    if len(candidates) < num:
        raise ValueError(f"Only {len(candidates)} held-out candidates under {root}, need {num}")
    rng = random.Random(derive_seed("heldout", seed))
    return sorted(rng.sample(candidates, num))


def relative_names(paths: list[Path], image_dir: str | Path) -> list[str]:
    root = resolve_path(image_dir)
    return [p.relative_to(root).as_posix() for p in paths]


def heldout_latents(model: SDXL, paths: list[Path]) -> list[torch.Tensor]:
    """fp32 ``[C, s, s]`` centre crops of the VAE latents (short side = resolution)."""
    cache = LatentCache.build(model, paths, hflip=False)
    gen = torch.Generator()
    return [cache.crop(lat, "center", gen) for lat in cache.latents]


def forward_noise(seed: int, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """ε of image ``name`` for noise seed ``seed`` (fp32, CPU), shared by all t."""
    return torch.randn(shape, generator=make_generator("diag_forward", seed, name))
