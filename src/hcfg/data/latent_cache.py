"""Latent cache with random crop / flip done in latent space.

docs/REPRODUCE.md, performance notes: the reference repo re-encoded a random 1024px crop with an
fp32 VAE on every step because ``crop: random`` disabled its latent cache. Here
each image is encoded **once** at full width (short side = resolution, so a
2048x1024 Cityscapes frame -> a 128x256 latent) and cropping happens on the
cached latent: a 1024px crop == a 128-latent crop, at an 8-pixel-aligned offset.

Horizontal flip is *not* applied to latents (``VAE(flip(x)) != flip(VAE(x))``);
the flipped image is encoded separately and cached too.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from hcfg.data.images import load_image
from hcfg.models.sdxl import SDXL
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)


class LatentCache:
    def __init__(self, latents: list[torch.Tensor], flipped: list[torch.Tensor] | None, crop_size: int):
        """``latents[i]``: ``[C, h, w_i]`` fp32 on the training device."""
        self.latents = latents
        self.flipped = flipped
        self.crop_size = crop_size
        for lat in latents:
            if lat.shape[-2] < crop_size or lat.shape[-1] < crop_size:
                raise ValueError(f"Latent {tuple(lat.shape)} smaller than crop {crop_size}")

    def __len__(self) -> int:
        return len(self.latents)

    @classmethod
    def build(cls, model: SDXL, paths: Sequence[str | Path], *, hflip: bool) -> LatentCache:
        latents, flipped = [], []
        for path in paths:
            image = load_image(path, model.resolution).unsqueeze(0)
            latents.append(model.encode_images(image)[0])
            if hflip:
                flipped.append(model.encode_images(image.flip(-1))[0])
        logger.info(
            "Cached %d latents (%s)%s", len(latents), tuple(latents[0].shape), " + flipped" if hflip else ""
        )
        return cls(latents, flipped if hflip else None, model.latent_size)

    def sample(self, batch_size: int, *, crop: str, hflip: bool, generator: torch.Generator) -> torch.Tensor:
        """``[B, C, s, s]`` batch: random image, random (or center) crop, random flip."""
        out = []
        for _ in range(batch_size):
            index = int(torch.randint(len(self.latents), (1,), generator=generator))
            flip = hflip and self.flipped is not None and bool(torch.randint(2, (1,), generator=generator))
            lat = (self.flipped if flip else self.latents)[index]
            out.append(self.crop(lat, crop, generator))
        return torch.stack(out)

    def crop(self, lat: torch.Tensor, mode: str, generator: torch.Generator) -> torch.Tensor:
        s = self.crop_size
        h, w = lat.shape[-2:]
        top = (h - s) // 2  # vertical window stays centred (landscape frames: no slack anyway)
        if mode == "center":
            left = (w - s) // 2
        else:
            left = int(torch.randint(w - s + 1, (1,), generator=generator))
        return lat[:, top : top + s, left : left + s]
