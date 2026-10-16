"""Deterministic seeding helpers."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def make_generator(*parts: object, device: str | torch.device = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(derive_seed(*parts))
