"""Where diagnostics may write and whether there is room — checked before any model load.

RunPod: only the network volume (``/workspace``) survives closing the pod; the
container disk (``/``, ``/tmp``, ``~``) is wiped. ``df`` on the network volume
reports the whole storage cluster (hundreds of TB), not this volume, so with a
persistent root the free space is ``storage.volume_quota_gb - usage``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hcfg.config import Config, StorageConfig, resolve_path
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

GB = 1024**3
RUNPOD_VOLUME = "/workspace"
ENV_PERSISTENT_ROOT = "HCFG_PERSISTENT_ROOT"

# Upper bounds used by the estimate (bytes).
RECORD_BASE_BYTES = 1200
"""One records.jsonl line without refine keys (~20 fields, float repr <= 24 chars)."""
RECORD_REFINE_BYTES = 250
"""Per refine key in a record (3 metrics + key)."""
TENSOR_FILE_OVERHEAD = 64 * 1024
"""safetensors header + JSON metadata of one tensor file."""
FIXED_BYTES = 16 * 1024**2
"""summary.json, plots, config, progress, held-out list, logs."""


class NotPersistentError(RuntimeError):
    """An output path would be lost when the pod is closed."""


class InsufficientDiskSpace(RuntimeError):
    """Not enough room for the estimated outputs."""


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def persistent_root(storage: StorageConfig) -> Path | None:
    """``HCFG_PERSISTENT_ROOT`` > ``storage.persistent_root`` > ``/workspace`` on RunPod > None."""
    env = os.environ.get(ENV_PERSISTENT_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    if storage.persistent_root:
        return resolve_path(storage.persistent_root).resolve()
    if os.environ.get("RUNPOD_VOLUME_ID"):
        return Path(RUNPOD_VOLUME)
    return None


def ensure_persistent(path: str | Path, storage: StorageConfig) -> Path | None:
    """Raise unless ``path`` lies under the persistent root (no-op when there is none)."""
    root = persistent_root(storage)
    if root is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise NotPersistentError(
            f"{resolved} is not under the persistent volume {root}: it would be lost when the pod "
            f"is closed. Write under {root} (output_dir) or set {ENV_PERSISTENT_ROOT}."
        )
    return root


# --------------------------------------------------------------------------------------
# Size estimate
# --------------------------------------------------------------------------------------


def latent_geometry(config: Config) -> tuple[int, int]:
    """(channels, side) of a latent, without loading the model."""
    if config.model.kind == "tiny":
        from hcfg.models.tiny import TINY_LATENT_CHANNELS, TINY_VAE_SCALE

        return TINY_LATENT_CHANNELS, config.model.resolution // TINY_VAE_SCALE
    return 4, config.model.resolution // 8  # SDXL VAE


def plan_counts(config: Config, sources: list[str]) -> dict[str, dict[str, int]]:
    """Samples, steps and units per source (the same arithmetic the run uses)."""
    d = config.diagnostics
    steps = config.generate.num_inference_steps
    counts = {}
    if "trajectory" in sources:
        per_seed_units = -(-d.images_per_seed // d.batch_size)
        counts["trajectory"] = {
            "samples": len(d.trajectory_prompts) * len(d.seeds) * d.images_per_seed,
            "steps": steps,
            "units": len(d.trajectory_prompts) * len(d.seeds) * per_seed_units,
        }
    if "forward" in sources:
        counts["forward"] = {
            "samples": len(d.forward_seeds) * d.num_heldout,
            "steps": len(d.timesteps) if d.timesteps is not None else steps,
            "units": len(d.forward_seeds) * -(-d.num_heldout // d.batch_size),
        }
    return counts


def tensor_names(config: Config) -> list[str]:
    """Saved tensor names; refine tensors expand to one per refine key."""
    names = []
    for name in config.diagnostics.save_tensors:
        if name in ("g_refine", "g_refine_base"):
            names += [f"{name}/{key}" for key in config.diagnostics.refine_keys]
        else:
            names.append(name)
    return names


def estimate_output_bytes(config: Config, sources: list[str] | None = None) -> dict[str, int]:
    """Upper-bound size of every output kind of one diagnostics run."""
    d = config.diagnostics
    sources = sources or d.sources
    counts = plan_counts(config, sources)
    channels, side = latent_geometry(config)
    per_record = RECORD_BASE_BYTES + RECORD_REFINE_BYTES * len(d.refine_keys)
    names = tensor_names(config)
    estimate = {"records": 0, "tensors": 0, "images": 0, "fixed": FIXED_BYTES}
    for source, c in counts.items():
        estimate["records"] += c["samples"] * c["steps"] * per_record
        if names:
            estimate["tensors"] += len(names) * c["samples"] * c["steps"] * channels * side * side * 2
            estimate["tensors"] += c["units"] * TENSOR_FILE_OVERHEAD
        if source == "trajectory" and d.save_images:
            png_max = int(config.model.resolution**2 * 3 * 1.05) + 4096  # incompressible RGB + header
            estimate["images"] += c["samples"] * png_max
    return estimate


# --------------------------------------------------------------------------------------
# Free space
# --------------------------------------------------------------------------------------


def nearest_existing(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    while not path.exists():
        path = path.parent
    return path


def directory_usage_bytes(path: str | Path) -> int:
    """Bytes used under ``path`` (``du`` when available, else a Python walk)."""
    path = Path(path)
    if shutil.which("du"):
        try:
            out = subprocess.run(
                ["du", "-sb", str(path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            if out.stdout.strip():
                return int(out.stdout.split()[0])
        except (OSError, ValueError):
            pass
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def free_bytes(out_dir: str | Path, storage: StorageConfig) -> tuple[int, str]:
    """Free space for ``out_dir`` and how it was measured."""
    disk_free = shutil.disk_usage(nearest_existing(out_dir)).free
    root = persistent_root(storage)
    if root is not None and storage.volume_quota_gb:
        used = directory_usage_bytes(root)
        quota_free = int(storage.volume_quota_gb * GB) - used
        method = f"quota {storage.volume_quota_gb:g} GB - usage of {root} ({used / GB:.1f} GB)"
        return min(quota_free, disk_free), method
    return disk_free, f"disk_usage({nearest_existing(out_dir)})"


def check_disk_space(out_dir: str | Path, estimate: dict[str, int], storage: StorageConfig) -> dict:
    """Raise :class:`InsufficientDiskSpace` unless free >= estimate x margin + min_free."""
    needed = int(sum(estimate.values()))
    required = int(needed * storage.disk_margin + storage.min_free_gb * GB)
    free, method = free_bytes(out_dir, storage)
    info = {"estimate": dict(estimate), "needed": needed, "required": required, "free": free, "method": method}
    if free < required:
        breakdown = ", ".join(f"{k} {v / GB:.2f} GB" for k, v in estimate.items())
        raise InsufficientDiskSpace(
            f"Need {required / GB:.2f} GB (estimate {needed / GB:.2f} GB x {storage.disk_margin:g} + "
            f"{storage.min_free_gb:g} GB reserve) but only {free / GB:.2f} GB free [{method}]. "
            f"Estimate: {breakdown}"
        )
    logger.info(
        "Disk check ok: estimate %.2f GB, required %.2f GB, free %.2f GB [%s]",
        needed / GB,
        required / GB,
        free / GB,
        method,
    )
    return info
