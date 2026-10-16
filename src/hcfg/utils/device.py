"""Device, dtype and A100 tensor-core settings.

Why this module exists (docs/REPRODUCE.md, performance notes): the reference reproduction ran
fp32 weights with ``mixed_precision: "no"`` and no TF32, so the A100 tensor cores
were idle (2.2 s/step vs ~0.36 s/step). Every entry point calls
:func:`configure_backends` and runs the UNet under :func:`autocast_context`.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch

from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(DTYPES)}") from None


def effective_weight_dtype(name: str, device: torch.device) -> torch.dtype:
    """Frozen-weight dtype; CPU falls back to fp32 (bf16 matmuls are slow/partial there)."""
    dtype = resolve_dtype(name)
    if device.type == "cpu" and dtype != torch.float32:
        logger.warning("CPU run: storing weights in float32 instead of %s", dtype)
        return torch.float32
    return dtype


def autocast_dtype(mixed_precision: str, device: torch.device) -> torch.dtype | None:
    """``None`` means no autocast. Autocast is CUDA-only in this project."""
    if device.type != "cuda" or mixed_precision == "no":
        return None
    return resolve_dtype(mixed_precision)


def autocast_context(dtype: torch.dtype | None, device: torch.device):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def configure_backends(*, allow_tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    """TF32 for any remaining fp32 matmul/conv, cuDNN autotuning for fixed shapes."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.backends.cudnn.benchmark = cudnn_benchmark
    name = torch.cuda.get_device_name()
    logger.info(
        "CUDA device %s | tf32=%s | bf16 supported=%s | cudnn.benchmark=%s",
        name,
        allow_tf32,
        torch.cuda.is_bf16_supported(),
        cudnn_benchmark,
    )


@contextlib.contextmanager
def deterministic_mode(enabled: bool = True) -> Iterator[dict]:
    """Deterministic kernels for measurements (docs/Implementation detail.md §7).

    ``warn_only=True``: an op without a deterministic kernel warns instead of
    aborting a long GPU run. Yields the state (recorded in result metadata) and
    restores the previous global flags on exit.
    """
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    if enabled:
        # cuBLAS reads this when its handle is created; set before the first matmul.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    state = {
        "enabled": enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        yield state
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]
