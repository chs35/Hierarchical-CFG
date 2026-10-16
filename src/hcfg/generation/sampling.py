"""Euler sampling with a hook at every step: standard CFG, hierarchical CFG, diagnostics.

The loop mirrors ``StableDiffusionXLPipeline.__call__`` (diffusers 0.40) so that,
with the plain two-row ε function, it reproduces :func:`generate_latents`:
initial latents from the per-image generators in ``weight_dtype`` times
``init_noise_sigma``, ``scale_model_input``, ``ε_u + w(ε_c − ε_u)`` on the raw
UNet outputs (no autocast), optional ``guidance_rescale``, ``scheduler.step``.

The z_t handed to the ε function is the *scaled* model input
``latents / sqrt(σ² + 1) = sqrt(ᾱ_t) x0 + sqrt(1 − ᾱ_t) ε`` — the same VP form
as ``SDXL.add_noise`` of the forward source — and ``t`` is the scheduler's
actual timestep (0-999), never the step index.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from hcfg.models.sdxl import SDXL, Cond
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StepInfo:
    index: int
    """Position in the sampling loop (ordering only; never reported as t)."""
    t: int
    """Actual timestep (0-999)."""
    sigma: float


EpsFn = Callable[[StepInfo, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
"""``(info, z_in [B,C,h,w], t [B]) -> (ε_u, ε_c)`` in the UNet's output dtype."""
GuidedFn = Callable[[StepInfo, torch.Tensor, torch.Tensor], torch.Tensor]
"""``(info, z_in, t [B]) -> guided ε`` in the UNet's output dtype (fed to ``scheduler.step``)."""


def make_scheduler(model: SDXL, num_inference_steps: int):
    """A private copy of the sampling scheduler (no state shared with the pipeline)."""
    scheduler = type(model.sample_scheduler).from_config(model.sample_scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=model.device)
    return scheduler


def actual_timestep(t: torch.Tensor | float) -> int:
    value = float(t)
    rounded = int(round(value))
    if abs(value - rounded) > 1e-4:
        logger.warning("Non-integral sampler timestep %.4f recorded as %d", value, rounded)
    return rounded


def sampler_timesteps(model: SDXL, num_inference_steps: int) -> list[int]:
    """The actual timesteps the sampler visits, in sampling order."""
    return [actual_timestep(t) for t in make_scheduler(model, num_inference_steps).timesteps]


def initial_latents(model: SDXL, image_seeds: list[int], scheduler) -> torch.Tensor:
    """z_T exactly as the pipeline draws it (one generator per image)."""
    from diffusers.utils.torch_utils import randn_tensor

    shape = (len(image_seeds), model.latent_channels, model.latent_size, model.latent_size)
    generators = [torch.Generator(device=model.device).manual_seed(s) for s in image_seeds]
    latents = randn_tensor(shape, generator=generators, device=model.device, dtype=model.weight_dtype)
    return latents * scheduler.init_noise_sigma


def two_row_eps_fn(model: SDXL, null: Cond, cond: Cond) -> EpsFn:
    """The pipeline's own CFG batch: ``[∅ × B, c × B]`` in one UNet call."""

    def eps_fn(info: StepInfo, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        both = Cond.cat([null.repeat(batch), cond.repeat(batch)])
        eps = model.predict(torch.cat([z, z]), torch.cat([t, t]), both)
        return eps[:batch], eps[batch:]

    return eps_fn


@torch.no_grad()
def euler_sample(model: SDXL, image_seeds: list[int], guided_fn: GuidedFn, *, num_inference_steps: int) -> torch.Tensor:
    """Final latents of Euler sampling; ``guided_fn`` returns the guided ε of every step."""
    scheduler = make_scheduler(model, num_inference_steps)
    latents = initial_latents(model, image_seeds, scheduler)
    batch = latents.shape[0]
    for index, t in enumerate(scheduler.timesteps):
        info = StepInfo(index=index, t=actual_timestep(t), sigma=float(scheduler.sigmas[index]))
        z_in = scheduler.scale_model_input(latents, t)
        guided = guided_fn(info, z_in, t.expand(batch))
        latents = scheduler.step(guided, t, latents, return_dict=False)[0]
    return latents


def cfg_combine(eps_u: torch.Tensor, eps_c: torch.Tensor, guidance_scale: float, guidance_rescale: float = 0.0) -> torch.Tensor:
    """The pipeline's guidance line, same expression and dtype (bit-identical results)."""
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

    if guidance_scale <= 1.0:  # the pipeline drops CFG entirely
        return eps_c
    guided = eps_u + guidance_scale * (eps_c - eps_u)
    if guidance_rescale > 0.0:
        guided = rescale_noise_cfg(guided, eps_c, guidance_rescale=guidance_rescale)
    return guided


@torch.no_grad()
def euler_cfg_sample(
    model: SDXL,
    image_seeds: list[int],
    eps_fn: EpsFn,
    *,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_rescale: float = 0.0,
) -> torch.Tensor:
    """Final latents of standard-CFG sampling, calling ``eps_fn`` at every step."""

    def guided_fn(info: StepInfo, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps_u, eps_c = eps_fn(info, z, t)
        return cfg_combine(eps_u, eps_c, guidance_scale, guidance_rescale)

    return euler_sample(model, image_seeds, guided_fn, num_inference_steps=num_inference_steps)
