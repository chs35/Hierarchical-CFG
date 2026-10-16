"""Standard-CFG image generation for trained fine-tuners (and the Base model).

This is *not* the hierarchical-CFG sampler (out of scope for this stage); it is
the diffusers SDXL pipeline with the conditioning passed in explicitly so that

* the negative branch is exactly :meth:`SDXL.null_cond` (the run's ∅ definition),
* the TI placeholder is expanded to the registered token,
* image ``i`` of a (prompt, seed) pair uses its own generator, so it is the
  same image whatever the batch size.

Guidance follows diffusers: ε_u + w(ε_c − ε_u), ``guidance_rescale`` off by default.
Latents are decoded outside the pipeline with the fp32 VAE (bf16 UNet latents
would otherwise hit an fp32 VAE without autocast).
"""

from __future__ import annotations

from pathlib import Path

import torch

from hcfg.data.images import to_pil
from hcfg.models.sdxl import SDXL, Cond
from hcfg.utils.logging import get_logger
from hcfg.utils.seed import derive_seed

logger = get_logger(__name__)


def image_seed(seed: int, index: int) -> int:
    return derive_seed("image", seed, index)


@torch.no_grad()
def decode_latents(model: SDXL, latents: torch.Tensor) -> torch.Tensor:
    images = model.vae.decode(latents.to(model.vae_dtype) / model.scaling_factor).sample
    return images.float().clamp(-1, 1)


@torch.no_grad()
def generate_latents(
    model: SDXL,
    cond: Cond,
    null: Cond,
    image_seeds: list[int],
    *,
    num_inference_steps: int = 25,
    guidance_scale: float = 5.0,
    guidance_rescale: float = 0.0,
    pipe=None,
) -> torch.Tensor:
    """Final latents of one image per entry of ``image_seeds`` (1-row ``cond``/``null``)."""
    pipe = pipe or model.pipeline()
    count = len(image_seeds)
    generators = [torch.Generator(device=model.device).manual_seed(s) for s in image_seeds]
    return pipe(
        prompt_embeds=cond.prompt_embeds.repeat(count, 1, 1),
        pooled_prompt_embeds=cond.pooled.repeat(count, 1),
        negative_prompt_embeds=null.prompt_embeds.repeat(count, 1, 1),
        negative_pooled_prompt_embeds=null.pooled.repeat(count, 1),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        generator=generators,
        height=model.resolution,
        width=model.resolution,
        output_type="latent",
    ).images


@torch.no_grad()
def generate_images(
    model: SDXL,
    prompt: str,
    *,
    num_images: int,
    seed: int,
    out_dir: str | Path,
    prefix: str = "",
    num_inference_steps: int = 25,
    guidance_scale: float = 5.0,
    guidance_rescale: float = 0.0,
    batch_size: int = 4,
    pipe=None,
) -> list[dict]:
    """Generate ``num_images`` for one prompt/seed; returns manifest records."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = pipe or model.pipeline()
    cond = model.encode_prompts([prompt])
    null = model.null_cond(1)
    was_training = model.unet.training
    model.unet.eval()
    records = []
    try:
        for start in range(0, num_images, batch_size):
            indices = list(range(start, min(start + batch_size, num_images)))
            seeds = [image_seed(seed, i) for i in indices]
            latents = generate_latents(
                model,
                cond,
                null,
                seeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                pipe=pipe,
            )
            images = decode_latents(model, latents)
            for index, image_seed_value, image in zip(indices, seeds, images):
                path = out_dir / f"{prefix}{index:04d}.png"
                to_pil(image).save(path)
                records.append({"path": str(path), "index": index, "seed": seed, "image_seed": image_seed_value})
    finally:
        model.unet.train(was_training)
    return records
