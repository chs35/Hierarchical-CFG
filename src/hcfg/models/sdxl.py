"""SDXL components, prompt encoding, the null condition ∅ and ε-prediction.

Everything that touches the conditioning goes through :class:`SDXL` so that the
three rules of docs/Implementation detail.md §5 hold by construction:

* ∅ has exactly one definition per run (``model.uncond``): ``zeros`` (SDXL's
  ``force_zeros_for_empty_prompt``, default) or ``empty`` (encoding of ``""``).
  Condition dropout during training and the negative branch at generation both
  call :meth:`SDXL.null_cond`.
* ``added_cond_kwargs`` (pooled ``text_embeds`` + ``time_ids``) always carry
  exactly the batch size of the latents; :meth:`SDXL.predict` asserts it instead
  of relying on broadcasting.
* ``time_ids`` are ``[H, W, 0, 0, H, W]`` at the model resolution, which is what
  the diffusers pipeline uses at inference with default arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from hcfg.config import ModelConfig
from hcfg.utils.device import effective_weight_dtype, resolve_device, resolve_dtype
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Cond:
    """SDXL text conditioning for a batch: ``[B, L, 2048]`` + pooled ``[B, 1280]``."""

    prompt_embeds: torch.Tensor
    pooled: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.prompt_embeds.shape[0]

    def to(self, device=None, dtype=None) -> Cond:
        return Cond(self.prompt_embeds.to(device=device, dtype=dtype), self.pooled.to(device=device, dtype=dtype))

    def repeat(self, n: int) -> Cond:
        return Cond(self.prompt_embeds.repeat(n, 1, 1), self.pooled.repeat(n, 1))

    def detach(self) -> Cond:
        return Cond(self.prompt_embeds.detach(), self.pooled.detach())

    @staticmethod
    def cat(conds: list[Cond]) -> Cond:
        return Cond(torch.cat([c.prompt_embeds for c in conds]), torch.cat([c.pooled for c in conds]))

    @staticmethod
    def where(mask: torch.Tensor, a: Cond, b: Cond) -> Cond:
        """Per-sample select: ``mask[i]`` True -> ``a[i]`` else ``b[i]``."""
        m3 = mask.view(-1, 1, 1)
        m2 = mask.view(-1, 1)
        return Cond(torch.where(m3, a.prompt_embeds, b.prompt_embeds), torch.where(m2, a.pooled, b.pooled))


class SDXL:
    def __init__(
        self,
        *,
        unet: nn.Module,
        vae: nn.Module,
        text_encoders: list[nn.Module],
        tokenizers: list[Any],
        noise_scheduler: Any,
        sample_scheduler: Any,
        device: torch.device,
        weight_dtype: torch.dtype,
        vae_dtype: torch.dtype,
        resolution: int,
        uncond: str = "zeros",
    ):
        self.unet = unet
        self.unet_fn = unet
        """What :meth:`predict` calls; the trainer may swap in ``torch.compile(unet)``."""
        self.vae = vae
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.noise_scheduler = noise_scheduler
        self.sample_scheduler = sample_scheduler
        self.device = device
        self.weight_dtype = weight_dtype
        self.vae_dtype = vae_dtype
        self.resolution = resolution
        self.uncond = uncond
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.scaling_factor = float(vae.config.scaling_factor)
        self.num_train_timesteps = int(noise_scheduler.config.num_train_timesteps)
        self.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        if str(getattr(noise_scheduler.config, "prediction_type", "epsilon")) != "epsilon":
            raise ValueError("This project assumes ε-prediction (SDXL base)")
        for module in (vae, *text_encoders, unet):
            module.requires_grad_(False)
        vae.eval()
        for encoder in text_encoders:
            encoder.eval()

    # -- shapes ------------------------------------------------------------------------

    @property
    def latent_size(self) -> int:
        return self.resolution // self.vae_scale_factor

    @property
    def latent_channels(self) -> int:
        return int(self.unet.config.in_channels)

    def time_ids(self, batch: int) -> torch.Tensor:
        r = self.resolution
        ids = torch.tensor([r, r, 0, 0, r, r], dtype=self.weight_dtype, device=self.device)
        return ids.unsqueeze(0).repeat(batch, 1)

    # -- encoding ----------------------------------------------------------------------

    def encode_prompts(self, prompts: list[str], *, grad: bool = False) -> Cond:
        """SDXL dual-encoder conditioning (penultimate hidden states, pooled from encoder 2).

        Same recipe as ``StableDiffusionXLPipeline.encode_prompt`` with ``clip_skip=None``.
        ``grad=True`` keeps the graph (Textual Inversion training).
        """
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            hiddens = []
            pooled = None
            for tokenizer, encoder in zip(self.tokenizers, self.text_encoders):
                ids = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(self.device)
                out = encoder(ids, output_hidden_states=True)
                pooled = out[0]  # text_embeds of CLIPTextModelWithProjection (encoder 2)
                hiddens.append(out.hidden_states[-2])
            embeds = torch.cat(hiddens, dim=-1)
        return Cond(embeds.to(self.weight_dtype), pooled.to(self.weight_dtype))

    def null_cond(self, batch: int) -> Cond:
        """∅ for ``batch`` samples — the single definition used everywhere."""
        if self.uncond == "empty":
            return self.encode_prompts([""]).repeat(batch)
        seq = self.tokenizers[0].model_max_length
        dim = sum(int(e.config.hidden_size) for e in self.text_encoders)
        pooled_dim = int(self.text_encoders[-1].config.projection_dim)
        return Cond(
            torch.zeros(batch, seq, dim, device=self.device, dtype=self.weight_dtype),
            torch.zeros(batch, pooled_dim, device=self.device, dtype=self.weight_dtype),
        )

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``[B,3,H,W]`` in ``[-1,1]`` -> scaled latents (distribution mode, deterministic), fp32."""
        images = images.to(device=self.device, dtype=self.vae_dtype)
        latents = self.vae.encode(images).latent_dist.mode() * self.scaling_factor
        return latents.float()

    # -- denoiser ----------------------------------------------------------------------

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        a = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return a.sqrt() * latents + (1 - a).sqrt() * noise

    def predict(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: Cond) -> torch.Tensor:
        batch = noisy.shape[0]
        if cond.prompt_embeds.shape[0] != batch or cond.pooled.shape[0] != batch or timesteps.shape[0] != batch:
            raise ValueError(
                f"Batch mismatch: latents {batch}, prompt_embeds {cond.prompt_embeds.shape[0]}, "
                f"pooled {cond.pooled.shape[0]}, timesteps {timesteps.shape[0]} (no implicit broadcast)"
            )
        added = {"text_embeds": cond.pooled, "time_ids": self.time_ids(batch)}
        return self.unet_fn(noisy, timesteps, encoder_hidden_states=cond.prompt_embeds, added_cond_kwargs=added).sample

    # -- pipeline ----------------------------------------------------------------------

    def pipeline(self):
        """A diffusers pipeline over *these* modules (adapters included)."""
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoders[0],
            text_encoder_2=self.text_encoders[1],
            tokenizer=self.tokenizers[0],
            tokenizer_2=self.tokenizers[1],
            unet=self.unet,
            scheduler=self.sample_scheduler,
            force_zeros_for_empty_prompt=self.uncond == "zeros",
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_sdxl(config: ModelConfig, *, device: torch.device | None = None, weight_dtype: str | None = None) -> SDXL:
    device = device or resolve_device(config.device)
    wdtype = effective_weight_dtype(weight_dtype or config.weight_dtype, device)
    vae_dtype = resolve_dtype(config.vae_dtype)
    if device.type == "cpu":
        vae_dtype = torch.float32
    if config.kind == "tiny":
        from hcfg.models.tiny import build_tiny_components

        parts = build_tiny_components()
    elif config.kind == "sdxl":
        parts = _load_pretrained(config, wdtype)
    else:
        raise ValueError(f"Unknown model.kind {config.kind!r}")

    parts["unet"].to(device=device, dtype=wdtype)
    for encoder in parts["text_encoders"]:
        encoder.to(device=device, dtype=wdtype)
    parts["vae"].to(device=device, dtype=vae_dtype)
    model = SDXL(
        **parts,
        device=device,
        weight_dtype=wdtype,
        vae_dtype=vae_dtype,
        resolution=config.resolution,
        uncond=config.uncond,
    )
    logger.info(
        "Loaded %s (%s) on %s | unet/text dtype %s | vae dtype %s | ∅=%s",
        config.kind,
        config.pretrained if config.kind == "sdxl" else "random",
        device,
        wdtype,
        vae_dtype,
        config.uncond,
    )
    return model


def _load_pretrained(config: ModelConfig, dtype: torch.dtype) -> dict:
    from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    kw = {"revision": config.revision}
    wkw = {**kw, "variant": config.variant} if config.variant else kw
    name = config.pretrained
    return {
        "unet": UNet2DConditionModel.from_pretrained(name, subfolder="unet", torch_dtype=dtype, **wkw),
        "vae": AutoencoderKL.from_pretrained(name, subfolder="vae", torch_dtype=torch.float32, **wkw),
        "text_encoders": [
            CLIPTextModel.from_pretrained(name, subfolder="text_encoder", torch_dtype=dtype, **wkw),
            CLIPTextModelWithProjection.from_pretrained(name, subfolder="text_encoder_2", torch_dtype=dtype, **wkw),
        ],
        "tokenizers": [
            CLIPTokenizer.from_pretrained(name, subfolder="tokenizer", **kw),
            CLIPTokenizer.from_pretrained(name, subfolder="tokenizer_2", **kw),
        ],
        "noise_scheduler": DDPMScheduler.from_pretrained(name, subfolder="scheduler", **kw),
        "sample_scheduler": EulerDiscreteScheduler.from_pretrained(name, subfolder="scheduler", **kw),
    }
