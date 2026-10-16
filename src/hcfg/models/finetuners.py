"""Attach a fine-tuner to an :class:`~hcfg.models.sdxl.SDXL`, for training or from a checkpoint.

=====================  ==========================  =======================  ==============
fine-tuner             trainable                   Base (for Diff_Null)     prompt set
=====================  ==========================  =======================  ==============
lora                   LoRA r=64, all attention    adapter scale 0          default
ca_lora                sliced LoRA, selected heads adapter scale 0          default
ca_lora + dropout      same, ``cond_dropout>0``    adapter scale 0          default
dreambooth_lora        LoRA r=64, all attention    adapter scale 0          dreambooth
textual_inversion      new token vectors only      same UNet (∅ has no S*)  textual_inversion
=====================  ==========================  =======================  ==============
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from hcfg.config import Config, resolve_path
from hcfg.models import lora as lora_lib
from hcfg.models import textual_inversion as ti_lib
from hcfg.models.heads import enumerate_projections, plans_from_units, plans_full
from hcfg.models.sdxl import SDXL
from hcfg.utils.io import load_json
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

LORA_KINDS = ("lora", "ca_lora", "dreambooth_lora")


def load_selection_units(path: str | Path) -> list[str]:
    """Head-unit names from a selection JSON (ours or the reference repo's)."""
    data = load_json(resolve_path(path))
    names = data["names"]
    if data.get("granularity", "head") != "head":
        logger.warning("Selection %s is not head-granular; whole matrices will be adapted", path)
    return list(names)


def attach_finetuner(model: SDXL, config: Config) -> list[nn.Parameter]:
    """Install the configured fine-tuner and return its trainable parameters."""
    ft = config.finetuner
    if ft.kind in LORA_KINDS:
        projections = enumerate_projections(model.unet, kinds=ft.lora.kinds, attn=ft.lora.attn)
        if ft.kind == "ca_lora":
            units = load_selection_units(ft.selection)
            all_projections = enumerate_projections(model.unet)
            plans = plans_from_units(all_projections, units)
            logger.info("CA-LoRA: %d head units over %d matrices from %s", len(units), len(plans), ft.selection)
        else:
            plans = plans_full(projections)
        lora_lib.attach_lora(
            model.unet, plans, rank=ft.lora.rank, alpha=ft.lora.alpha, dropout=ft.lora.dropout, init=ft.lora.init
        )
        logger.info("LoRA summary: %s", lora_lib.lora_summary(model.unet))
        return lora_lib.lora_parameters(model.unet)
    if ft.kind == "textual_inversion":
        cfg = ft.textual_inversion
        ti_lib.install_textual_inversion(
            model.tokenizers,
            model.text_encoders,
            token=cfg.token,
            num_vectors=cfg.num_vectors,
            initializer=cfg.initializer,
        )
        return ti_lib.ti_parameters(model.text_encoders)
    raise ValueError(f"Unknown fine-tuner {ft.kind!r}")


def save_finetuner(model: SDXL, config: Config, path: str | Path, *, extra: dict | None = None) -> None:
    meta = {"finetuner": config.finetuner.kind, "run": config.name, "prompt_set": config.prompts.set, **(extra or {})}
    if config.finetuner.kind in LORA_KINDS:
        lora_lib.save_lora(model.unet, path, metadata=meta)
    else:
        cfg = config.finetuner.textual_inversion
        meta.update({"placeholder": cfg.placeholder})
        ti_lib.save_textual_inversion(model.text_encoders, path, token=cfg.token, metadata=meta)


def load_finetuner(model: SDXL, path: str | Path) -> dict:
    """Install a trained fine-tuner from its checkpoint alone; returns its metadata."""
    path = resolve_path(path)
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        fmt = (handle.metadata() or {}).get("format")
    if fmt == lora_lib.FORMAT:
        lora_lib.load_lora(model.unet, path, enumerate_projections(model.unet))
        meta = lora_lib.read_lora_metadata(path)
        # Place LoRA params with the model (fp32 master weights are kept in fp32).
        for _, module in lora_lib.iter_lora(model.unet):
            module.to(model.device)
        return meta
    if fmt == ti_lib.FORMAT:
        meta = ti_lib.load_textual_inversion(model.tokenizers, model.text_encoders, path)
        for encoder in model.text_encoders:
            encoder.to(model.device)
        return meta
    raise ValueError(f"Unrecognised checkpoint format {fmt!r} in {path}")


def expand_ti_prompt(prompt: str, meta_or_config) -> str:
    """Replace the paper notation (``S*``) with the registered TI token(s)."""
    if isinstance(meta_or_config, Config):
        cfg = meta_or_config.finetuner.textual_inversion
        return ti_lib.expand_prompt(prompt, cfg.placeholder, cfg.token, cfg.num_vectors)
    meta = meta_or_config
    if "placeholder" not in meta:
        return prompt
    return ti_lib.expand_prompt(prompt, meta["placeholder"], meta["token"], int(meta["num_vectors"]))
