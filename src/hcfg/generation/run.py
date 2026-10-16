"""Generate the evaluation image set of one run: every prompt key × seed × N images.

Layout::

    <run_dir>/<samples_name>/<key>/seed<s>/<i>.png
    <run_dir>/<samples_name>/manifest.json   # prompts, settings, per-image seeds
"""

from __future__ import annotations

from pathlib import Path

import torch

from hcfg.config import Config
from hcfg.data.prompts import load_prompt_set
from hcfg.generation.generate import generate_images
from hcfg.models.finetuners import expand_ti_prompt, load_finetuner
from hcfg.models.sdxl import load_sdxl
from hcfg.utils.device import configure_backends, resolve_device
from hcfg.utils.io import save_json
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)


def generate_for_run(
    config: Config,
    *,
    checkpoint: str | Path | None,
    samples_name: str = "samples",
    keys: list[str] | None = None,
) -> Path:
    """``checkpoint=None`` samples the Base model (control)."""
    configure_backends(allow_tf32=config.perf.allow_tf32, cudnn_benchmark=False)
    device = resolve_device(config.model.device)
    model = load_sdxl(config.model, device=device)
    meta: dict = {}
    if checkpoint is not None:
        meta = load_finetuner(model, checkpoint)
        logger.info("Loaded fine-tuner %s from %s", meta.get("finetuner"), checkpoint)
    prompts = load_prompt_set(config.prompts)
    gcfg = config.generate
    out_root = config.run_dir / samples_name
    pipe = model.pipeline()

    manifest = {
        "run": config.name,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "finetuner": meta.get("finetuner", "base"),
        "prompt_set": prompts.name,
        "uncond": config.model.uncond,
        "num_inference_steps": gcfg.num_inference_steps,
        "guidance_scale": gcfg.guidance_scale,
        "guidance_rescale": gcfg.guidance_rescale,
        "scheduler": type(model.sample_scheduler).__name__,
        "resolution": model.resolution,
        "prompts": {},
    }
    for key in keys or gcfg.prompts:
        prompt = prompts.prompt(key)
        model_prompt = expand_ti_prompt(prompt, meta) if meta else prompt
        entry = {"prompt": prompt, "model_prompt": model_prompt, "records": []}
        for seed in gcfg.seeds:
            logger.info("[%s] seed %d: %d images of %r", key, seed, gcfg.images_per_prompt, prompt)
            entry["records"] += generate_images(
                model,
                model_prompt,
                num_images=gcfg.images_per_prompt,
                seed=seed,
                out_dir=out_root / key / f"seed{seed}",
                num_inference_steps=gcfg.num_inference_steps,
                guidance_scale=gcfg.guidance_scale,
                guidance_rescale=gcfg.guidance_rescale,
                batch_size=gcfg.batch_size,
                pipe=pipe,
            )
        for record in entry["records"]:
            record["path"] = str(Path(record["path"]).relative_to(out_root))
        manifest["prompts"][key] = entry
        save_json(manifest, out_root / "manifest.json")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_root
