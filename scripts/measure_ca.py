"""Measure style CA per attention head and write the CA-LoRA selection JSON.

    python scripts/measure_ca.py --config configs/ca_lora_style.yaml --set data.image_dir=/data/cityscapes_train10

Writes ``finetuner.selection`` (default runs/ca_selection/selection_style_t81.json)
plus the full per-head CA vector next to it.
"""

import torch

from hcfg.cli import base_parser, load_from_args
from hcfg.ca.measure import measure_ca, select_top_heads, unit_names
from hcfg.config import resolve_path
from hcfg.data.images import training_image_paths
from hcfg.data.latent_cache import LatentCache
from hcfg.data.prompts import load_prompt_set
from hcfg.models.heads import enumerate_projections
from hcfg.models.sdxl import load_sdxl
from hcfg.utils.device import autocast_dtype, configure_backends, resolve_device
from hcfg.utils.io import load_json, save_json


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--weight-dtype", default="float32", help="UNet weights during measurement (grads share it)")
    args = parser.parse_args()
    config = load_from_args(args, "measure_ca")
    ca = config.finetuner.ca
    configure_backends(allow_tf32=config.perf.allow_tf32)
    device = resolve_device(config.model.device)
    model = load_sdxl(config.model, device=device, weight_dtype=args.weight_dtype)
    amp = autocast_dtype(config.perf.mixed_precision, device)

    cache = LatentCache.build(model, training_image_paths(config.data), hflip=False)
    gen = torch.Generator()
    latents = [cache.crop(lat, "center", gen) for lat in cache.latents]
    prompt = load_prompt_set(config.prompts).domain
    templates = load_json(resolve_path(ca.augmentation_file))["axes"][ca.axis]
    cond = model.encode_prompts([prompt])
    aug_conds = [model.encode_prompts([t]) for t in templates]
    model.vae.to("cpu")

    projections = enumerate_projections(model.unet)
    scores = measure_ca(
        model, projections, latents, cond, aug_conds,
        timestep=ca.timestep, num_samples=ca.num_samples, augs_per_sample=ca.augs_per_sample,
        seed=ca.seed, amp=amp,
    )
    meta = {
        "axis": ca.axis, "timestep": ca.timestep, "num_samples": ca.num_samples,
        "augs_per_sample": ca.augs_per_sample, "prompt": prompt, "templates": templates,
        "weight_dtype": args.weight_dtype, "autocast": str(amp),
    }
    selection = select_top_heads(projections, scores, fraction=ca.fraction, meta=meta)
    out = resolve_path(config.finetuner.selection)
    save_json(selection, out)
    save_json({"names": unit_names(projections), "ca": scores.tolist(), "meta": meta}, out.with_name(out.stem + "_all_scores.json"))
    print(f"Selected {selection['num_projections']} / {selection['units_total']} heads -> {out}")


if __name__ == "__main__":
    main()
