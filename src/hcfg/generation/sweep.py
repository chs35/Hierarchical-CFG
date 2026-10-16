"""The (w1, w2) grid of one fine-tuner with hierarchical CFG (report §4.2).

Layout (``<run_dir>/hcfg/<sweep>/``, sweep = variant [+ _split [+ _a<a>]] [+ _schedule])::

    sweep.json                                  # settings (a re-run must match), cells, counts
    w1_<a>_w2_<b>/manifest.json                 # same format as generation/run.py -> evaluate_samples
    w1_<a>_w2_<b>/<key>/seed<s>/<i>.png

Cells are the w1 × w2 grid, or ``hcfg.cells``; with the Diff_Null split each cell also
has a λ (``w1_<a>_w2_<b>_lam_<λ>``), from ``hcfg.split_lambda`` or the cell's third entry.

Image i of (key, seed) starts from the same z_T as ``generate_images`` (``image_seed(s, i)``),
so every cell and the standard-CFG samples share their initial noise. Images are
written atomically and a re-run skips the ones that exist, so an interrupted sweep
is resumed by running the same command again. Persistence and disk space are checked
before the model is loaded.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import torch

from hcfg.config import Config, PromptConfig, resolve_path, save_config
from hcfg.data.images import to_pil
from hcfg.data.prompts import load_prompt_set
from hcfg.diagnostics.run import check_base_prompts, read_checkpoint_meta
from hcfg.diagnostics.storage import GB, check_disk_space, ensure_persistent
from hcfg.generation.generate import decode_latents, image_seed
from hcfg.generation.hierarchical import HCFGConds, WeightSchedule, hcfg_sample
from hcfg.models.finetuners import expand_ti_prompt, load_finetuner
from hcfg.models.lora import merge_lora
from hcfg.models.sdxl import load_sdxl
from hcfg.utils.device import configure_backends, resolve_device
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

FINAL_CHECKPOINT = "finetuner-final.safetensors"
FIXED_BYTES = 16 * 1024**2
SPLIT_KEYS = ("split_lambda", "split_par", "cells")


def sweep_name(config: Config) -> str:
    h = config.hcfg
    name = h.variant
    if h.split:
        name += "_split" if h.split_par is None else f"_split_a{h.split_par:g}"
    return name if h.schedule == "constant" else f"{name}_{h.schedule}"


def cell_name(w1: float, w2: float, lam: float | None = None) -> str:
    name = f"w1_{w1:g}_w2_{w2:g}"
    return name if lam is None else f"{name}_lam_{lam:g}"


def sweep_cells(config: Config) -> list[tuple[float, float, float | None]]:
    """(w1, w2, λ) of every cell; λ is None without the Diff_Null split."""
    h = config.hcfg
    if h.cells and len(h.cells[0]) == 3:
        return [(a, b, lam) for a, b, lam in h.cells]
    pairs = [(a, b) for a, b in h.cells] if h.cells else [(a, b) for a in h.w1 for b in h.w2]
    return [(a, b, lam) for a, b in pairs for lam in (h.split_lambda or [None])]


def sweep_root(config: Config, out_name: str | None = None) -> Path:
    return config.run_dir / "hcfg" / (out_name or sweep_name(config))


def image_path(root: Path, w1: float, w2: float, key: str, seed: int, index: int, lam: float | None = None) -> Path:
    return root / cell_name(w1, w2, lam) / key / f"seed{seed}" / f"{index:04d}.png"


def planned_images(config: Config, root: Path) -> list[Path]:
    h = config.hcfg
    return [
        image_path(root, a, b, key, seed, i, lam)
        for a, b, lam in sweep_cells(config) for key in h.keys for seed in h.seeds for i in range(h.images_per_prompt)
    ]


def estimate_sweep_bytes(config: Config, num_images: int) -> dict[str, int]:
    return {"images": int(num_images * config.hcfg.image_mb_estimate * 1024**2), "fixed": FIXED_BYTES}


def settings(config: Config, checkpoint: Path) -> dict:
    """Everything that changes the images; a resumed sweep must match exactly."""
    asdict = dataclasses.asdict
    h = {k: v for k, v in asdict(config.hcfg).items()
         if k not in ("batch_size", "image_mb_estimate") and not (k in SPLIT_KEYS and v is None)}  # older sweep.json
    gen = config.generate
    return {
        "checkpoint": str(checkpoint),
        "hcfg": h,
        "num_inference_steps": gen.num_inference_steps,
        "guidance_rescale": gen.guidance_rescale,
        "model": {k: v for k, v in asdict(config.model).items() if k != "device"},
        "prompts": asdict(config.prompts),
    }


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def preflight(config: Config, checkpoint: Path, root: Path) -> tuple[dict, int]:
    """Persistence, settings match and disk space; nothing is written or loaded before."""
    ensure_persistent(root, config.storage)
    ensure_persistent(config.run_dir, config.storage)
    wanted = settings(config, checkpoint)
    existing = root / "sweep.json"
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))["settings"]
        if previous != wanted:
            diff = sorted(k for k in wanted if previous.get(k) != wanted[k])
            raise ValueError(f"{root} holds a sweep with different settings ({diff}); use another --out-name")
    missing = sum(not p.exists() for p in planned_images(config, root))
    info = check_disk_space(root, estimate_sweep_bytes(config, missing), config.storage)
    return info, missing


def run_sweep(config: Config, *, checkpoint: str | Path, out_name: str | None = None) -> Path:
    h, gen = config.hcfg, config.generate
    checkpoint = resolve_path(checkpoint)
    root = sweep_root(config, out_name)
    storage_info, missing = preflight(config, checkpoint, root)
    total = len(planned_images(config, root))
    logger.info("Sweep %s: %d images planned, %d to generate (%.1f GB allowance)",
                root, total, missing, missing * h.image_mb_estimate / 1024)

    ckpt_meta = read_checkpoint_meta(checkpoint)
    prompts = load_prompt_set(config.prompts)
    base_prompts = load_prompt_set(PromptConfig(file=config.prompts.file, set=h.base_prompt_set or config.prompts.set))
    base_texts = {"domain": base_prompts.domain, **{k: base_prompts.refine(k) for k in h.keys}}
    need_base = h.variant == "alt_refine" or h.split  # the split's r = ε(FT,c) − ε(Base,c)
    if need_base:
        check_base_prompts(ckpt_meta, base_texts)
    for key in h.keys:
        prompts.refine(key)  # unknown key -> KeyError before loading

    root.mkdir(parents=True, exist_ok=True)
    save_config(config, root / "config.yaml")
    cells = [cell_name(a, b, lam) for a, b, lam in sweep_cells(config)]
    _write_json(root / "sweep.json", {"settings": settings(config, checkpoint), "cells": cells,
                                      "images_total": total, "storage": storage_info})
    if missing == 0:
        logger.info("Nothing to generate")
        return root

    configure_backends(allow_tf32=config.perf.allow_tf32, cudnn_benchmark=False)
    model = load_sdxl(config.model, device=resolve_device(config.model.device))
    model.unet.eval()
    base = {k: model.encode_prompts([v]) for k, v in base_texts.items()} if need_base else {}
    meta = load_finetuner(model, checkpoint)
    if h.merge_lora:
        merge_lora(model.unet)
    expand = lambda p: expand_ti_prompt(p, meta)  # noqa: E731
    null, domain = model.null_cond(1), model.encode_prompts([expand(prompts.domain)])
    conds = {
        key: HCFGConds(null=null, domain=domain, refine=model.encode_prompts([expand(prompts.refine(key))]),
                       base_domain=base.get("domain"), base_refine=base.get(key))
        for key in h.keys
    }

    done, started = 0, time.time()
    for a, b, lam in sweep_cells(config):
        cell = root / cell_name(a, b, lam)
        manifest = {
            "run": config.name, "checkpoint": str(checkpoint), "finetuner": meta.get("finetuner", "base"),
            "variant": h.variant, "w1": a, "w2": b, "schedule": h.schedule, "schedule_end": h.schedule_end,
            "reduce_equal_weights": h.reduce_equal_weights, "merge_lora": h.merge_lora,
            "prompt_set": prompts.name, "uncond": config.model.uncond,
            "num_inference_steps": gen.num_inference_steps, "guidance_rescale": gen.guidance_rescale,
            "scheduler": type(model.sample_scheduler).__name__, "resolution": model.resolution, "prompts": {},
        }
        if lam is not None:  # a = null: the variant's own (basic 1, alt_domain w2)
            manifest["split"] = {"lambda": lam, "par": h.split_par, "r": "diff_cond"}
        w1s, w2s = WeightSchedule(a, h.schedule, h.schedule_end), WeightSchedule(b, h.schedule, h.schedule_end)
        for key in h.keys:
            entry = {"prompt": prompts.refine(key), "model_prompt": expand(prompts.refine(key)), "records": []}
            for seed in h.seeds:
                for start in range(0, h.images_per_prompt, h.batch_size):
                    indices = list(range(start, min(start + h.batch_size, h.images_per_prompt)))
                    paths = [image_path(root, a, b, key, seed, i, lam) for i in indices]
                    seeds = [image_seed(seed, i) for i in indices]
                    todo = [not p.exists() for p in paths]
                    if any(todo):  # the whole batch: the same bf16 batch layout as an uninterrupted run
                        latents = hcfg_sample(
                            model, conds[key], seeds, variant=h.variant, w1=w1s, w2=w2s,
                            num_inference_steps=gen.num_inference_steps, guidance_rescale=gen.guidance_rescale,
                            reduce_equal_weights=h.reduce_equal_weights, split_lambda=lam, split_par=h.split_par,
                        )
                        paths[0].parent.mkdir(parents=True, exist_ok=True)
                        for path, image, need in zip(paths, decode_latents(model, latents), todo):
                            if need:
                                tmp = path.with_name(path.name + ".partial")
                                to_pil(image).save(tmp, format="PNG")
                                os.replace(tmp, path)
                        done += sum(todo)
                        rate = (time.time() - started) / done
                        logger.info(
                            "%s %s seed%d [%d-%d] | %d/%d new images | %.2f s/img | ETA %.1f min",
                            cell.name, key, seed, indices[0], indices[-1], done, missing, rate,
                            rate * (missing - done) / 60,
                        )
                    for index, s, path in zip(indices, seeds, paths):
                        entry["records"].append(
                            {"path": path.relative_to(cell).as_posix(), "index": index, "seed": seed, "image_seed": s}
                        )
            manifest["prompts"][key] = entry
            _write_json(cell / "manifest.json", manifest)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Sweep done: %s (%d new images, %.1f min)", root, done, (time.time() - started) / 60)
    return root


def check_only(configs: list[Config], out_name: str | None = None) -> dict:
    """Disk/persistence check of several sweeps together (``scripts/sweep.py --check-disk``)."""
    total_missing, report = 0, {}
    for config in configs:
        root = sweep_root(config, out_name)
        ensure_persistent(root, config.storage)
        missing = sum(not p.exists() for p in planned_images(config, root))
        report[config.name] = missing
        total_missing += missing
    estimate = estimate_sweep_bytes(configs[0], total_missing)
    info = check_disk_space(configs[0].run_dir.parent, estimate, configs[0].storage)
    info["images_to_generate"] = report
    logger.info("%d images to generate: %.1f GB allowance", total_missing, estimate["images"] / GB)
    return info
