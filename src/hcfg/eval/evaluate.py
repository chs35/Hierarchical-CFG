"""Evaluate one generated sample set (see :mod:`hcfg.generation.run` for its layout).

Per prompt key: CLIP-score against the prompt (TI/DreamBooth special tokens
stripped), CMMD to real Cityscapes images, zero-shot weather / vehicle-type
distributions, and — for keys with vehicle targets — OWLv2 type+colour rates.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from hcfg.config import Config, resolve_path
from hcfg.data.images import list_images
from hcfg.data.prompts import load_prompt_set
from hcfg.eval import metrics as M
from hcfg.utils.io import load_json, save_json
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

_SPECIAL = re.compile(r"\s*with (sks|S\*) style")


def plain_prompt(prompt: str) -> str:
    """Drop the DreamBooth/TI identifier phrase so CLIP-score compares like with like."""
    return _SPECIAL.sub("", prompt)


def evaluate_samples(
    config: Config,
    samples_dir: str | Path,
    *,
    embedder=None,
    cmmd_embedder=None,
    detector=None,
) -> dict:
    samples_dir = Path(samples_dir)
    manifest = load_json(samples_dir / "manifest.json")
    prompts = load_prompt_set(config.prompts)
    ecfg = config.eval
    if embedder is None:
        from hcfg.eval.embedders import CLIPEmbedder

        embedder = CLIPEmbedder(ecfg.clip_model, device=config.model.device, batch_size=ecfg.batch_size)
    logit_scale = getattr(embedder, "logit_scale", 100.0)

    reference = None
    if ecfg.reference_dir:
        if cmmd_embedder is None:
            from hcfg.eval.embedders import CLIPEmbedder

            cmmd_embedder = (
                embedder if ecfg.cmmd_clip_model == ecfg.clip_model
                else CLIPEmbedder(ecfg.cmmd_clip_model, device=config.model.device, batch_size=ecfg.batch_size)
            )
        ref_paths = list_images(resolve_path(ecfg.reference_dir))[: ecfg.reference_num_images]
        reference = cmmd_embedder.embed_images(ref_paths)
        logger.info("CMMD reference: %d images from %s", len(ref_paths), ecfg.reference_dir)
    else:
        logger.warning("eval.reference_dir not set: CMMD skipped")

    labels = {
        "weather": (list(M.WEATHER_LABELS), embedder.embed_texts(list(M.WEATHER_LABELS.values()))),
        "vehicle": (list(M.VEHICLE_LABELS), embedder.embed_texts(list(M.VEHICLE_LABELS.values()))),
    }
    color_labels = list(M.COLOR_LABELS)
    color_feats = embedder.embed_texts(list(M.COLOR_LABELS.values()))
    if detector is None and ecfg.run_detection:
        from hcfg.eval.embedders import OWLv2Detector

        detector = OWLv2Detector(ecfg.detector, device=config.model.device, threshold=ecfg.detection_threshold)

    results: dict = {"run": manifest["run"], "finetuner": manifest["finetuner"], "samples_dir": str(samples_dir), "per_prompt": {}}
    all_feats = []
    for key, entry in manifest["prompts"].items():
        paths = [samples_dir / r["path"] for r in entry["records"]]
        feats = embedder.embed_images(paths)
        all_feats.append(feats)
        text = plain_prompt(entry["prompt"])
        row: dict = {"prompt": entry["prompt"], "clip_prompt": text, "num_images": len(paths)}
        row.update(M.clip_score(feats, embedder.embed_texts([text]).expand_as(feats)))
        if reference is not None:
            gen = feats if cmmd_embedder is embedder else cmmd_embedder.embed_images(paths)
            row["cmmd"] = M.mmd_rbf(reference, gen)
        target = prompts.targets.get(key, {})
        for name, (names, text_feats) in labels.items():
            zs = M.zero_shot(feats, text_feats, names, logit_scale)
            row[f"{name}_histogram"] = zs["histogram"]
            row[f"{name}_mean_prob"] = zs["mean_prob"]
            if name == "weather" and "weather" in target:
                row["weather_target"] = M.target_rate(zs, target["weather"])
        if "vehicles" in target:
            vehicle_types = [v["type"] for v in target["vehicles"]]
            zs = M.zero_shot(feats, labels["vehicle"][1], labels["vehicle"][0], logit_scale)
            row["vehicle_target_zero_shot"] = {t: M.target_rate(zs, t) for t in vehicle_types}
            if detector is not None:
                row["vehicle_detection"] = M.vehicle_detection(
                    [str(p) for p in paths],
                    target["vehicles"],
                    detector=detector,
                    embedder=embedder,
                    color_features=color_feats,
                    color_labels=color_labels,
                )
        results["per_prompt"][key] = row
        logger.info("[%s] clip %.2f%s", key, row["clip_score"], f" cmmd {row['cmmd']:.3f}" if "cmmd" in row else "")
    if reference is not None and cmmd_embedder is embedder and all_feats:
        results["cmmd_all"] = M.mmd_rbf(reference, torch.cat(all_feats))
    save_json(results, samples_dir.parent / f"eval_{samples_dir.name}.json")
    return results
