"""Generation metrics for the fine-tuner comparison.

* ``cmmd`` — CLIP-MMD to real Cityscapes images (domain fit, lower = closer).
* ``clip_score`` — 100·max(0, cos(image, prompt)).
* ``zero_shot`` — CLIP zero-shot classification over a label set (weather /
  vehicle type / colour), the refine-condition read-out.
* ``vehicle_detection`` — OWLv2 open-vocabulary detection of the requested
  vehicle types, colour of each detected box by CLIP zero-shot on the crop.

The planned domain classifier P(Cityscapes | x) is not implemented (report §9.2);
CMMD stands in as the domain-fit metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

CMMD_SIGMA = 10.0
CMMD_SCALE = 1000.0

WEATHER_LABELS = {
    "clear": "a photo of a street on a clear day",
    "snowy": "a photo of a street in snowy weather",
    "foggy": "a photo of a street in foggy weather",
    "night": "a photo of a street at night",
}
VEHICLE_LABELS = {
    "passenger car": "a photo of a passenger car",
    "truck": "a photo of a truck",
    "SUV": "a photo of an SUV",
    "motorcycle": "a photo of a motorcycle",
}
COLOR_LABELS = {c: f"a photo of a {c} vehicle" for c in (
    "red", "orange", "yellow", "green", "blue", "white", "black", "gray",
)}
DETECTION_QUERIES = {
    "passenger car": "a photo of a passenger car",
    "truck": "a photo of a truck",
    "SUV": "a photo of an SUV",
    "motorcycle": "a photo of a motorcycle",
}


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, *, sigma: float = CMMD_SIGMA, scale: float = CMMD_SCALE) -> float:
    """Squared MMD, Gaussian RBF kernel, biased V-statistic (reference CMMD implementation)."""
    x, y = x.double(), y.double()
    gamma = 1.0 / (2.0 * sigma**2)

    def k(a, b):
        d = a.pow(2).sum(1)[:, None] + b.pow(2).sum(1)[None, :] - 2.0 * a @ b.T
        return torch.exp(-gamma * d).mean()

    return float(scale * (k(x, x) + k(y, y) - 2.0 * k(x, y)))


def clip_score(image_features: torch.Tensor, text_features: torch.Tensor, weight: float = 100.0) -> dict:
    cos = (image_features * text_features).sum(-1).clamp(min=0.0)
    scores = weight * cos
    return {"clip_score": float(scores.mean()), "clip_score_std": float(scores.std()) if len(scores) > 1 else 0.0}


def zero_shot(image_features: torch.Tensor, label_features: torch.Tensor, labels: Sequence[str], logit_scale: float = 100.0) -> dict:
    """Per-image probabilities over ``labels`` + top-1 predictions and label histogram."""
    probs = (logit_scale * image_features @ label_features.T).softmax(-1)
    top = probs.argmax(-1).tolist()
    hist = {label: 0 for label in labels}
    for i in top:
        hist[labels[i]] += 1
    return {
        "probs": probs,
        "pred": [labels[i] for i in top],
        "histogram": hist,
        "mean_prob": {label: float(probs[:, j].mean()) for j, label in enumerate(labels)},
    }


def target_rate(result: dict, target: str) -> dict:
    n = len(result["pred"])
    return {
        "target": target,
        "top1_rate": sum(p == target for p in result["pred"]) / max(n, 1),
        "mean_prob": result["mean_prob"][target],
    }


def vehicle_detection(
    images: Sequence[str | Image.Image],
    targets: Sequence[dict],
    *,
    detector,
    embedder,
    color_features: torch.Tensor,
    color_labels: Sequence[str],
) -> dict:
    """For each ``{type, color}`` target: fraction of images with a detected ``type`` box,
    and with such a box classified as ``color``; plus fraction satisfying *all* targets."""
    type_names = list(DETECTION_QUERIES)
    queries = [DETECTION_QUERIES[t] for t in type_names]
    per_target = {f"{t['color']} {t['type']}": {"type": 0, "type_color": 0} for t in targets}
    all_ok = 0
    detections_per_image = []
    for image in images:
        pil = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        found = []
        for label, score, box in detector.detect(pil, queries):
            x0, y0, x1, y1 = box
            crop = pil.crop((max(0, x0), max(0, y0), min(pil.width, x1), min(pil.height, y1)))
            if crop.width < 4 or crop.height < 4:
                continue
            feat = embedder.embed_images([crop])
            color = color_labels[int((feat @ color_features.T).argmax())]
            found.append({"type": type_names[label], "score": score, "color": color, "box": box})
        detections_per_image.append(found)
        ok_all = True
        for t in targets:
            key = f"{t['color']} {t['type']}"
            has_type = any(d["type"] == t["type"] for d in found)
            has_both = any(d["type"] == t["type"] and d["color"] == t["color"] for d in found)
            per_target[key]["type"] += has_type
            per_target[key]["type_color"] += has_both
            ok_all &= has_both
        all_ok += ok_all
    n = max(len(images), 1)
    return {
        "per_target": {k: {"type_rate": v["type"] / n, "type_color_rate": v["type_color"] / n} for k, v in per_target.items()},
        "all_targets_rate": all_ok / n,
        "mean_detections": sum(len(d) for d in detections_per_image) / n,
    }
