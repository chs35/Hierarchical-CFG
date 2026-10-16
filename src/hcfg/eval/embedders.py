"""CLIP image/text embeddings (L2-normalised) and a zero-shot detector wrapper.

Both are injectable: tests use tiny fakes with the same two/one methods, so the
metric plumbing runs offline. Numbers from fakes must never be reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

from hcfg.utils.device import resolve_device


def _features(output) -> torch.Tensor:
    """Projected CLIP features across transformers 4.x (Tensor) and 5.x (ModelOutput)."""
    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    raise TypeError(f"Cannot read CLIP features from {type(output).__name__}")


def _open(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as img:
        return img.convert("RGB")


class CLIPEmbedder:
    def __init__(self, model_name: str, *, device: str = "auto", batch_size: int = 16):
        from transformers import CLIPModel, CLIPProcessor

        self.device = resolve_device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self.name = model_name

    @torch.no_grad()
    def embed_images(self, images: Sequence[Image.Image | str | Path]) -> torch.Tensor:
        feats = []
        for start in range(0, len(images), self.batch_size):
            chunk = [_open(x) for x in images[start : start + self.batch_size]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            feats.append(_features(self.model.get_image_features(**inputs)).float())
        return torch.nn.functional.normalize(torch.cat(feats), dim=-1).cpu()

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        feats = _features(self.model.get_text_features(**inputs)).float()
        return torch.nn.functional.normalize(feats, dim=-1).cpu()

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp())


class OWLv2Detector:
    """Open-vocabulary detection: ``detect(image, queries) -> [(query_index, score, box)]``."""

    def __init__(self, model_name: str, *, device: str = "auto", threshold: float = 0.2):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = resolve_device(device)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.threshold = threshold

    @torch.no_grad()
    def detect(self, image: Image.Image | str | Path, queries: Sequence[str]) -> list[tuple[int, float, tuple]]:
        image = _open(image)
        inputs = self.processor(text=[list(queries)], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # OWLv2 pads to a square; boxes are relative to the padded square.
        side = max(image.size)
        target = torch.tensor([[side, side]], device=self.device)
        post = getattr(self.processor, "post_process_grounded_object_detection", None) or self.processor.post_process_object_detection
        result = post(outputs=outputs, threshold=self.threshold, target_sizes=target)[0]
        return [
            (int(label), float(score), tuple(float(v) for v in box))
            for label, score, box in zip(result["labels"], result["scores"], result["boxes"])
        ]
