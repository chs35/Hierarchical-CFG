"""End-to-end on the tiny model: train each fine-tuner, CA selection, generation, evaluation."""

import json

import pytest
import torch

from hcfg.ca.measure import head_norms, measure_ca, select_top_heads, unit_names
from hcfg.config import load_config
from hcfg.data.images import training_image_paths
from hcfg.data.latent_cache import LatentCache
from hcfg.eval.evaluate import evaluate_samples, plain_prompt
from hcfg.eval.metrics import mmd_rbf, zero_shot
from hcfg.generation.run import generate_for_run
from hcfg.models.heads import enumerate_projections
from hcfg.models.sdxl import load_sdxl
from hcfg.train.trainer import train
from hcfg.utils.io import save_json


def _config(tmp_path, image_dir, *overrides):
    return load_config(
        "configs/tiny_cpu.yaml",
        [f"output_dir={tmp_path.as_posix()}/runs", f"data.image_dir={image_dir.as_posix()}", *overrides],
    )


def _write_selection(tmp_path, model):
    projections = enumerate_projections(model.unet)
    scores = torch.arange(sum(p.num_heads for p in projections), dtype=torch.float64)
    selection = select_top_heads(projections, scores, fraction=0.1)
    path = tmp_path / "selection.json"
    save_json(selection, path)
    return path


FINETUNERS = {
    "vanilla_lora": ["finetuner.kind=lora"],
    "ca_lora_style": ["finetuner.kind=ca_lora"],
    "ca_lora_style_dropout": ["finetuner.kind=ca_lora", "train.cond_dropout=0.5"],
    "dreambooth_lora": [
        "finetuner.kind=dreambooth_lora",
        "prompts.set=dreambooth",
        "finetuner.dreambooth.num_class_images=2",
        "finetuner.dreambooth.class_num_inference_steps=2",
        "finetuner.dreambooth.class_batch_size=2",
    ],
    "textual_inversion": ["finetuner.kind=textual_inversion", "prompts.set=textual_inversion"],
    "lora_phi_anchor": ["finetuner.kind=lora", "train.phi_anchor_lambda=0.5"],
}


@pytest.mark.parametrize("name", list(FINETUNERS))
def test_train_each_finetuner(tmp_path, image_dir, tiny_model, name):
    overrides = [f"name={name}", *FINETUNERS[name]]
    if "ca_lora" in name:
        overrides.append(f"finetuner.selection={_write_selection(tmp_path, tiny_model).as_posix()}")
    config = _config(tmp_path, image_dir, *overrides)
    summary = train(config)
    run = config.run_dir
    assert (run / "finetuner-final.safetensors").exists()
    assert summary["num_train_images"] == 10
    log = [json.loads(line) for line in (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in log)
    if name == "dreambooth_lora":
        assert "loss_prior" in log[-1] and summary["prior_prompt"] == "Photorealistic first-person urban street view"
    if name == "lora_phi_anchor":
        assert "loss_anchor" in log[-1]
    if name == "textual_inversion":
        assert "<s*>" in summary["train_prompt"]

    # generation from the checkpoint alone, then evaluation with offline fakes
    config = load_config(
        "configs/tiny_cpu.yaml",
        [*overrides, f"output_dir={tmp_path.as_posix()}/runs", "generate.prompts=[domain, snowy, green_truck]"],
    )
    samples = generate_for_run(config, checkpoint=run / "finetuner-final.safetensors")
    manifest = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["prompts"]) == {"domain", "snowy", "green_truck"}
    assert len(manifest["prompts"]["snowy"]["records"]) == 2
    result = evaluate_samples(config, samples, embedder=FakeEmbedder(), detector=FakeDetector())
    assert result["per_prompt"]["snowy"]["weather_target"]["target"] == "snowy"
    assert "vehicle_detection" in result["per_prompt"]["green_truck"]


def test_training_is_deterministic(tmp_path, image_dir):
    a = train(_config(tmp_path / "a", image_dir, "name=a"))
    b = train(_config(tmp_path / "b", image_dir, "name=b"))
    assert a["final_loss"] == b["final_loss"]


def test_generation_independent_of_batch_size(tmp_path, tiny_model):
    from hcfg.generation.generate import generate_images

    r1 = generate_images(tiny_model, "a street", num_images=3, seed=0, out_dir=tmp_path / "b1", num_inference_steps=2, batch_size=1)
    r3 = generate_images(tiny_model, "a street", num_images=3, seed=0, out_dir=tmp_path / "b3", num_inference_steps=2, batch_size=3)
    from PIL import Image
    import numpy as np

    for x, y in zip(r1, r3):
        a = np.asarray(Image.open(x["path"]), dtype=np.int16)
        b = np.asarray(Image.open(y["path"]), dtype=np.int16)
        assert np.abs(a - b).max() <= 1


def test_latent_cache_crops(tiny_model, image_dir, tiny_config):
    tiny_config.data.image_dir = image_dir.as_posix()
    cache = LatentCache.build(tiny_model, training_image_paths(tiny_config.data), hflip=True)
    s = tiny_model.latent_size
    assert cache.latents[0].shape[-2:] == (s, 2 * s)  # 2:1 frame -> wide latent
    assert not torch.allclose(cache.latents[0], cache.flipped[0].flip(-1))  # flip encoded, not flipped
    gen = torch.Generator().manual_seed(0)
    batch = cache.sample(4, crop="random", hflip=True, generator=gen)
    assert batch.shape == (4, 4, s, s)
    center = cache.crop(cache.latents[0], "center", gen)
    torch.testing.assert_close(center, cache.latents[0][:, :, s // 2 : s // 2 + s])


def test_wrong_image_count_raises(tmp_path, image_dir):
    (image_dir / "img_00.png").unlink()
    with pytest.raises(ValueError, match="Expected 10"):
        training_image_paths(_config(tmp_path, image_dir).data)


def test_head_norms_partition_the_matrix():
    from hcfg.models.heads import Projection

    p = Projection("x.attn1.to_q", "q", 1, num_heads=3, head_dim=2, in_features=5, out_features=6)
    g = torch.randn(6, 5)
    n = head_norms(g, p)
    torch.testing.assert_close(n, torch.stack([g[i * 2 : i * 2 + 2].norm() for i in range(3)]))
    torch.testing.assert_close(n.pow(2).sum(), g.norm().pow(2))
    p_out = Projection("x.attn1.to_out.0", "out", 1, num_heads=3, head_dim=2, in_features=6, out_features=5)
    g = torch.randn(5, 6)
    torch.testing.assert_close(head_norms(g, p_out), torch.stack([g[:, i * 2 : i * 2 + 2].norm() for i in range(3)]))


def test_measure_ca_and_select(tiny_model, image_dir, tiny_config):
    tiny_config.data.image_dir = image_dir.as_posix()
    cache = LatentCache.build(tiny_model, training_image_paths(tiny_config.data), hflip=False)
    gen = torch.Generator()
    latents = [cache.crop(lat, "center", gen) for lat in cache.latents]
    projections = enumerate_projections(tiny_model.unet)
    cond = tiny_model.encode_prompts(["Photorealistic first-person urban street view"])
    augs = [tiny_model.encode_prompts([t]) for t in ["Sketch of a street", "Watercolor of a street", "Pop-art of a street"]]
    scores = measure_ca(tiny_model, projections, latents, cond, augs, timestep=81, num_samples=3, augs_per_sample=2)
    assert scores.shape == (len(unit_names(projections)),) and torch.all(scores >= 0) and scores.sum() > 0
    assert not any(p.requires_grad for p in tiny_model.unet.parameters())  # grad state restored
    selection = select_top_heads(projections, scores, fraction=0.1)
    assert selection["num_projections"] == round(0.1 * selection["units_total"])
    assert all("@h" in n for n in selection["names"])


def test_mmd_properties():
    x = torch.randn(20, 8)
    assert abs(mmd_rbf(x, x)) < 1e-9
    assert mmd_rbf(x, x + 3.0) > 0


def test_plain_prompt():
    assert plain_prompt("Photorealistic first-person urban street view with sks style, in snowy weather") == (
        "Photorealistic first-person urban street view, in snowy weather"
    )
    assert plain_prompt("A with S* style") == "A"


def test_zero_shot_histogram():
    img = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1]]), dim=-1)
    txt = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    out = zero_shot(img, txt, ["a", "b"])
    assert out["histogram"] == {"a": 2, "b": 1}


class FakeEmbedder:
    """Offline stand-in for CLIP (hash-seeded vectors). Never report its numbers."""

    logit_scale = 100.0

    def _vec(self, key):
        g = torch.Generator().manual_seed(abs(hash(key)) % (2**31))
        return torch.nn.functional.normalize(torch.randn(16, generator=g), dim=0)

    def embed_images(self, images):
        return torch.stack([self._vec(str(getattr(i, "size", i))) for i in images])

    def embed_texts(self, texts):
        return torch.stack([self._vec(t) for t in texts])


class FakeDetector:
    def detect(self, image, queries):
        return [(1, 0.9, (0.0, 0.0, float(image.width), float(image.height)))]
