"""CPU smoke test: the whole pipeline for all five fine-tuners on the random tiny model.

    python scripts/smoke_test.py

Creates 14 dummy 2:1 images (10 listed as training images, 4 held out), runs CA
measurement -> selection, trains each fine-tuner for a few steps, generates from
each checkpoint, evaluates with an offline fake embedder and runs the diagnostics
(numbers are meaningless; this checks the plumbing only). The work directory is
created under runs/ so it stays on the persistent volume on RunPod.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_pipeline_e2e import FakeDetector, FakeEmbedder  # noqa: E402

from hcfg.ca.measure import measure_ca, select_top_heads  # noqa: E402
from hcfg.config import load_config, resolve_path  # noqa: E402
from hcfg.data.images import training_image_paths  # noqa: E402
from hcfg.data.latent_cache import LatentCache  # noqa: E402
from hcfg.diagnostics.plot import plot_summaries  # noqa: E402
from hcfg.diagnostics.run import diagnose_run  # noqa: E402
from hcfg.eval.evaluate import evaluate_samples  # noqa: E402
from hcfg.generation.run import generate_for_run  # noqa: E402
from hcfg.models.heads import enumerate_projections  # noqa: E402
from hcfg.models.sdxl import load_sdxl  # noqa: E402
from hcfg.train.trainer import train  # noqa: E402
from hcfg.utils.io import load_json, save_json  # noqa: E402
from hcfg.utils.logging import setup_logging  # noqa: E402

RUNS = {
    "vanilla_lora": ["finetuner.kind=lora"],
    "ca_lora_style": ["finetuner.kind=ca_lora"],
    "ca_lora_style_dropout": ["finetuner.kind=ca_lora", "train.cond_dropout=0.1"],
    "dreambooth_lora": ["finetuner.kind=dreambooth_lora", "prompts.set=dreambooth",
                        "finetuner.dreambooth.num_class_images=2", "finetuner.dreambooth.class_num_inference_steps=2"],
    "textual_inversion": ["finetuner.kind=textual_inversion", "prompts.set=textual_inversion",
                          "diagnostics.base_prompt_set=default"],
}


def main() -> None:
    setup_logging()
    resolve_path("runs").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hcfg_smoke_", dir=resolve_path("runs")))
    images = work / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for i in range(14):
        Image.fromarray(rng.integers(0, 255, (32, 64, 3), dtype=np.uint8)).save(images / f"{i:02d}.png")
    image_list = work / "train10.txt"
    image_list.write_text("".join(f"{i:02d}.png\n" for i in range(10)), encoding="utf-8")
    common = [
        f"output_dir={work.as_posix()}/runs",
        f"data.image_dir={images.as_posix()}",
        f"data.image_list={image_list.as_posix()}",
        "diagnostics.save_tensors=[eps_ft_null, eps_base_null, g_domain]",
    ]
    selection = work / "selection.json"

    # CA measurement -> selection (what scripts/measure_ca.py does)
    config = load_config("configs/tiny_cpu.yaml", common)
    model = load_sdxl(config.model)
    cache = LatentCache.build(model, training_image_paths(config.data), hflip=False)
    latents = [cache.crop(lat, "center", torch.Generator()) for lat in cache.latents]
    templates = load_json("configs/ca_augmentations.json")["axes"]["style"]
    projections = enumerate_projections(model.unet)
    scores = measure_ca(
        model, projections, latents, model.encode_prompts(["Photorealistic first-person urban street view"]),
        [model.encode_prompts([t]) for t in templates], timestep=81, num_samples=4, augs_per_sample=4,
    )
    save_json(select_top_heads(projections, scores, fraction=0.05), selection)

    summaries = []
    for name, overrides in RUNS.items():
        cfg = load_config("configs/tiny_cpu.yaml", [*common, f"name={name}", f"finetuner.selection={selection.as_posix()}", *overrides])
        summary = train(cfg)
        samples = generate_for_run(cfg, checkpoint=summary["final_checkpoint"])
        evaluate_samples(cfg, samples, embedder=FakeEmbedder(), detector=FakeDetector())
        diagnostics = load_json(diagnose_run(cfg, checkpoint=summary["final_checkpoint"]) / "summary.json")
        if name == "textual_inversion":
            assert diagnostics["meta"]["checks"]["ti_diff_null_exact_zero"], "TI Diff_Null must be exactly 0"
        summaries.append(diagnostics)
        print(f"[ok] {name}: {summary['trainable_params']} trainable params, loss {summary['final_loss']:.4f}")
    plot_summaries(summaries, work / "diagnostics_overlay")
    print(f"Smoke test passed. Outputs in {work}")


if __name__ == "__main__":
    main()
