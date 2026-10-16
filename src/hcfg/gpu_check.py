"""GPU verification of the whole pipeline on the real SDXL checkpoint (A100).

The CPU test suite runs every code path on a random tiny model. What it cannot
check is anything that depends on the real weights or the real hardware; that is
what the stages here cover:

``env``         CUDA / A100 / bf16 / TF32 / library versions.
``exact``       On real SDXL (bf16): repeat-forward determinism, adapter scale 0 ==
                Base bit-exactly for the three LoRA fine-tuners, TI Diff_Null == 0,
                the reference CA selection maps onto real SDXL (208 heads / 65
                matrices), sliced-LoRA checkpoint round trip, dtypes, SDPA attention.
``tensorcore``  torch.profiler over real training steps: share of GEMM/conv/attention
                CUDA time spent in tensor-core kernels (docs/REPRODUCE.md, performance notes).
``train``       Short real training run for each fine-tuner: finite losses, s/step,
                peak memory, checkpoint written. Optional fp32 baseline for the speedup.
``generate``    Generation from each short-run checkpoint: images are not NaN/blank.
``evaluate``    Real CLIP (+ OWLv2) evaluation of one sample set.
``ca``          CA measurement on a few samples: runs, timing extrapolated to the config.

Each check yields PASS / WARN / FAIL / SKIP; the report is written as JSON.
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hcfg.config import Config, load_config, resolve_path
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

FINETUNER_CONFIGS = {
    "vanilla_lora": "configs/vanilla_lora.yaml",
    "ca_lora_style": "configs/ca_lora_style.yaml",
    "ca_lora_style_dropout": "configs/ca_lora_style_dropout.yaml",
    "dreambooth_lora": "configs/dreambooth_lora.yaml",
    "textual_inversion": "configs/textual_inversion.yaml",
}
REF_SELECTION = "configs/selections/ref_style_t81.json"
STAGES = ("env", "exact", "tensorcore", "train", "generate", "evaluate", "ca")

# Seconds per step at 1024px, batch 1, bf16 on A100 (docs/REPRODUCE.md, performance notes: paper
# ~0.36 s/step on V100-class settings; the broken fp32 reproduction was 2.2 s/step).
SEC_PER_STEP_WARN = 0.6
SEC_PER_STEP_FAIL = 1.0
TENSOR_CORE_SHARE_MIN = 0.8


@dataclass
class Check:
    stage: str
    name: str
    status: str  # PASS / WARN / FAIL / SKIP
    value: object = None
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, stage: str, name: str, ok: bool | None, value=None, detail: str = "", warn: bool = False) -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else ("WARN" if warn else "FAIL"))
        self.checks.append(Check(stage, name, status, _jsonable(value), detail))
        logger.info("[%s] %-4s %s = %s %s", stage, status, name, value, detail)

    @property
    def failed(self) -> bool:
        return any(c.status == "FAIL" for c in self.checks)

    def to_dict(self) -> dict:
        return {"info": self.info, "checks": [asdict(c) for c in self.checks]}


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def synthetic_images(directory: Path, count: int = 10, size=(2048, 1024)) -> Path:
    """Smooth random 2:1 images (Cityscapes shape) for plumbing/speed checks only."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        small = rng.integers(0, 255, (size[1] // 64, size[0] // 64, 3), dtype=np.uint8)
        Image.fromarray(small).resize(size, Image.BICUBIC).save(directory / f"synthetic_{i:02d}.png")
    return directory


def finetuner_config(name: str, out_dir: Path, image_dir: Path, *extra: str) -> Config:
    overrides = [
        f"name={name}",
        f"output_dir={out_dir.as_posix()}",
        f"data.image_dir={image_dir.as_posix()}",
    ]
    if "ca_lora" in name:
        overrides.append(f"finetuner.selection={REF_SELECTION}")
    if name == "dreambooth_lora":
        overrides += [
            "finetuner.dreambooth.num_class_images=4",
            f"finetuner.dreambooth.class_data_dir={(out_dir / 'class_images').as_posix()}",
        ]
    return load_config(FINETUNER_CONFIGS[name], [*overrides, *extra])


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------


def stage_env(report: Report) -> None:
    import diffusers
    import transformers

    cuda = torch.cuda.is_available()
    report.info.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        }
    )
    report.add("env", "cuda_available", cuda, cuda)
    if not cuda:
        return
    from hcfg.utils.device import configure_backends

    configure_backends()
    name = torch.cuda.get_device_name()
    props = torch.cuda.get_device_properties(0)
    report.info.update({"gpu": name, "gpu_memory_gb": round(props.total_memory / 2**30, 1), "sm": f"{props.major}.{props.minor}"})
    report.add("env", "gpu_is_a100", "A100" in name, name, warn=True)
    report.add("env", "compute_capability>=8.0", props.major >= 8, f"{props.major}.{props.minor}")
    report.add("env", "bf16_supported", torch.cuda.is_bf16_supported())
    report.add("env", "tf32_matmul_enabled", torch.backends.cuda.matmul.allow_tf32)
    report.add("env", "tf32_cudnn_enabled", torch.backends.cudnn.allow_tf32)
    report.add("env", "sdpa_flash_available", torch.backends.cuda.flash_sdp_enabled(), warn=True)


def _eps(model, cond, seed: int = 0):
    gen = torch.Generator(device=model.device).manual_seed(seed)
    z = torch.randn(cond.batch_size, 4, model.latent_size, model.latent_size, generator=gen, device=model.device)
    t = torch.tensor([500, 81][: cond.batch_size], device=model.device)
    with torch.no_grad():
        return model.predict(z.to(model.weight_dtype), t, cond).float()


def _max_diff(a, b) -> float:
    return float((a - b).abs().max())


def stage_exact(report: Report, base_config: Config) -> None:
    """Bit-exact identities on the real bf16 model (deterministic kernels)."""
    from hcfg.models.finetuners import attach_finetuner, expand_ti_prompt, load_finetuner, save_finetuner
    from hcfg.models.finetuners import load_selection_units
    from hcfg.models.heads import enumerate_projections, plans_from_units
    from hcfg.models.lora import disable_adapter, iter_lora, lora_summary
    from hcfg.models.sdxl import load_sdxl

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prompts = ["Photorealistic first-person urban street view", "Photorealistic first-person urban street view, in snowy weather"]

    # -- dtypes / attention / determinism on the Base model
    model = load_sdxl(base_config.model)
    unet_dtype = next(model.unet.parameters()).dtype
    report.add("exact", "unet_weight_dtype_bf16", unet_dtype == torch.bfloat16, str(unet_dtype))
    report.add("exact", "vae_dtype_fp32", model.vae.dtype == torch.float32, str(model.vae.dtype))
    processors = {type(p).__name__ for p in model.unet.attn_processors.values()}
    report.add("exact", "attention_is_sdpa", processors == {"AttnProcessor2_0"}, sorted(processors), warn=True)
    cond = model.encode_prompts(prompts)
    null = model.null_cond(2)
    e1, e2 = _eps(model, cond), _eps(model, cond)
    deterministic = torch.equal(e1, e2)
    report.add("exact", "base_forward_repeatable", deterministic, _max_diff(e1, e2),
               "exactness checks below need this; otherwise they fall back to tolerance", warn=True)
    tol = 0.0 if deterministic else 1e-3
    report.add("exact", "eps_finite", bool(torch.isfinite(e1).all()))

    # -- the reference CA selection on real SDXL
    projections = enumerate_projections(model.unet)
    units = load_selection_units(REF_SELECTION)
    plans = plans_from_units(projections, units)
    heads = sum(len(p.heads) if p.heads is not None else p.projection.num_heads for p in plans)
    report.add("exact", "ref_selection_units", heads == 208, heads)
    report.add("exact", "ref_selection_modules", len(plans) == 65, len(plans))
    report.info["num_attention_projections"] = len(projections)
    report.info["num_head_units"] = sum(p.num_heads for p in projections)
    del model
    _free()

    # -- adapter scale 0 == Base, for every LoRA fine-tuner (§8-3)
    for name in ("vanilla_lora", "ca_lora_style", "dreambooth_lora"):
        config = load_config(FINETUNER_CONFIGS[name], [f"finetuner.selection={REF_SELECTION}"] if "ca" in name else [])
        model = load_sdxl(config.model)
        base_c, base_null = _eps(model, cond), _eps(model, null)
        attach_finetuner(model, config)
        g = torch.Generator(device=model.device).manual_seed(0)
        for _, m in iter_lora(model.unet):
            m.lora_B.data.copy_(torch.randn(m.lora_B.shape, generator=g, device=model.device) * 0.01)
        ft_c = _eps(model, cond)
        with disable_adapter(model.unet):
            off_c, off_null = _eps(model, cond), _eps(model, null)
        diff = max(_max_diff(off_c, base_c), _max_diff(off_null, base_null))
        report.add("exact", f"{name}:scale0_equals_base", diff <= tol, diff)
        report.add("exact", f"{name}:adapter_changes_output", _max_diff(ft_c, base_c) > 0, _max_diff(ft_c, base_c))
        lora_a = next(m for _, m in iter_lora(model.unet)).lora_A.dtype
        report.add("exact", f"{name}:lora_master_fp32", lora_a == torch.float32, str(lora_a))
        if name == "ca_lora_style":
            summary = lora_summary(model.unet)
            report.info["ca_lora_ref_selection"] = summary
            report.add("exact", "ca_lora:sliced_modules", summary["num_sliced_modules"] > 0, summary["num_sliced_modules"])
            # checkpoint round trip (heads recovered from the checkpoint alone)
            path = Path(tempfile.mkdtemp()) / "ca.safetensors"
            save_finetuner(model, config, path)
            del model
            _free()
            fresh = load_sdxl(config.model)
            load_finetuner(fresh, path)
            rt = _eps(fresh, cond)
            report.add("exact", "ca_lora:checkpoint_roundtrip", _max_diff(rt, ft_c) <= tol, _max_diff(rt, ft_c))
            model = fresh
        del model
        _free()

    # -- Textual Inversion: Diff_Null == 0 (§8-2)
    config = load_config(FINETUNER_CONFIGS["textual_inversion"])
    model = load_sdxl(config.model)
    base_null = _eps(model, model.null_cond(2))
    params = attach_finetuner(model, config)
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p) * 0.1)
    ft_null_cond = model.null_cond(2)
    diff = _max_diff(_eps(model, ft_null_cond), base_null)
    report.add("exact", "textual_inversion:diff_null_zero", diff == 0.0 if deterministic else diff <= tol, diff)
    s_prompt = expand_ti_prompt("Photorealistic first-person urban street view with S* style", config)
    token_ids = [tok.convert_tokens_to_ids(config.finetuner.textual_inversion.token) for tok in model.tokenizers]
    in_prompt = all(tid in tok(s_prompt).input_ids for tid, tok in zip(token_ids, model.tokenizers))
    report.add("exact", "textual_inversion:token_in_both_tokenizers", in_prompt, s_prompt)
    del model
    _free()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


_TC_MARKERS = ("16816", "1688", "tensorop", "xmma", "hmma", "bf16", "tf32", "flash", "fmha", "cutlass_80", "sm80", "sm90")
_MATH_MARKERS = ("gemm", "conv", "fprop", "dgrad", "wgrad", "flash", "fmha", "attention", "xmma", "cutlass", "implicit")


def classify_kernels(events) -> dict:
    """Split CUDA time of GEMM/conv/attention kernels into tensor-core vs other."""
    tc = other = 0.0
    top_other: list[tuple[str, float]] = []
    for evt in events:
        time_us = getattr(evt, "self_device_time_total", None)
        if time_us is None:
            time_us = getattr(evt, "self_cuda_time_total", 0.0)
        if not time_us:
            continue
        name = evt.key.lower()
        if not any(m in name for m in _MATH_MARKERS):
            continue
        if "sgemm" not in name and any(m in name for m in _TC_MARKERS):
            tc += time_us
        else:
            other += time_us
            top_other.append((evt.key, time_us))
    total = tc + other
    top_other.sort(key=lambda x: -x[1])
    return {
        "tensor_core_share": tc / total if total else 0.0,
        "math_time_ms": total / 1e3,
        "top_non_tensor_core": [(k[:120], round(v / 1e3, 2)) for k, v in top_other[:5]],
    }


def stage_tensorcore(report: Report, out_dir: Path, image_dir: Path) -> None:
    from torch.profiler import ProfilerActivity, profile, schedule

    from hcfg.train.trainer import train

    for name in ("vanilla_lora", "ca_lora_style", "textual_inversion"):
        config = finetuner_config(
            name, out_dir / "tensorcore", image_dir,
            "train.max_steps=8", "train.log_every=4", "train.checkpoint_every=0",
        )
        with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=2, warmup=2, active=3)) as prof:
            train(config, step_callback=lambda _step: prof.step())
        stats = classify_kernels(prof.key_averages())
        report.info[f"tensorcore_{name}"] = stats
        share = stats["tensor_core_share"]
        report.add("tensorcore", f"{name}:tensor_core_share", share >= TENSOR_CORE_SHARE_MIN, round(share, 3),
                   f"top non-TC kernels: {stats['top_non_tensor_core'][:2]}", warn=share >= 0.5)
        _free()


def _window_sec_per_step(run_dir: Path) -> float | None:
    rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    steady = [r["sec_per_step"] for r in rows if r["step"] > rows[0]["step"] and r["step"] > 20]
    return statistics.median(steady) if steady else None


def stage_train(report: Report, out_dir: Path, image_dir: Path, steps: int, names: list[str], compare_fp32: bool) -> dict:
    from hcfg.train.trainer import train

    checkpoints = {}
    runs = list(names)
    if compare_fp32:
        runs.append("vanilla_lora_fp32")
    for name in runs:
        extra = [f"train.max_steps={steps}", "train.log_every=20", "train.checkpoint_every=0"]
        config_name = name
        if name == "vanilla_lora_fp32":
            config_name = "vanilla_lora"
            extra += ["model.weight_dtype=float32", "perf.mixed_precision=no", "perf.allow_tf32=false"]
        try:
            config = finetuner_config(config_name, out_dir / "train", image_dir, *extra, f"name={name}")
            start = time.perf_counter()
            summary = train(config)
            wall = time.perf_counter() - start
        except Exception as exc:  # keep going: one broken fine-tuner should not hide the others
            report.add("train", f"{name}:runs", False, None, f"{type(exc).__name__}: {exc}")
            logger.error(traceback.format_exc())
            _free()
            continue
        rows = [json.loads(line) for line in (config.run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        finite = all(np.isfinite(r["loss"]) for r in rows)
        sps = _window_sec_per_step(config.run_dir) or summary["sec_per_step_mean"]
        report.info[f"train_{name}"] = {**{k: summary[k] for k in ("trainable_params", "peak_memory_gb", "final_loss")},
                                        "sec_per_step": sps, "wall_sec": wall}
        report.add("train", f"{name}:finite_loss", finite, rows[-1]["loss"])
        report.add("train", f"{name}:checkpoint_written", Path(summary["final_checkpoint"]).exists(), summary["final_checkpoint"])
        if name != "vanilla_lora_fp32":
            report.add("train", f"{name}:sec_per_step", sps <= SEC_PER_STEP_WARN, round(sps, 3),
                       f"warn>{SEC_PER_STEP_WARN}, fail>{SEC_PER_STEP_FAIL}; 10k steps ~ {sps * 1e4 / 3600:.2f} h",
                       warn=sps <= SEC_PER_STEP_FAIL)
            checkpoints[name] = (config, summary["final_checkpoint"])
        peak = summary["peak_memory_gb"] or 0.0
        report.add("train", f"{name}:peak_memory_gb", peak < 75, round(peak, 1))
        _free()
    if compare_fp32 and "train_vanilla_lora" in report.info and "train_vanilla_lora_fp32" in report.info:
        speedup = report.info["train_vanilla_lora_fp32"]["sec_per_step"] / report.info["train_vanilla_lora"]["sec_per_step"]
        report.add("train", "bf16_speedup_over_fp32", speedup >= 2.0, round(speedup, 2), "expected ~3-5x", warn=True)
    return checkpoints


def _image_ok(path: Path) -> tuple[bool, float]:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    std = float(array.std())
    return bool(np.isfinite(array).all() and std > 0.02), std


def stage_generate(report: Report, checkpoints: dict) -> dict:
    from hcfg.generation.run import generate_for_run

    samples = {}
    for name, (config, checkpoint) in checkpoints.items():
        config.generate.images_per_prompt = 2
        config.generate.batch_size = 2
        start = time.perf_counter()
        try:
            out = generate_for_run(config, checkpoint=checkpoint, samples_name="samples_check", keys=["domain", "snowy"])
        except Exception as exc:
            report.add("generate", f"{name}:runs", False, None, f"{type(exc).__name__}: {exc}")
            logger.error(traceback.format_exc())
            _free()
            continue
        sec_per_image = (time.perf_counter() - start) / 4
        images = sorted(out.rglob("*.png"))
        results = [_image_ok(p) for p in images]
        report.add("generate", f"{name}:images_valid", len(images) == 4 and all(ok for ok, _ in results),
                   [round(s, 3) for _, s in results])
        report.add("generate", f"{name}:sec_per_image", sec_per_image < 15, round(sec_per_image, 2),
                   "includes model load; 25 Euler steps at 1024px", warn=True)
        samples[name] = (config, out)
        _free()
    return samples


def stage_evaluate(report: Report, samples: dict, reference_dir: str | None) -> None:
    from hcfg.eval.evaluate import evaluate_samples

    if not samples:
        report.add("evaluate", "samples_available", None, detail="no generate stage output")
        return
    name, (config, out) = next(iter(samples.items()))
    if reference_dir:
        config.eval.reference_dir = reference_dir
        config.eval.reference_num_images = 50
    try:
        result = evaluate_samples(config, out)
    except Exception as exc:
        report.add("evaluate", "runs", False, None, f"{type(exc).__name__}: {exc}")
        logger.error(traceback.format_exc())
        return
    row = result["per_prompt"]["snowy"]
    report.add("evaluate", f"{name}:clip_score_finite", np.isfinite(row["clip_score"]), round(row["clip_score"], 2))
    report.add("evaluate", f"{name}:weather_zero_shot", "weather_target" in row, row.get("weather_target"))
    if reference_dir:
        report.add("evaluate", f"{name}:cmmd", np.isfinite(row.get("cmmd", np.nan)), row.get("cmmd"))
    else:
        report.add("evaluate", "cmmd", None, detail="pass --reference-dir to check CMMD")
    _free()


def stage_ca(report: Report, image_dir: Path) -> None:
    from hcfg.ca.measure import measure_ca, select_top_heads
    from hcfg.data.images import training_image_paths
    from hcfg.data.latent_cache import LatentCache
    from hcfg.models.heads import enumerate_projections
    from hcfg.models.sdxl import load_sdxl
    from hcfg.utils.device import autocast_dtype
    from hcfg.utils.io import load_json

    config = load_config(FINETUNER_CONFIGS["ca_lora_style"], [f"data.image_dir={image_dir.as_posix()}"])
    ca = config.finetuner.ca
    model = load_sdxl(config.model, weight_dtype="float32")
    cache = LatentCache.build(model, training_image_paths(config.data), hflip=False)
    latents = [cache.crop(lat, "center", torch.Generator()) for lat in cache.latents]
    templates = load_json(resolve_path(ca.augmentation_file))["axes"][ca.axis]
    cond = model.encode_prompts(["Photorealistic first-person urban street view"])
    augs = [model.encode_prompts([t]) for t in templates]
    projections = enumerate_projections(model.unet)
    torch.cuda.reset_peak_memory_stats()
    n = 4
    start = time.perf_counter()
    scores = measure_ca(model, projections, latents, cond, augs, timestep=ca.timestep, num_samples=n,
                        augs_per_sample=ca.augs_per_sample, amp=autocast_dtype(config.perf.mixed_precision, model.device))
    torch.cuda.synchronize()
    per_sample = (time.perf_counter() - start) / n
    selection = select_top_heads(projections, scores, fraction=ca.fraction)
    report.add("ca", "scores_finite_positive", bool(torch.isfinite(scores).all() and (scores > 0).any()),
               float(scores.mean()))
    report.add("ca", "selection_size", selection["num_projections"] == round(ca.fraction * len(scores)),
               selection["num_projections"])
    report.add("ca", "sec_per_sample", True, round(per_sample, 2),
               f"full measurement ({ca.num_samples} samples) ~ {per_sample * ca.num_samples / 60:.1f} min")
    report.add("ca", "peak_memory_gb", torch.cuda.max_memory_allocated() / 2**30 < 75,
               round(torch.cuda.max_memory_allocated() / 2**30, 1))
    del model
    _free()


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_gpu_check(
    *,
    out_dir: str | Path = "runs/gpu_check",
    image_dir: str | None = None,
    reference_dir: str | None = None,
    stages: list[str] | None = None,
    steps: int = 60,
    finetuners: list[str] | None = None,
    compare_fp32: bool = False,
) -> Report:
    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = stages or list(STAGES)
    names = finetuners or list(FINETUNER_CONFIGS)
    report = Report()
    report.info["stages"] = stages
    stage_env(report)
    if not torch.cuda.is_available():
        report.add("env", "abort", False, detail="no CUDA device: run this on the GPU server")
        return _finish(report, out_dir)

    if image_dir:
        images = resolve_path(image_dir)
        report.info["image_source"] = str(images)
    else:
        images = synthetic_images(out_dir / "synthetic_images")
        report.info["image_source"] = "synthetic (speed/plumbing only)"

    base_config = load_config("configs/base.yaml")
    checkpoints: dict = {}
    samples: dict = {}
    for stage in stages:
        if stage == "env":
            continue
        start = time.perf_counter()
        logger.info("===== stage %s =====", stage)
        try:
            if stage == "exact":
                stage_exact(report, base_config)
            elif stage == "tensorcore":
                stage_tensorcore(report, out_dir, images)
            elif stage == "train":
                checkpoints = stage_train(report, out_dir, images, steps, names, compare_fp32)
            elif stage == "generate":
                samples = stage_generate(report, checkpoints)
            elif stage == "evaluate":
                stage_evaluate(report, samples, reference_dir)
            elif stage == "ca":
                stage_ca(report, images)
            else:
                raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}")
        except Exception as exc:
            report.add(stage, "stage_crashed", False, None, f"{type(exc).__name__}: {exc}")
            logger.error(traceback.format_exc())
            _free()
        report.info[f"stage_seconds_{stage}"] = round(time.perf_counter() - start, 1)
        _finish(report, out_dir, quiet=True)  # partial report survives a later crash
    return _finish(report, out_dir)


def _finish(report: Report, out_dir: Path, quiet: bool = False) -> Report:
    path = out_dir / "report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if not quiet:
        counts = {s: sum(c.status == s for c in report.checks) for s in ("PASS", "WARN", "FAIL", "SKIP")}
        logger.info("GPU check: %s -> %s", counts, path)
        for c in report.checks:
            if c.status in ("FAIL", "WARN"):
                logger.info("  %s [%s] %s = %s %s", c.status, c.stage, c.name, c.value, c.detail)
    return report
