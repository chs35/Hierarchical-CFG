"""Run the diagnostics of one fine-tuner checkpoint (report §3, docs/diagnostics.md).

Output (``<run_dir>/<out_name>/``, default ``diagnostics``)::

    config.yaml  heldout_images.txt  progress.json
    records.jsonl                        # one line per (sample, t), appended + fsynced per unit
    summary.json  *.png                  # rebuilt from records (+ tensors) at the end
    trajectory_images/<key>/seed<s>/<i>.png
    tensors/<unit>.safetensors           # optional, fp16 [N, T, C, h, w]

Work is split into *units* (a batch of held-out images for one noise seed, or a
batch of trajectories of one prompt and seed). A unit's tensors, records and its
entry in ``progress.json`` are written only once it is complete, so a pod that
is closed mid-run loses at most the unit in flight; ``resume=True`` skips the
completed units and produces the same records as an uninterrupted run.
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
from hcfg.data.prompts import CONTROL_KEY, PromptSet, load_prompt_set
from hcfg.diagnostics.branches import NULL_KEY, encode_rows, predict_ft_base, predict_rows, row_texts
from hcfg.diagnostics.forward import forward_noise, heldout_latents, relative_names, select_heldout_images
from hcfg.diagnostics.metrics import diagnostic_tensors, sample_metrics
from hcfg.diagnostics.storage import check_disk_space, ensure_persistent, estimate_output_bytes, tensor_names
from hcfg.diagnostics.summary import summarize
from hcfg.diagnostics.trajectory import euler_cfg_sample, sampler_timesteps
from hcfg.generation.generate import decode_latents, image_seed
from hcfg.models.finetuners import expand_ti_prompt, load_finetuner
from hcfg.models.sdxl import SDXL, load_sdxl
from hcfg.utils.device import configure_backends, deterministic_mode, resolve_device
from hcfg.utils.io import save_json
from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

FINAL_CHECKPOINT = "finetuner-final.safetensors"


# --------------------------------------------------------------------------------------
# Durable, resumable output
# --------------------------------------------------------------------------------------


def _fsync_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + fsync + rename)."""
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def unit_filename(unit: str) -> str:
    return unit.replace("/", "__") + ".safetensors"


class ProgressStore:
    """``records.jsonl`` + ``tensors/`` + ``progress.json`` of one output directory."""

    def __init__(self, out_dir: Path, fingerprint: dict, *, resume: bool):
        self.out_dir = out_dir
        self.progress_path = out_dir / "progress.json"
        self.records_path = out_dir / "records.jsonl"
        self.tensor_dir = out_dir / "tensors"
        self.fingerprint = fingerprint
        if resume:
            if not self.progress_path.exists():
                raise FileNotFoundError(f"Nothing to resume in {out_dir} (no progress.json)")
            state = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if state["fingerprint"] != fingerprint:
                diff = sorted(k for k in fingerprint if state["fingerprint"].get(k) != fingerprint[k])
                raise ValueError(f"Cannot resume {out_dir}: settings changed ({diff}); use a new --out-name")
            self.done = list(state["units"])
            done = set(self.done)
            kept = [line for line in self._lines() if json.loads(line)["unit"] in done]
            _fsync_write(self.records_path, "".join(kept))
            for stray in list(out_dir.rglob("*.partial")):
                stray.unlink()
            logger.info("Resuming %s: %d units already done", out_dir, len(self.done))
        else:
            if self.progress_path.exists() or self.records_path.exists():
                raise FileExistsError(
                    f"{out_dir} already holds diagnostics; pass resume=True (--resume) or another out_name"
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            self.done = []
            _fsync_write(self.records_path, "")
            self._save()

    def _lines(self) -> list[str]:
        if not self.records_path.exists():
            return []
        return [line for line in self.records_path.read_text(encoding="utf-8").splitlines(keepends=True) if line.strip()]

    def _save(self) -> None:
        _fsync_write(self.progress_path, json.dumps({"fingerprint": self.fingerprint, "units": self.done}, indent=1))

    def is_done(self, unit: str) -> bool:
        return unit in self.done

    def commit(self, unit: str, records: list[dict], tensors: dict[str, torch.Tensor], tensor_meta: dict) -> None:
        if tensors:
            from safetensors.torch import save_file

            self.tensor_dir.mkdir(parents=True, exist_ok=True)
            path = self.tensor_dir / unit_filename(unit)
            tmp = path.with_name(path.name + ".partial")
            meta = {k: v if isinstance(v, str) else json.dumps(v) for k, v in tensor_meta.items()}
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp), metadata=meta)
            os.replace(tmp, path)
        text = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
        with open(self.records_path, "a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        self.done.append(unit)
        self._save()

    def records(self) -> list[dict]:
        return [json.loads(line) for line in self._lines()]

    def tensor_files(self) -> list[Path]:
        return [self.tensor_dir / unit_filename(u) for u in self.done if (self.tensor_dir / unit_filename(u)).exists()]


class TensorBuffer:
    """Collects the requested terms of one unit over its steps -> ``[N, T, C, h, w]`` fp16."""

    def __init__(self, names: list[str]):
        self.names = names
        self.steps: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    def add(self, terms: dict[str, torch.Tensor]) -> None:
        for name in self.names:
            self.steps[name].append(terms[name].to(device="cpu", dtype=torch.float16))

    def stacked(self) -> dict[str, torch.Tensor]:
        return {name: torch.stack(v, dim=1) for name, v in self.steps.items() if v}


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------


def default_out_name(checkpoint: Path) -> str:
    return "diagnostics" if checkpoint.name == FINAL_CHECKPOINT else f"diagnostics_{checkpoint.stem}"


def read_checkpoint_meta(path: Path) -> dict:
    """Checkpoint metadata without installing it (both LoRA and TI formats)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        raw = dict(handle.metadata() or {})
    extra = json.loads(raw.pop("extra", "{}") or "{}")
    return {**raw, **extra}


def base_prompt_set(config: Config) -> PromptSet:
    name = config.diagnostics.base_prompt_set or config.prompts.set
    return load_prompt_set(PromptConfig(file=config.prompts.file, set=name))


def check_base_prompts(ckpt_meta: dict, texts: dict[str, str]) -> None:
    """Textual Inversion: Base has no S*, so its c / c_refine must be the plain prompts."""
    marks = [m for m in (ckpt_meta.get("placeholder"), ckpt_meta.get("token")) if m]
    bad = {k: v for k, v in texts.items() if any(m in v for m in marks)}
    if bad:
        raise ValueError(
            f"Base prompts contain the Textual Inversion placeholder {marks}: {bad}. "
            "Set diagnostics.base_prompt_set=default (configs/textual_inversion.yaml does)."
        )


def fingerprint(config: Config, checkpoint: Path, sources: list[str]) -> dict:
    """Everything that changes the measured values or the unit layout."""
    asdict = dataclasses.asdict
    model = {k: v for k, v in asdict(config.model).items() if k != "device"}
    gen = config.generate
    return {
        "checkpoint": str(checkpoint),
        "sources": list(sources),
        "diagnostics": {k: v for k, v in asdict(config.diagnostics).items() if k != "max_rows"},
        "model": model,
        "generate": [gen.num_inference_steps, gen.guidance_scale, gen.guidance_rescale],
        "prompts": asdict(config.prompts),
        "data": [config.data.image_dir, config.data.image_list],
    }


def preflight(config: Config, out_dir: Path, sources: list[str]) -> dict:
    """Persistence + disk checks; raises before anything is written or loaded."""
    ensure_persistent(out_dir, config.storage)
    ensure_persistent(config.run_dir, config.storage)
    return check_disk_space(out_dir, estimate_output_bytes(config, sources), config.storage)


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Context:
    config: Config
    model: SDXL
    store: ProgressStore
    ft_rows: dict
    base_rows: dict
    drive_rows: dict
    refine_keys: list[str]
    tensor_names: list[str]
    units_total: int
    started: float = dataclasses.field(default_factory=time.time)
    units_run: int = 0

    def finished(self, unit: str) -> None:
        self.units_run += 1
        elapsed = time.time() - self.started
        left = self.units_total - len(self.store.done)
        eta = elapsed / self.units_run * left
        logger.info(
            "unit %s done (%d/%d) | %.1f s/unit | ETA %.1f min",
            unit, len(self.store.done), self.units_total, elapsed / self.units_run, eta / 60,
        )

    def measure(self, z: torch.Tensor, t: torch.Tensor, buffer: TensorBuffer):
        eps_ft, eps_base = predict_ft_base(
            self.model, z, t, self.ft_rows, self.base_rows, max_rows=self.config.diagnostics.max_rows
        )
        terms = diagnostic_tensors(eps_ft, eps_base, self.refine_keys)
        buffer.add(terms)
        return eps_ft, sample_metrics(terms, self.refine_keys)


def run_forward(ctx: Context, names: list[str], latents: list[torch.Tensor], timesteps: list[int]) -> None:
    d = ctx.config.diagnostics
    model = ctx.model
    for seed in d.forward_seeds:
        for start in range(0, len(names), d.batch_size):
            unit = f"forward/seed{seed}/{start:04d}"
            if ctx.store.is_done(unit):
                continue
            batch = names[start : start + d.batch_size]
            x0 = torch.stack(latents[start : start + d.batch_size]).to(model.device)
            noise = torch.stack([forward_noise(seed, name, tuple(x0.shape[1:])) for name in batch]).to(model.device)
            records, buffer = [], TensorBuffer(ctx.tensor_names)
            for t in timesteps:
                t_vec = torch.full((len(batch),), t, dtype=torch.long, device=model.device)
                z = model.add_noise(x0, noise, t_vec).to(model.weight_dtype)
                _, metrics = ctx.measure(z, t_vec, buffer)
                for name, m in zip(batch, metrics):
                    records.append({"source": "forward", "unit": unit, "image": name, "seed": seed, "t": t, **m})
            meta = {"unit": unit, "source": "forward", "seed": seed, "t": timesteps, "samples": batch}
            ctx.store.commit(unit, records, buffer.stacked(), meta)
            ctx.finished(unit)


def run_trajectories(ctx: Context, texts: dict[str, str], out_dir: Path) -> None:
    d = ctx.config.diagnostics
    gen = ctx.config.generate
    model = ctx.model
    for key in d.trajectory_prompts:
        for seed in d.seeds:
            for start in range(0, d.images_per_seed, d.batch_size):
                unit = f"trajectory/{key}/seed{seed}/{start:04d}"
                if ctx.store.is_done(unit):
                    continue
                indices = list(range(start, min(start + d.batch_size, d.images_per_seed)))
                seeds = [image_seed(seed, i) for i in indices]
                records, buffer, steps = [], TensorBuffer(ctx.tensor_names), []

                def eps_fn(info, z, t, key=key, seed=seed, indices=indices, seeds=seeds, unit=unit):
                    eps_ft, metrics = ctx.measure(z, t, buffer)
                    steps.append(info.t)
                    for index, s, m in zip(indices, seeds, metrics):
                        records.append({
                            "source": "trajectory", "unit": unit, "prompt_key": key, "seed": seed,
                            "index": index, "image_seed": s, "step": info.index, "t": info.t,
                            "sigma": info.sigma, **m,
                        })
                    if key in ctx.ft_rows:
                        drive = eps_ft[key]
                    else:
                        drive = predict_rows(model, z, t, {key: ctx.drive_rows[key]}, max_rows=d.max_rows)[key]
                    return eps_ft[NULL_KEY], drive

                latents = euler_cfg_sample(
                    model, seeds, eps_fn,
                    num_inference_steps=gen.num_inference_steps,
                    guidance_scale=gen.guidance_scale,
                    guidance_rescale=gen.guidance_rescale,
                )
                if d.save_images:
                    image_dir = out_dir / "trajectory_images" / key / f"seed{seed}"
                    image_dir.mkdir(parents=True, exist_ok=True)
                    for index, image in zip(indices, decode_latents(model, latents)):
                        to_pil(image).save(image_dir / f"{index:04d}.png")
                meta = {
                    "unit": unit, "source": "trajectory", "prompt_key": key, "prompt": texts[key],
                    "seed": seed, "t": steps, "samples": seeds,
                }
                ctx.store.commit(unit, records, buffer.stacked(), meta)
                ctx.finished(unit)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def diagnose_run(
    config: Config,
    *,
    checkpoint: str | Path,
    out_name: str | None = None,
    sources: list[str] | None = None,
    resume: bool = False,
) -> Path:
    """Measure one checkpoint; returns the output directory (holding ``summary.json``)."""
    d = config.diagnostics
    sources = [s for s in ("forward", "trajectory") if s in (sources or d.sources)]
    checkpoint = resolve_path(checkpoint)
    out_dir = config.run_dir / (out_name or default_out_name(checkpoint))
    storage_info = preflight(config, out_dir, sources)

    # Everything that can fail cheaply fails before the model is loaded.
    ckpt_meta = read_checkpoint_meta(checkpoint)
    ft_prompts, base_prompts = load_prompt_set(config.prompts), base_prompt_set(config)
    keys = list(d.refine_keys)
    base_texts = row_texts(base_prompts, keys)
    check_base_prompts(ckpt_meta, base_texts)
    for key in d.trajectory_prompts:
        ft_prompts.prompt(key)  # unknown key -> KeyError now, not after an hour
    heldout = []
    if "forward" in sources:
        heldout = select_heldout_images(config.data, num=d.num_heldout, seed=d.heldout_seed, pinned=d.heldout_list)

    store = ProgressStore(out_dir, fingerprint(config, checkpoint, sources), resume=resume)
    save_config(config, out_dir / "config.yaml")
    names = relative_names(heldout, config.data.image_dir) if heldout else []
    if heldout:
        _fsync_write(out_dir / "heldout_images.txt", "".join(n + "\n" for n in names))

    started = time.time()
    with deterministic_mode(d.deterministic) as det:
        configure_backends(allow_tf32=config.perf.allow_tf32, cudnn_benchmark=False)
        device = resolve_device(config.model.device)
        model = load_sdxl(config.model, device=device)
        model.unet.eval()
        # Base rows before the fine-tuner is installed (TI modifies the text encoders).
        base_rows = encode_rows(model, base_texts)
        meta = load_finetuner(model, checkpoint)
        logger.info("Loaded fine-tuner %s from %s", meta.get("finetuner"), checkpoint)
        expand = lambda p: expand_ti_prompt(p, meta)  # noqa: E731
        ft_texts = row_texts(ft_prompts, keys, expand)
        ft_rows = encode_rows(model, ft_texts)
        drive_texts = {k: expand(ft_prompts.prompt(k)) for k in d.trajectory_prompts}
        drive_rows = {k: model.encode_prompts([v]) for k, v in drive_texts.items() if k not in ft_rows}

        from hcfg.diagnostics.storage import plan_counts

        counts = plan_counts(config, sources)
        ctx = Context(
            config=config, model=model, store=store, ft_rows=ft_rows, base_rows=base_rows,
            drive_rows=drive_rows, refine_keys=keys, tensor_names=tensor_names(config),
            units_total=sum(c["units"] for c in counts.values()),
        )
        grid = sampler_timesteps(model, config.generate.num_inference_steps)
        forward_t = [int(t) for t in d.timesteps] if d.timesteps is not None else grid
        if "forward" in sources:
            logger.info("forward source: %d held-out images x %d seeds x %d t", len(names), len(d.forward_seeds), len(forward_t))
            latents = heldout_latents(model, heldout)
            run_forward(ctx, names, latents, forward_t)
        if "trajectory" in sources:
            logger.info("trajectory source: %s x seeds %s x %d images", d.trajectory_prompts, d.seeds, d.images_per_seed)
            run_trajectories(ctx, drive_texts, out_dir)

    records = store.records()
    checks = _checks(records, ckpt_meta)
    summary_meta = {
        "run": config.name,
        "checkpoint": str(checkpoint),
        "finetuner": meta.get("finetuner"),
        "prompt_sets": {"ft": ft_prompts.name, "base": base_prompts.name},
        "prompts": {k: {"ft": ft_texts[k], "base": base_texts[k]} for k in base_texts},
        "drive_prompts": drive_texts,
        "uncond": config.model.uncond,
        "weight_dtype": str(model.weight_dtype),
        "autocast": None,
        "device": str(device),
        "determinism": det,
        "scheduler": type(model.sample_scheduler).__name__,
        "timestep_spacing": model.sample_scheduler.config.get("timestep_spacing"),
        "num_inference_steps": config.generate.num_inference_steps,
        "guidance_scale": config.generate.guidance_scale,
        "timesteps": {"forward": forward_t if "forward" in sources else None, "trajectory": grid},
        "heldout_images": names,
        "seeds": {"trajectory": d.seeds, "forward": d.forward_seeds},
        "versions": _versions(),
        "storage": storage_info,
        "elapsed_sec": round(time.time() - started, 1),
        "checks": checks,
    }
    summary = summarize(records, tensor_files=store.tensor_files(), meta=summary_meta)
    save_json(summary, out_dir / "summary.json")
    from hcfg.diagnostics.plot import plot_summaries

    plot_summaries([summary], out_dir)
    logger.info("Diagnostics written to %s", out_dir)
    return out_dir


def _checks(records: list[dict], ckpt_meta: dict) -> dict:
    values = [r["ms_diff_null"] for r in records if r["ms_diff_null"] is not None]
    max_dn = max(values) if values else None
    checks = {"max_ms_diff_null": max_dn}
    if ckpt_meta.get("format") == "hcfg-ti-v1":
        exact = max_dn == 0.0
        checks["ti_diff_null_exact_zero"] = exact
        if not exact:
            logger.error("Textual Inversion Diff_Null is not exactly 0 (max ms %.3e): check row layout/determinism", max_dn)
    return checks


def _versions() -> dict:
    import diffusers
    import transformers

    return {"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__}
