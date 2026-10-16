r"""One training loop for all five fine-tuners.

Per step (ε-prediction, DDPM forward process, uniform t ∈ [0, 1000))::

    z0 ~ latent cache (random latent-space crop / cached flip)
    z_t = sqrt(ᾱ_t) z0 + sqrt(1-ᾱ_t) ε
    L = ‖ε_θ(z_t, t, c) − ε‖²                                   (task)
      + w_prior ‖ε_θ(z'_t, t, c_prior) − ε'‖²                   (DreamBooth prior preservation)
      + λ ‖ε_θ(z_t, t, ∅) − sg[ε_Base(z_t, t, ∅)]‖²             (optional ∅-anchoring)

``c`` is replaced by ∅ with probability ``cond_dropout`` (CA-LoRA + condition
dropout control); that ∅ is the very tensor used at inference
(:meth:`SDXL.null_cond`). All rows that share the trainable forward are packed
into **one** UNet call.

A100 throughput (docs/REPRODUCE.md, performance notes): frozen weights in bf16 + bf16 autocast
(tensor cores), TF32 on, fp32 master weights for the trainable parameters, no
gradient checkpointing, latents cached once, fused AdamW, no host sync except at
log steps. Step time is logged separately from checkpoint time.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from hcfg.config import Config, resolve_path, save_config
from hcfg.data.images import list_images, training_image_paths
from hcfg.data.latent_cache import LatentCache
from hcfg.data.prompts import load_prompt_set
from hcfg.generation.generate import generate_images
from hcfg.models.finetuners import attach_finetuner, expand_ti_prompt, save_finetuner
from hcfg.models.lora import disable_adapter
from hcfg.models.sdxl import SDXL, Cond, load_sdxl
from hcfg.utils.device import autocast_context, autocast_dtype, configure_backends, resolve_device
from hcfg.utils.io import save_json
from hcfg.utils.logging import get_logger
from hcfg.utils.seed import make_generator, seed_everything

logger = get_logger(__name__)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def _lr_lambda(config: Config):
    warmup = config.train.lr_warmup_steps
    total = config.train.max_steps
    kind = config.train.lr_scheduler

    def fn(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        if kind == "cosine":
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        if kind == "linear":
            return max(0.0, 1 - (step - warmup) / max(1, total - warmup))
        raise ValueError(f"Unknown lr_scheduler {kind!r}")

    return fn


def ensure_class_images(model: SDXL, config: Config, prompt: str) -> list[Path]:
    """DreamBooth prior images from the *Base* model (call before attaching adapters)."""
    db = config.finetuner.dreambooth
    directory = resolve_path(db.class_data_dir) if db.class_data_dir else (
        resolve_path(config.output_dir) / f"dreambooth_class_images_{model.resolution}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("*.png"))
    if len(existing) >= db.num_class_images:
        logger.info("Using %d existing class images in %s", db.num_class_images, directory)
        return existing[: db.num_class_images]
    logger.info("Generating %d class images for %r -> %s", db.num_class_images, prompt, directory)
    records = generate_images(
        model,
        prompt,
        num_images=db.num_class_images,
        seed=config.train.seed,
        out_dir=directory,
        prefix="class_",
        num_inference_steps=db.class_num_inference_steps,
        guidance_scale=db.class_guidance_scale,
        batch_size=db.class_batch_size,
    )
    save_json({"prompt": prompt, "records": records}, directory / "manifest.json")
    return sorted(directory.glob("*.png"))[: db.num_class_images]


def _offload_text_encoders(model: SDXL) -> None:
    for encoder in model.text_encoders:
        encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(
    config: Config,
    *,
    image_paths: list[Path] | None = None,
    step_callback: Callable[[int], None] | None = None,
) -> dict:
    """Train the configured fine-tuner; returns the run summary.

    ``step_callback(step)`` runs after every optimizer step (e.g. ``profiler.step``
    in scripts/gpu_check.py); it is not called during checkpointing.
    """
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    seed_everything(config.train.seed)
    configure_backends(allow_tf32=config.perf.allow_tf32, cudnn_benchmark=config.perf.cudnn_benchmark)

    device = resolve_device(config.model.device)
    model = load_sdxl(config.model, device=device)
    amp = autocast_dtype(config.perf.mixed_precision, device)
    kind = config.finetuner.kind
    prompts = load_prompt_set(config.prompts)
    tcfg = config.train
    batch = tcfg.batch_size

    # ---- data: latents cached once -----------------------------------------------------
    paths = image_paths if image_paths is not None else training_image_paths(config.data)
    cache = LatentCache.build(model, paths, hflip=config.data.hflip)
    prior_cache = None
    if kind == "dreambooth_lora":
        if not prompts.prior:
            raise ValueError("DreamBooth needs a 'prior' prompt in the prompt set")
        class_paths = ensure_class_images(model, config, prompts.prior)
        prior_cache = LatentCache.build(model, class_paths, hflip=config.data.hflip)
    model.vae.to("cpu")  # no VAE work inside the loop

    # ---- fine-tuner ----------------------------------------------------------------------
    params = attach_finetuner(model, config)
    num_trainable = sum(p.numel() for p in params)
    logger.info("%s: %d trainable tensors, %.2fM params", kind, len(params), num_trainable / 1e6)

    domain_prompt = prompts.domain
    if kind == "textual_inversion":
        domain_prompt = expand_ti_prompt(domain_prompt, config)
    logger.info("Training prompt c = %r", domain_prompt)

    cond_c = cond_prior = None
    if kind != "textual_inversion":
        cond_c = model.encode_prompts([domain_prompt]).repeat(batch)
        if prior_cache is not None:
            cond_prior = model.encode_prompts([prompts.prior]).repeat(batch)
        null = model.null_cond(batch)  # before offloading (uncond='empty' encodes "")
        _offload_text_encoders(model)
    else:
        null = model.null_cond(batch)

    # ---- performance knobs ---------------------------------------------------------------
    unet = model.unet
    if config.perf.channels_last and device.type == "cuda":
        unet.to(memory_format=torch.channels_last)
    if config.perf.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()
    if config.perf.torch_compile:
        model.unet_fn = torch.compile(unet)

    optimizer_kwargs = dict(
        lr=tcfg.learning_rate,
        betas=(tcfg.adam_beta1, tcfg.adam_beta2),
        weight_decay=tcfg.adam_weight_decay,
        eps=tcfg.adam_epsilon,
    )
    if config.perf.fused_optimizer and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(config))

    cpu_gen = make_generator("data", tcfg.seed)
    dev_gen = make_generator("noise", tcfg.seed, device=device)
    anchor = tcfg.phi_anchor_lambda
    log_path = run_dir / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    history: list[dict] = []
    checkpoint_time = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logger.info(
        "Start: %d steps, batch %d, lr %.2e, cond_dropout %.2f, ∅-anchor λ %.3g, autocast %s",
        tcfg.max_steps, batch, tcfg.learning_rate, tcfg.cond_dropout, anchor, amp,
    )
    loop_start = window_start = time.perf_counter()
    last_logged = 0
    for step in range(1, tcfg.max_steps + 1):
        # z0 and the rows of the single trainable forward
        latents = cache.sample(batch, crop=config.data.crop, hflip=config.data.hflip, generator=cpu_gen)
        latents = latents.to(device, non_blocking=True)
        if prior_cache is not None:
            prior = prior_cache.sample(batch, crop="center", hflip=config.data.hflip, generator=cpu_gen)
            latents = torch.cat([latents, prior.to(device, non_blocking=True)])
        noise = torch.randn(latents.shape, generator=dev_gen, device=device)
        timesteps = torch.randint(0, model.num_train_timesteps, (latents.shape[0],), generator=dev_gen, device=device)
        noisy = model.add_noise(latents, noise, timesteps).to(model.weight_dtype)

        with autocast_context(amp, device):
            if kind == "textual_inversion":
                cond = model.encode_prompts([domain_prompt] * batch, grad=True)
            else:
                cond = cond_c
            if tcfg.cond_dropout > 0:
                drop = torch.rand(batch, generator=dev_gen, device=device) < tcfg.cond_dropout
                cond = Cond.where(drop, null, cond)
            rows_cond = [cond]
            rows_noisy = [noisy]
            rows_t = [timesteps]
            if cond_prior is not None:
                rows_cond.append(cond_prior)
            if anchor > 0:  # ∅ branch on the *training-data* z_t
                rows_cond.append(null)
                rows_noisy.append(noisy[:batch])
                rows_t.append(timesteps[:batch])
            all_noisy = torch.cat(rows_noisy)
            if config.perf.channels_last and device.type == "cuda":
                all_noisy = all_noisy.contiguous(memory_format=torch.channels_last)
            pred = model.predict(all_noisy, torch.cat(rows_t), Cond.cat(rows_cond))

            loss_task = _mse(pred[:batch], noise[:batch])
            loss = loss_task
            terms = {"loss_task": loss_task}
            offset = batch
            if cond_prior is not None:
                loss_prior = _mse(pred[offset : offset + batch], noise[batch : 2 * batch])
                loss = loss + config.finetuner.dreambooth.prior_loss_weight * loss_prior
                terms["loss_prior"] = loss_prior
                offset += batch
            if anchor > 0:
                with torch.no_grad(), disable_adapter(model.unet):
                    base_phi = model.predict(noisy[:batch], timesteps[:batch], null)
                loss_anchor = _mse(pred[offset : offset + batch], base_phi)
                loss = loss + anchor * loss_anchor
                terms["loss_anchor"] = loss_anchor

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tcfg.max_grad_norm) if tcfg.max_grad_norm > 0 else None
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step_callback is not None:
            step_callback(step)

        if step % tcfg.log_every == 0 or step == 1 or step == tcfg.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            steps_in_window = step - last_logged
            last_logged = step
            record = {
                "step": step,
                "loss": float(loss.detach()),
                **{k: float(v.detach()) for k, v in terms.items()},
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "lr": scheduler.get_last_lr()[0],
                "sec_per_step": (now - window_start) / steps_in_window,
            }
            history.append(record)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            logger.info(
                "step %d/%d loss %.4f (task %.4f) grad %.3f lr %.2e | %.3f s/step",
                step, tcfg.max_steps, record["loss"], record["loss_task"],
                record["grad_norm"] or 0.0, record["lr"], record["sec_per_step"],
            )
            window_start = time.perf_counter()

        if tcfg.checkpoint_every and step % tcfg.checkpoint_every == 0 and step != tcfg.max_steps:
            ck_start = time.perf_counter()
            save_finetuner(model, config, run_dir / f"checkpoint-{step:06d}.safetensors", extra={"step": step})
            elapsed = time.perf_counter() - ck_start
            checkpoint_time += elapsed
            window_start += elapsed  # keep s/step a pure training number

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - loop_start - checkpoint_time
    final = run_dir / "finetuner-final.safetensors"
    save_finetuner(model, config, final, extra={"step": tcfg.max_steps})
    summary = {
        "run": config.name,
        "finetuner": kind,
        "final_checkpoint": str(final),
        "steps": tcfg.max_steps,
        "trainable_params": num_trainable,
        "train_prompt": domain_prompt,
        "prior_prompt": prompts.prior if prior_cache is not None else None,
        "num_train_images": len(paths),
        "train_images": [str(p) for p in paths],
        "train_time_sec": train_time,
        "checkpoint_time_sec": checkpoint_time,
        "sec_per_step_mean": train_time / max(tcfg.max_steps, 1),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
        "final_loss": history[-1]["loss"] if history else None,
        "autocast": str(amp),
        "weight_dtype": str(model.weight_dtype),
        "uncond": config.model.uncond,
    }
    save_json(summary, run_dir / "train_summary.json")
    _plot_loss(history, run_dir / "loss_curve.png")
    logger.info("Done: %s", json.dumps({k: summary[k] for k in ("steps", "train_time_sec", "sec_per_step_mean")}))
    return summary


def _plot_loss(history: list[dict], path: Path) -> None:
    if len(history) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    steps = [h["step"] for h in history]
    for key in ("loss_task", "loss_prior", "loss_anchor"):
        if key in history[0]:
            ax.plot(steps, [h[key] for h in history], label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


__all__ = ["train", "ensure_class_images", "list_images"]
