r"""CA (Concept-Awareness) measurement and head selection for style CA-LoRA.

CA-LoRA (arXiv 2503.22172), per attention *head* unit θ (a row band of
``to_q/k/v`` or a column band of ``to_out.0``)::

    CA(θ) = E[ ‖∇_θ L_Concept‖ / ‖∇_θ L_Diff‖ ]
    L_Diff    = ‖ε(z_t, t, c) − ε‖²
    L_Concept = ‖ε(z_t, t, c) − sg[ε(z_t, t, c_aug)]‖²

``c_aug`` are the paper's style rewrites of the training prompt
(configs/ca_augmentations.json). Selection keeps the top ``fraction`` of head
units by CA (paper: t = 81, style axis, 2%). Output JSON uses the reference
repo's format (``names`` = ``<module>@h<i>``), so selections are interchangeable.

This module is CA-LoRA-specific: the ``concept``/``aug`` vocabulary is confined
here (docs/Implementation detail.md §1) and nothing in it is reused by the
hierarchical-CFG code.

Every sample shares one grad-enabled forward for ``L_Diff`` and all its
``L_Concept`` terms (``retain_graph``); only the ``c_aug`` forwards are extra and
they are no-grad. Measurement uses center crops (deterministic) and noise that
depends on the sample index only.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from hcfg.models.heads import Projection, head_unit_name
from hcfg.models.sdxl import SDXL, Cond
from hcfg.utils.device import autocast_context
from hcfg.utils.logging import get_logger
from hcfg.utils.seed import make_generator

logger = get_logger(__name__)


def head_norms(grad: torch.Tensor, proj: Projection) -> torch.Tensor:
    """Per-head Frobenius norms ``[H]`` of a weight-shaped gradient (fp32)."""
    g = grad.float()
    if proj.slice_dim == "row":
        return g.view(proj.num_heads, proj.head_dim, -1).pow(2).sum(dim=(1, 2)).sqrt()
    return g.view(g.shape[0], proj.num_heads, proj.head_dim).pow(2).sum(dim=(0, 2)).sqrt()


def unit_names(projections: Sequence[Projection]) -> list[str]:
    return [head_unit_name(p.name, h) for p in projections for h in range(p.num_heads)]


def measure_ca(
    model: SDXL,
    projections: Sequence[Projection],
    latents: Sequence[torch.Tensor],
    cond: Cond,
    aug_conds: Sequence[Cond],
    *,
    timestep: int,
    num_samples: int,
    augs_per_sample: int,
    seed: int = 0,
    amp: torch.dtype | None = None,
    eps: float = 1e-12,
    log_every: int = 16,
) -> torch.Tensor:
    """CA per head unit, ``[U]`` in :func:`unit_names` order (fp64, CPU)."""
    if not 0 <= timestep < model.num_train_timesteps:
        raise ValueError(f"timestep {timestep} outside [0, {model.num_train_timesteps})")
    if not 1 <= augs_per_sample <= len(aug_conds):
        raise ValueError(f"augs_per_sample={augs_per_sample} but {len(aug_conds)} templates")
    unet, device = model.unet, model.device
    weights = [unet.get_submodule(p.name).weight for p in projections]
    grad_state = {n: p.requires_grad for n, p in unet.named_parameters()}
    unet.requires_grad_(False)
    for w in weights:
        w.requires_grad_(True)
    unet.eval()

    ca_sum = torch.zeros(sum(p.num_heads for p in projections), dtype=torch.float64, device=device)
    count = 0
    t = torch.full((1,), timestep, device=device, dtype=torch.long)
    started = time.perf_counter()
    try:
        for i in range(num_samples):
            z0 = latents[i % len(latents)].unsqueeze(0).to(device)
            gen = make_generator("ca", seed, i)
            noise = torch.randn(z0.shape, generator=gen).to(device)
            picks = torch.randperm(len(aug_conds), generator=gen)[:augs_per_sample].tolist()
            noisy = model.add_noise(z0, noise, t).to(model.weight_dtype)
            with autocast_context(amp, device):
                pred = model.predict(noisy, t, cond)
            loss_d = F.mse_loss(pred.float(), noise.float())
            grads = torch.autograd.grad(loss_d, weights, retain_graph=True)
            norm_d = torch.cat([head_norms(g, p) for g, p in zip(grads, projections)]).double()
            del grads
            for j, a in enumerate(picks):
                with torch.no_grad(), autocast_context(amp, device):
                    pred_aug = model.predict(noisy, t, aug_conds[a])
                loss_c = F.mse_loss(pred.float(), pred_aug.float())
                grads = torch.autograd.grad(loss_c, weights, retain_graph=j < len(picks) - 1)
                norm_c = torch.cat([head_norms(g, p) for g, p in zip(grads, projections)]).double()
                del grads
                ca_sum += norm_c / norm_d.clamp_min(eps)
                count += 1
            if log_every and (i + 1) % log_every == 0:
                logger.info("CA t=%d: %d/%d samples (%.1fs)", timestep, i + 1, num_samples, time.perf_counter() - started)
    finally:
        for name, param in unet.named_parameters():
            param.requires_grad_(grad_state[name])
    return (ca_sum / max(count, 1)).cpu()


def select_top_heads(
    projections: Sequence[Projection],
    scores: torch.Tensor,
    *,
    fraction: float,
    meta: dict | None = None,
) -> dict:
    """Top-``fraction`` head units by CA (unit-count budget, as in the paper)."""
    names = unit_names(projections)
    if scores.numel() != len(names):
        raise ValueError(f"{scores.numel()} scores for {len(names)} units")
    k = max(1, round(fraction * len(names)))
    top = torch.topk(scores, k).indices.sort().values.tolist()
    head_numel = [p.head_numel for p in projections for _ in range(p.num_heads)]
    kinds = [p.kind for p in projections for _ in range(p.num_heads)]
    attn = [f"attn{p.attn}" for p in projections for _ in range(p.num_heads)]
    breakdown = {"kind": {}, "attn": {}}
    for i in top:
        breakdown["kind"][kinds[i]] = breakdown["kind"].get(kinds[i], 0) + 1
        breakdown["attn"][attn[i]] = breakdown["attn"].get(attn[i], 0) + 1
    return {
        "strategy": "single",
        "granularity": "head",
        "num_projections": k,
        "num_modules": len({names[i].rsplit("@h", 1)[0] for i in top}),
        "units_total": len(names),
        "fraction_requested": fraction,
        "unit_fraction_achieved": k / len(names),
        "params_selected": sum(head_numel[i] for i in top),
        "params_total": sum(head_numel),
        "breakdown": breakdown,
        "meta": meta or {},
        "names": [names[i] for i in top],
        "scores": [float(scores[i]) for i in top],
    }
