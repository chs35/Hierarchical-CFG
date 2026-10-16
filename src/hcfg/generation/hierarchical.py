"""Hierarchical CFG (report §2.3, docs/Implementation detail.md §3).

All branches are ε-predictions at the same (z_t, t); w1 = refine strength,
w2 = domain strength, diffusers guidance convention (w = 1: no guidance)::

    basic       ε(FT,∅)   + w1·[ε(FT,c_r) − ε(FT,c)]     + w2·[ε(FT,c) − ε(FT,∅)]
    alt_domain  ε(Base,∅) + w1·[ε(FT,c_r) − ε(FT,c)]     + w2·[ε(FT,c) − ε(Base,∅)]
    alt_refine  ε(FT,∅)   + w1·[ε(Base,c_r) − ε(Base,c)] + w2·[ε(FT,c) − ε(FT,∅)]

Where w1(t) = w2(t) = w, ``basic`` reduces to standard CFG on c_refine and
``alt_domain`` to ε(Base,∅) + w·[ε(FT,c_r) − ε(Base,∅)] (Unconditional Priors
Matter). With ``reduce_equal_weights`` those steps are computed in reduced form
(report §2.3): ``basic`` then uses the pipeline's own [∅ × B, c_r × B] batch and
guidance line, so the w1 = w2 diagonal of a grid *is* standard CFG, bit for bit.
``alt_refine`` has no reduction.

Base branches run on the same UNet with the adapter off (``disable_adapter``);
only ``basic`` may use a merged LoRA.

Diff_Null split (report §7): Diff_Null = ε(FT,∅) − ε(Base,∅)
is split by Gram-Schmidt on r = Diff_cond = ε(FT,c) − ε(Base,c), per sample, into
Diff_∥ = (⟨Diff_Null, r⟩ / ‖r‖²)·r and Diff_⊥ = Diff_Null − Diff_∥, and sampled as::

    ε(Base,∅) + w1·G_refine + w2·G_domain + a·Diff_∥ + λ·Diff_⊥

``basic`` (a = 1) and ``alt_domain`` (a = w2) are the two members with λ = 1 / λ = w2;
λ = 0 removes Diff_⊥. Every split step runs FT [∅, c, c_r] + Base [∅, c] (no reduction).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hcfg.config import HCFG_VARIANTS, SCHEDULE_KINDS
from hcfg.generation.sampling import StepInfo, cfg_combine, euler_sample, sampler_timesteps
from hcfg.models.lora import disable_adapter
from hcfg.models.sdxl import SDXL, Cond


@dataclass(frozen=True)
class WeightSchedule:
    """w(t): constant, or ``end + (start − end)·(t / t_first)^p`` with p = 1 (linear) / 2 (quadratic)."""

    start: float
    kind: str = "constant"
    end: float = 1.0

    def __post_init__(self):
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(f"Unknown schedule {self.kind!r}; choose from {SCHEDULE_KINDS}")

    def __call__(self, t: float, t_first: float) -> float:
        if self.kind == "constant":
            return float(self.start)
        s = min(max(float(t) / float(t_first), 0.0), 1.0)
        power = 1 if self.kind == "linear" else 2
        return float(self.end + (self.start - self.end) * s**power)


@dataclass
class HCFGConds:
    """1-row conditions. ``base_*`` (Base-prompt c / c_refine): ``alt_refine``; ``base_domain`` also the split."""

    null: Cond
    domain: Cond
    refine: Cond
    base_domain: Cond | None = None
    base_refine: Cond | None = None


def _rows(model: SDXL, z: torch.Tensor, t: torch.Tensor, conds: list[Cond]) -> list[torch.Tensor]:
    """ε of each condition at the same (z, t), one UNet call, layout [c0 × B, c1 × B, ...]."""
    batch = z.shape[0]
    eps = model.predict(
        z.repeat(len(conds), 1, 1, 1), t.repeat(len(conds)), Cond.cat([c.repeat(batch) for c in conds])
    )
    return list(eps.split(batch))


def combine(variant: str, w1: float, w2: float, eps: dict[str, torch.Tensor]) -> torch.Tensor:
    """The three formulas, in float32 (cast back by the caller).

    ``eps`` keys: ``ft_null``, ``ft_domain``, ``ft_refine``, ``base_null``,
    ``base_domain``, ``base_refine`` (only those the variant needs).
    """
    e = {k: v.float() for k, v in eps.items()}
    if variant == "basic":
        return e["ft_null"] + w1 * (e["ft_refine"] - e["ft_domain"]) + w2 * (e["ft_domain"] - e["ft_null"])
    if variant == "alt_domain":
        return e["base_null"] + w1 * (e["ft_refine"] - e["ft_domain"]) + w2 * (e["ft_domain"] - e["base_null"])
    if variant == "alt_refine":
        return e["ft_null"] + w1 * (e["base_refine"] - e["base_domain"]) + w2 * (e["ft_domain"] - e["ft_null"])
    raise ValueError(f"Unknown variant {variant!r}; choose from {HCFG_VARIANTS}")


def split_diff_null(diff_null: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(Diff_∥, Diff_⊥) of ``diff_null`` against ``r``, one projection per sample (float32)."""
    diff_null, r = diff_null.float(), r.float()
    dims = tuple(range(1, r.ndim))
    coef = (diff_null * r).sum(dims, keepdim=True) / (r * r).sum(dims, keepdim=True).clamp_min(1e-8)
    par = coef * r
    return par, diff_null - par


def split_par_default(variant: str, w2: float) -> float:
    """a of the variant the split generalises: ``basic`` 1, ``alt_domain`` w2."""
    if variant == "basic":
        return 1.0
    if variant == "alt_domain":
        return float(w2)
    raise ValueError(f"The Diff_Null split is defined for basic / alt_domain, not {variant!r}")


def combine_split(variant: str, w1: float, w2: float, par: float, lam: float,
                  eps: dict[str, torch.Tensor]) -> torch.Tensor:
    """ε(Base,∅) + w1·G_refine + w2·G_domain + a·Diff_∥ + λ·Diff_⊥, in float32.

    Computed as the variant's own formula + (a − a_v)·Diff_∥ + (λ − a_v)·Diff_⊥ (a_v = λ_v =
    :func:`split_par_default`), so a = λ = a_v adds exact zeros and gives the unsplit
    variant bit for bit; summing from ε(Base,∅) instead differs in the last float32 bit, which
    flips bf16 roundings and grows over the steps (SDXL, w1 = 7: PSNR 33–42 dB).

    ``eps`` keys: ``ft_null``, ``ft_domain``, ``ft_refine``, ``base_null``, ``base_domain``.
    """
    e = {k: v.float() for k, v in eps.items()}
    d_par, d_perp = split_diff_null(e["ft_null"] - e["base_null"], e["ft_domain"] - e["base_domain"])
    own = split_par_default(variant, w2)
    return combine(variant, w1, w2, e) + (par - own) * d_par + (lam - own) * d_perp


def hcfg_guided_fn(
    model: SDXL,
    conds: HCFGConds,
    *,
    variant: str,
    w1: WeightSchedule,
    w2: WeightSchedule,
    t_first: float,
    guidance_rescale: float = 0.0,
    reduce_equal_weights: bool = True,
    split_lambda: float | None = None,
    split_par: float | None = None,
):
    """The guided ε of one sampling step, for :func:`~hcfg.generation.sampling.euler_sample`.

    ``split_lambda`` (λ) turns the Diff_Null split on (``basic`` / ``alt_domain``); ``split_par``
    (a) defaults to the variant's own value. Both are constant over t; a = w2(t) follows w2's schedule.
    """
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

    if variant not in HCFG_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {HCFG_VARIANTS}")
    if variant == "alt_refine" and (conds.base_domain is None or conds.base_refine is None):
        raise ValueError("alt_refine needs the Base-prompt conditions (base_domain, base_refine)")
    split = split_lambda is not None
    if split:
        split_par_default(variant, 1.0)  # basic / alt_domain only
        if conds.base_domain is None:
            raise ValueError("The Diff_Null split needs the Base-prompt condition base_domain (r = Diff_cond)")
    elif split_par is not None:
        raise ValueError("split_par without split_lambda")

    def guided_fn(info: StepInfo, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, b = w1(info.t, t_first), w2(info.t, t_first)
        reduce = reduce_equal_weights and a == b and variant != "alt_refine" and not split
        if reduce and variant == "basic":
            if a == 1.0:  # ε(FT, c_r) alone, like the pipeline at guidance_scale 1
                return _rows(model, z, t, [conds.refine])[0]
            eps_u, eps_c = _rows(model, z, t, [conds.null, conds.refine])
            return cfg_combine(eps_u, eps_c, a, guidance_rescale)
        if reduce:  # alt_domain: ε(Base,∅) + w·[ε(FT,c_r) − ε(Base,∅)]
            (ft_refine,) = _rows(model, z, t, [conds.refine])
            with disable_adapter(model.unet):
                (base_null,) = _rows(model, z, t, [conds.null])
            guided = base_null.float() + a * (ft_refine.float() - base_null.float())
            text = ft_refine
        elif split:  # FT rows laid out as the unsplit basic form's
            eps = dict(zip(("ft_null", "ft_domain", "ft_refine"), _rows(model, z, t, [conds.null, conds.domain, conds.refine])))
            with disable_adapter(model.unet):
                eps["base_null"], eps["base_domain"] = _rows(model, z, t, [conds.null, conds.base_domain])
            par = split_par_default(variant, b) if split_par is None else split_par
            guided = combine_split(variant, a, b, par, split_lambda, eps)
            text = eps["ft_refine"]
        else:
            eps: dict[str, torch.Tensor] = {}
            if variant == "basic":
                eps["ft_null"], eps["ft_domain"], eps["ft_refine"] = _rows(model, z, t, [conds.null, conds.domain, conds.refine])
            elif variant == "alt_domain":
                eps["ft_domain"], eps["ft_refine"] = _rows(model, z, t, [conds.domain, conds.refine])
                with disable_adapter(model.unet):
                    (eps["base_null"],) = _rows(model, z, t, [conds.null])
            else:
                eps["ft_null"], eps["ft_domain"] = _rows(model, z, t, [conds.null, conds.domain])
                with disable_adapter(model.unet):
                    eps["base_domain"], eps["base_refine"] = _rows(model, z, t, [conds.base_domain, conds.base_refine])
            guided = combine(variant, a, b, eps)
            text = eps.get("ft_refine", eps.get("base_refine"))
        if guidance_rescale > 0.0:  # diffusers rescale against the c_refine branch
            guided = rescale_noise_cfg(guided, text.float(), guidance_rescale=guidance_rescale)
        return guided.to(model.weight_dtype)

    return guided_fn


@torch.no_grad()
def hcfg_sample(
    model: SDXL,
    conds: HCFGConds,
    image_seeds: list[int],
    *,
    variant: str,
    w1: WeightSchedule,
    w2: WeightSchedule,
    num_inference_steps: int,
    guidance_rescale: float = 0.0,
    reduce_equal_weights: bool = True,
    split_lambda: float | None = None,
    split_par: float | None = None,
) -> torch.Tensor:
    """Final latents of hierarchical-CFG sampling; image i starts from ``image_seeds[i]``'s z_T."""
    t_first = sampler_timesteps(model, num_inference_steps)[0]
    guided_fn = hcfg_guided_fn(
        model, conds, variant=variant, w1=w1, w2=w2, t_first=t_first,
        guidance_rescale=guidance_rescale, reduce_equal_weights=reduce_equal_weights,
        split_lambda=split_lambda, split_par=split_par,
    )
    return euler_sample(model, image_seeds, guided_fn, num_inference_steps=num_inference_steps)
