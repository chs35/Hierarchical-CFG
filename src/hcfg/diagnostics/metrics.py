"""Per-sample diagnostic scalars from the FT / Base ε of one (z_t, t).

``ms`` = mean square per element (docs/Implementation detail.md §7), so
``‖a‖/‖b‖ = sqrt(ms_a / ms_b)`` for same-shaped tensors. Everything is computed
in float64 from the raw (possibly bf16) ε.

=====================  ============================================
G_refine               ε(FT, c_refine) − ε(FT, c)
G_refine^Base          ε(Base, c_refine) − ε(Base, c)
G_domain               ε(FT, c) − ε(FT, ∅)
Diff_Null              ε(FT, ∅) − ε(Base, ∅)        (diagnostic only)
G_domain,hybrid        ε(FT, c) − ε(Base, ∅) = G_domain + Diff_Null
=====================  ============================================
"""

from __future__ import annotations

import math

import torch

from hcfg.data.prompts import DOMAIN_KEY
from hcfg.diagnostics.branches import NULL_KEY


def mean_sq(x: torch.Tensor) -> torch.Tensor:
    """``[B]`` mean square per element."""
    return x.double().pow(2).flatten(1).mean(1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[B]`` cosine similarity; NaN where either vector is exactly zero."""
    a, b = a.double().flatten(1), b.double().flatten(1)
    na, nb = a.norm(dim=1), b.norm(dim=1)
    out = (a * b).sum(1) / (na * nb)
    return torch.where((na == 0) | (nb == 0), torch.full_like(out, math.nan), out)


def diagnostic_tensors(
    eps_ft: dict[str, torch.Tensor], eps_base: dict[str, torch.Tensor], refine_keys: list[str]
) -> dict[str, torch.Tensor]:
    """float64 guidance terms: ``eps_ft_null``, ``eps_base_null``, ``diff_null``, ``g_domain``,
    ``g_domain_base``, ``g_domain_hybrid``, ``g_refine/<k>``, ``g_refine_base/<k>``."""
    f0, fc = eps_ft[NULL_KEY].double(), eps_ft[DOMAIN_KEY].double()
    b0, bc = eps_base[NULL_KEY].double(), eps_base[DOMAIN_KEY].double()
    out = {
        "eps_ft_null": f0,
        "eps_base_null": b0,
        "diff_null": f0 - b0,
        "g_domain": fc - f0,
        "g_domain_base": bc - b0,
        "g_domain_hybrid": fc - b0,
    }
    for key in refine_keys:
        out[f"g_refine/{key}"] = eps_ft[key].double() - fc
        out[f"g_refine_base/{key}"] = eps_base[key].double() - bc
    return out


def sample_metrics(terms: dict[str, torch.Tensor], refine_keys: list[str]) -> list[dict]:
    """One dict of scalars per sample (NaN cosines become ``None`` for JSON)."""
    ms = {name: mean_sq(terms[name]) for name in (
        "eps_ft_null", "eps_base_null", "diff_null", "g_domain", "g_domain_base", "g_domain_hybrid"
    )}
    rel = (ms["diff_null"] / ms["g_domain"]).sqrt()
    cos_dn_gd = cosine(terms["diff_null"], terms["g_domain"])
    cos_null = cosine(terms["eps_ft_null"], terms["eps_base_null"])
    refine = {
        key: (
            mean_sq(terms[f"g_refine/{key}"]),
            mean_sq(terms[f"g_refine_base/{key}"]),
            cosine(terms[f"g_refine/{key}"], terms[f"g_refine_base/{key}"]),
        )
        for key in refine_keys
    }
    rows = []
    for i in range(rel.shape[0]):
        row = {f"ms_{name}": _num(v[i]) for name, v in ms.items()}
        row["rel_diff_null"] = _num(rel[i])
        row["cos_diff_null_g_domain"] = _num(cos_dn_gd[i])
        row["cos_eps_null"] = _num(cos_null[i])
        row["refine"] = {
            key: {"ms_g_refine": _num(a[i]), "ms_g_refine_base": _num(b[i]), "cos_g_refine": _num(c[i])}
            for key, (a, b, c) in refine.items()
        }
        rows.append(row)
    return rows


def _num(x: torch.Tensor) -> float | None:
    value = float(x)
    return value if math.isfinite(value) else None
