"""Per-t curves, pooled ratios and seed consistency from ``records.jsonl`` (numpy only).

All curves are aligned to ``t`` in sampling order (descending). ``pooled`` ratios
average the mean squares over samples before dividing (docs/Implementation detail.md
§7: mean square per element, averaged over samples): ``rel_diff_null = sqrt(E[ms Diff_Null] / E[ms G_domain])`` and
``g_refine_ratio[k] = E[ms G_refine] / E[ms G_refine^Base]`` (diagnostic 1).

Seed consistency (H1 vs H3, docs/diagnostics.md §3) is reported as data, never as a verdict:
per-seed curves with their spread, and — when the unit tensors were saved — the
mean pairwise cosine of Diff_Null across samples (``cross_cos``) next to the same
quantity for G_domain as the reference for a systematic direction.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SAMPLE_METRICS = (
    "ms_eps_ft_null",
    "ms_eps_base_null",
    "ms_diff_null",
    "ms_g_domain",
    "ms_g_domain_base",
    "ms_g_domain_hybrid",
    "rel_diff_null",
    "cos_diff_null_g_domain",
    "cos_eps_null",
)
REFINE_METRICS = ("ms_g_refine", "ms_g_refine_base", "cos_g_refine")
SEED_METRICS = ("ms_diff_null", "rel_diff_null", "cos_diff_null_g_domain", "cos_eps_null")
CROSS_NAMES = ("diff_null", "g_domain")


def _values(records: list[dict], getter) -> np.ndarray:
    return np.array([np.nan if (v := getter(r)) is None else v for r in records], dtype=np.float64)


def _stat(values: np.ndarray, fn) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(fn(finite))


def _ratio(num: float | None, den: float | None, sqrt: bool = False) -> float | None:
    if num is None or den is None or den == 0:
        return None
    value = num / den
    return math.sqrt(value) if sqrt else value


def _tmean(curve: list[float | None]) -> float | None:
    values = [v for v in curve if v is not None]
    return float(np.mean(values)) if values else None


def _group_by_t(records: list[dict]) -> tuple[list[int], dict[int, list[dict]]]:
    by_t: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_t[record["t"]].append(record)
    return sorted(by_t, reverse=True), by_t


def summarize_group(records: list[dict]) -> dict:
    ts, by_t = _group_by_t(records)
    refine_keys = list(records[0]["refine"]) if records else []
    out: dict = {"t": ts, "n": [len(by_t[t]) for t in ts], "metrics": {}, "refine": {}, "pooled": {}}
    for name in SAMPLE_METRICS:
        values = [_values(by_t[t], lambda r, n=name: r[n]) for t in ts]
        out["metrics"][name] = {
            "mean": [_stat(v, np.mean) for v in values],
            "std": [_stat(v, np.std) for v in values],
        }
    for key in refine_keys:
        out["refine"][key] = {}
        for name in REFINE_METRICS:
            values = [_values(by_t[t], lambda r, k=key, n=name: r["refine"][k][n]) for t in ts]
            out["refine"][key][name] = {
                "mean": [_stat(v, np.mean) for v in values],
                "std": [_stat(v, np.std) for v in values],
            }
    ms = out["metrics"]
    out["pooled"]["rel_diff_null"] = [
        _ratio(a, b, sqrt=True) for a, b in zip(ms["ms_diff_null"]["mean"], ms["ms_g_domain"]["mean"])
    ]
    out["pooled"]["g_refine_ratio"] = {
        key: [
            _ratio(a, b)
            for a, b in zip(out["refine"][key]["ms_g_refine"]["mean"], out["refine"][key]["ms_g_refine_base"]["mean"])
        ]
        for key in refine_keys
    }
    out["headline"] = {
        "rel_diff_null": _tmean(out["pooled"]["rel_diff_null"]),
        "ms_diff_null": _tmean(ms["ms_diff_null"]["mean"]),
        "cos_diff_null_g_domain": _tmean(ms["cos_diff_null_g_domain"]["mean"]),
        "cos_eps_null": _tmean(ms["cos_eps_null"]["mean"]),
        "g_refine_ratio": {k: _tmean(v) for k, v in out["pooled"]["g_refine_ratio"].items()},
        "cos_g_refine": {k: _tmean(out["refine"][k]["cos_g_refine"]["mean"]) for k in refine_keys},
    }
    return out


def seed_consistency(records: list[dict]) -> dict:
    """Per-seed mean curves and their spread across seeds (std, cv = std/|mean|)."""
    ts, _ = _group_by_t(records)
    seeds = sorted({r["seed"] for r in records})
    out: dict = {"t": ts, "seeds": seeds, "metrics": {}}
    for name in SEED_METRICS:
        curves = {}
        for seed in seeds:
            _, by_t = _group_by_t([r for r in records if r["seed"] == seed])
            curves[seed] = [_stat(_values(by_t.get(t, []), lambda r, n=name: r[n]), np.mean) for t in ts]
        spread_mean, spread_std, cv = [], [], []
        for j in range(len(ts)):
            values = np.array([c[j] for c in curves.values() if c[j] is not None], dtype=np.float64)
            if values.size < 2:
                spread_mean.append(_stat(values, np.mean))
                spread_std.append(None)
                cv.append(None)
                continue
            mean, std = float(values.mean()), float(values.std())
            spread_mean.append(mean)
            spread_std.append(std)
            cv.append(std / abs(mean) if mean != 0 else None)
        out["metrics"][name] = {
            "seed_means": {str(s): c for s, c in curves.items()},
            "mean": spread_mean,
            "std": spread_std,
            "cv": cv,
            "cv_t_mean": _tmean(cv),
        }
    return out


def cross_sample_cos(tensor_files: list[Path], source: str) -> dict[str, dict]:
    """Mean pairwise cosine across samples, per group and t, for Diff_Null and G_domain.

    Exact from running sums of unit vectors: ``(‖Σu‖² − n) / (n(n − 1))``. Exactly
    zero vectors (Textual Inversion's Diff_Null) are skipped.
    """
    from safetensors import safe_open

    sums: dict = {}
    counts: dict = defaultdict(int)
    t_order: dict[str, list[int]] = {}
    for path in tensor_files:
        with safe_open(str(path), framework="pt") as handle:
            meta = handle.metadata() or {}
            if meta.get("source") != source:
                continue
            available = set(handle.keys())
            tensors = {}
            for name in CROSS_NAMES:
                if name in available:
                    tensors[name] = handle.get_tensor(name)
            if "diff_null" not in tensors and {"eps_ft_null", "eps_base_null"} <= available:
                tensors["diff_null"] = handle.get_tensor("eps_ft_null").double() - handle.get_tensor("eps_base_null").double()
        ts = json.loads(meta["t"])
        groups = ["all"] + ([meta["prompt_key"]] if meta.get("prompt_key") else [])
        for name, x in tensors.items():
            x = x.double().flatten(2)  # [N, T, D]
            norms = x.norm(dim=2, keepdim=True)
            units = torch.where(norms > 0, x / norms.clamp_min(1e-300), torch.zeros_like(x))
            nonzero = (norms.squeeze(2) > 0).sum(0)  # [T]
            for group in groups:
                t_order.setdefault(group, ts)
                for j, t in enumerate(ts):
                    key = (group, name, t)
                    total = units[:, j].sum(0)
                    sums[key] = total if key not in sums else sums[key] + total
                    counts[key] += int(nonzero[j])
    out: dict[str, dict] = {}
    for group, ts in t_order.items():
        ts = sorted(ts, reverse=True)
        entry = {"t": ts}
        for name in CROSS_NAMES:
            values = []
            for t in ts:
                n = counts.get((group, name, t), 0)
                if n < 2:
                    values.append(None)
                    continue
                s = sums[(group, name, t)]
                values.append(float((s.dot(s) - n) / (n * (n - 1))))
            entry[name] = values
        out[group] = entry
    return out


def summarize(records: list[dict], *, tensor_files: list[Path] | None = None, meta: dict | None = None) -> dict:
    out: dict = {"meta": meta or {}, "sources": {}}
    for source in ("forward", "trajectory"):
        recs = [r for r in records if r["source"] == source]
        if not recs:
            continue
        groups = {"all": recs}
        if source == "trajectory":
            for key in dict.fromkeys(r["prompt_key"] for r in recs):
                groups[key] = [r for r in recs if r["prompt_key"] == key]
        entry = {
            "groups": {name: summarize_group(rs) for name, rs in groups.items()},
            "seed_consistency": {name: seed_consistency(rs) for name, rs in groups.items()},
        }
        if tensor_files:
            entry["cross_cos"] = cross_sample_cos(tensor_files, source)
        out["sources"][source] = entry
    return out
