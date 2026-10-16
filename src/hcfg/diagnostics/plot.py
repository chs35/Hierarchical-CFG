"""Figures of one or several diagnostics summaries (matplotlib, imported lazily).

One colour per run (fixed slots, the palette of the other report figures), line
style per z_t source: solid = sampling trajectory, dashed = held-out q(z_t|x0).
x = actual timestep, drawn from t = 999 (left) to 0, i.e. in sampling order.
"""

from __future__ import annotations

from pathlib import Path

from hcfg.utils.logging import get_logger

logger = get_logger(__name__)

RUN_COLORS = {
    "vanilla_lora": "#2a78d6",
    "ca_lora_style": "#eb6834",
    "dreambooth_lora": "#1baf7a",
    "textual_inversion": "#eda100",
    "ca_lora_style_dropout": "#e87ba4",
}
EXTRA_COLORS = ["#008300", "#4a3aa7", "#e34948"]
FALLBACK_COLOR = "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SOURCE_STYLES = {"trajectory": "-", "forward": "--"}
SOURCE_LABELS = {"trajectory": "sampling trajectory", "forward": "held-out q(z_t|x0)"}


def run_colors(runs: list[str]) -> dict[str, str]:
    """Known runs keep their colour; others take the free slots in order (never cycled)."""
    extra = iter(EXTRA_COLORS)
    return {run: RUN_COLORS.get(run) or next(extra, FALLBACK_COLOR) for run in runs}


def _positive(values):
    return [v if v is not None and v > 0 else float("nan") for v in values]


def _clean(values):
    return [float("nan") if v is None else v for v in values]


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("t", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlim(1000, 0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)


def _legend(fig, colors: dict[str, str], sources: list[str], extra: list | None = None) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, linewidth=2, label=run) for run, c in colors.items()]
    handles += [Line2D([], [], color=MUTED, linestyle=SOURCE_STYLES[s], linewidth=1.5, label=SOURCE_LABELS[s]) for s in sources]
    handles += extra or []
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=8, labelcolor=INK)


def _group(summary: dict, source: str) -> dict | None:
    return summary["sources"].get(source, {}).get("groups", {}).get("all")


def plot_summaries(summaries: list[dict], out_dir: str | Path, *, labels: list[str] | None = None) -> list[Path]:
    """Write ``diff_null.png``, ``g_refine_norm.png``, ``cos_g_refine.png``, ``consistency.png``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping diagnostics plots")
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or [s["meta"].get("run", f"run{i}") for i, s in enumerate(summaries)]
    colors = run_colors(labels)
    sources = [s for s in SOURCE_STYLES if any(s in x["sources"] for x in summaries)]
    written = []

    def save(fig, name):
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        path = out_dir / name
        fig.savefig(path, dpi=130, facecolor="white")
        plt.close(fig)
        written.append(path)

    # 1. Diff_Null scale and direction (report §3.2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        (axes[0, 0], "‖Diff_Null‖ / ‖G_domain‖ (pooled)", "ratio (log)", lambda g: g["pooled"]["rel_diff_null"], True),
        (axes[0, 1], "cos(Diff_Null, G_domain)", "cosine", lambda g: g["metrics"]["cos_diff_null_g_domain"]["mean"], False),
        (axes[1, 0], "‖Diff_Null‖² (mean square per element)", "ms (log)", lambda g: g["metrics"]["ms_diff_null"]["mean"], True),
        (axes[1, 1], "cos(ε(FT,∅), ε(Base,∅))", "cosine", lambda g: g["metrics"]["cos_eps_null"]["mean"], False),
    ]
    zero_runs = set()
    for ax, title, ylabel, get, log in panels:
        for label, summary in zip(labels, summaries):
            for source in sources:
                group = _group(summary, source)
                if group is None:
                    continue
                values = get(group)
                if log and all(v in (None, 0) for v in values):
                    zero_runs.add(label)
                ys = _positive(values) if log else _clean(values)
                ax.plot(group["t"], ys, color=colors[label], linestyle=SOURCE_STYLES[source], linewidth=1.5)
        if log:
            ax.set_yscale("log")
        _style_axes(ax, title, ylabel)
    if zero_runs:
        fig.text(0.01, 0.005, f"Diff_Null = 0 exactly (absent from log panels): {', '.join(sorted(zero_runs))}", fontsize=8, color=MUTED)
    _legend(fig, colors, sources)
    save(fig, "diff_null.png")

    # 2./3. G_refine after vs before training (diagnostics 1 and 3), one panel per refine key
    keys = list(dict.fromkeys(k for s in summaries for src in s["sources"].values() for k in src["groups"]["all"]["refine"]))
    for name, title, ylabel, get, log2 in (
        ("g_refine_norm.png", "‖G_refine‖² / ‖G_refine^Base‖²", "ratio (log2)", lambda g, k: g["pooled"]["g_refine_ratio"][k], True),
        ("cos_g_refine.png", "cos(G_refine, G_refine^Base)", "cosine", lambda g, k: g["refine"][k]["cos_g_refine"]["mean"], False),
    ):
        cols = min(len(keys), 3) or 1
        rows = -(-len(keys) // cols) or 1
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows + 0.6), squeeze=False)
        for ax, key in zip(axes.flat, keys):
            for label, summary in zip(labels, summaries):
                for source in sources:
                    group = _group(summary, source)
                    if group is None or key not in group["refine"]:
                        continue
                    ys = _positive(get(group, key)) if log2 else _clean(get(group, key))
                    ax.plot(group["t"], ys, color=colors[label], linestyle=SOURCE_STYLES[source], linewidth=1.5)
            if log2:
                ax.set_yscale("log", base=2)
                ax.axhline(1.0, color=MUTED, linewidth=0.8)
            _style_axes(ax, f"{title}: {key}", ylabel)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        _legend(fig, colors, sources)
        save(fig, name)

    # 4. Seed consistency (H1 vs H3, docs/diagnostics.md §3): trajectory source
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    has_cross = False
    for label, summary in zip(labels, summaries):
        src = summary["sources"].get("trajectory") or summary["sources"].get("forward")
        if src is None:
            continue
        cross = src.get("cross_cos", {}).get("all")
        if cross:
            has_cross = True
            axes[0].plot(cross["t"], _clean(cross["diff_null"]), color=colors[label], linewidth=1.5)
            axes[0].plot(cross["t"], _clean(cross["g_domain"]), color=colors[label], linewidth=1.0, linestyle=":")
        sc = src["seed_consistency"]["all"]
        axes[1].plot(sc["t"], _clean(sc["metrics"]["rel_diff_null"]["cv"]), color=colors[label], linewidth=1.5)
    _style_axes(axes[0], "mean pairwise cos across samples", "cosine")
    if not has_cross:
        axes[0].text(0.5, 0.5, "unit tensors not saved\n(--save-tensors)", ha="center", va="center", color=MUTED, transform=axes[0].transAxes)
    _style_axes(axes[1], "cross-seed cv of ‖Diff_Null‖/‖G_domain‖", "std / |mean|")
    extra = [
        Line2D([], [], color=MUTED, linewidth=1.5, label="Diff_Null"),
        Line2D([], [], color=MUTED, linewidth=1.0, linestyle=":", label="G_domain (reference)"),
    ]
    _legend(fig, colors, [], extra)
    save(fig, "consistency.png")
    return written
