"""ε of every diagnostic branch at the *same* (z_t, t) (docs/Implementation detail.md §7).

A "row" is one named condition: ``null`` (∅), ``domain`` (c) and one per refine
key (c_refine). All rows share the input latents and timesteps; FT and Base use
the identical row layout and micro-batching, Base under ``disable_adapter``
(LoRA scale 0 is bit-exact Base; for Textual Inversion the UNet has no adapter
and only the prompts differ, so its ∅ rows reproduce exactly).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from hcfg.data.prompts import DOMAIN_KEY, PromptSet
from hcfg.models.lora import disable_adapter
from hcfg.models.sdxl import SDXL, Cond

NULL_KEY = "null"


def row_texts(prompts: PromptSet, refine_keys: list[str], expand: Callable[[str], str] | None = None) -> dict[str, str]:
    """``{'domain': c, <key>: c_refine}`` (∅ has no text)."""
    texts = {DOMAIN_KEY: prompts.domain, **{key: prompts.refine(key) for key in refine_keys}}
    return {k: expand(v) for k, v in texts.items()} if expand else texts


def encode_rows(model: SDXL, texts: dict[str, str]) -> dict[str, Cond]:
    """1-row conditions in a fixed order: ∅ first, then ``texts`` in order."""
    rows = {NULL_KEY: model.null_cond(1)}
    for key, text in texts.items():
        rows[key] = model.encode_prompts([text])
    return rows


@torch.no_grad()
def predict_rows(
    model: SDXL, z: torch.Tensor, t: torch.Tensor, rows: dict[str, Cond], *, max_rows: int
) -> dict[str, torch.Tensor]:
    """ε for every row at the same ``(z, t)``.

    ``z``: ``[B, C, h, w]`` UNet input (VP z_t) in ``weight_dtype``; ``t``: ``[B]``
    actual timesteps. Rows are laid out name-major (all B samples of row 0, then
    row 1, ...) and pushed through :meth:`SDXL.predict` in chunks of ``max_rows``.
    Returns raw-dtype ``[B, C, h, w]`` per row name.
    """
    names = list(rows)
    batch = z.shape[0]
    z_all = z.repeat(len(names), 1, 1, 1)
    t_all = t.repeat(len(names))
    cond = Cond.cat([rows[name].repeat(batch) for name in names])
    out = []
    for start in range(0, z_all.shape[0], max_rows):
        stop = start + max_rows
        chunk = Cond(cond.prompt_embeds[start:stop], cond.pooled[start:stop])
        out.append(model.predict(z_all[start:stop], t_all[start:stop], chunk))
    eps = torch.cat(out)
    return {name: eps[i * batch : (i + 1) * batch] for i, name in enumerate(names)}


@torch.no_grad()
def predict_ft_base(
    model: SDXL,
    z: torch.Tensor,
    t: torch.Tensor,
    ft_rows: dict[str, Cond],
    base_rows: dict[str, Cond],
    *,
    max_rows: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """FT rows as loaded, then Base rows (adapter off) with the identical layout."""
    if list(ft_rows) != list(base_rows):
        raise ValueError(f"FT rows {list(ft_rows)} and Base rows {list(base_rows)} must match")
    eps_ft = predict_rows(model, z, t, ft_rows, max_rows=max_rows)
    with disable_adapter(model.unet):
        eps_base = predict_rows(model, z, t, base_rows, max_rows=max_rows)
    return eps_ft, eps_base
