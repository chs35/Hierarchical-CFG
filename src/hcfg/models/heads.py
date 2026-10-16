"""Attention projections and their per-head slices.

Unit names follow the reference repo (chs35/Improving-CA-LoRA) so its selection
JSONs load unchanged: a whole projection is its module path
(``...attn2.to_k``), one head is ``<module path>@h<index>``.

Which weight axis a head owns:

* ``to_q`` / ``to_k`` / ``to_v`` write ``inner_dim = H * d_h`` -> a head owns a
  band of **rows** (output features).
* ``to_out.0`` reads ``inner_dim`` -> a head owns a band of **columns** (input
  features).

``H`` is always read from ``Attention.heads``; ``d_h = to_q.out_features // H``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from torch import nn

PROJECTION_PATTERN = re.compile(
    r"(?:^|\.)attn(?P<attn>\d+)\.to_(?P<kind>q|k|v|out)(?:\.0)?(?P<wrapper>\.base_layer)?$"
)
HEAD_SEPARATOR = "@h"
SLICE_DIM = {"q": "row", "k": "row", "v": "row", "out": "col"}


def head_unit_name(module_name: str, head: int) -> str:
    return f"{module_name}{HEAD_SEPARATOR}{head}"


def parse_unit_name(name: str) -> tuple[str, int | None]:
    module, sep, suffix = name.rpartition(HEAD_SEPARATOR)
    if not sep:
        return name, None
    if not suffix.isdigit():
        raise ValueError(f"Malformed head unit name {name!r}")
    return module, int(suffix)


@dataclass(frozen=True)
class Projection:
    name: str
    """Canonical module path (without any ``.base_layer`` wrapper suffix)."""
    kind: str
    attn: int
    num_heads: int
    head_dim: int
    in_features: int
    out_features: int

    @property
    def slice_dim(self) -> str:
        return SLICE_DIM[self.kind]

    @property
    def head_numel(self) -> int:
        """Weights one head owns (biases are not attributed to heads)."""
        return self.head_dim * (self.in_features if self.slice_dim == "row" else self.out_features)


def _attention_path(module_name: str) -> str:
    return module_name.rsplit(".to_", 1)[0]


def enumerate_projections(
    model: nn.Module,
    *,
    kinds: Iterable[str] = ("q", "k", "v", "out"),
    attn: Iterable[int] = (1, 2),
) -> list[Projection]:
    """Every attention projection ``nn.Linear`` of a diffusers UNet, in module order."""
    kinds, attn = set(kinds), set(attn)
    modules = dict(model.named_modules())
    projections: list[Projection] = []
    for name, module in modules.items():
        if not isinstance(module, nn.Linear):
            continue
        match = PROJECTION_PATTERN.search(name)
        if match is None or match.group("kind") not in kinds or int(match.group("attn")) not in attn:
            continue
        canonical = name[: -len(match.group("wrapper"))] if match.group("wrapper") else name
        attention = modules[_attention_path(canonical)]
        heads = int(attention.heads)
        inner = attention.to_q.out_features if isinstance(attention.to_q, nn.Linear) else (
            attention.to_q.base_layer.out_features
        )
        if inner % heads:
            raise ValueError(f"{canonical}: inner dim {inner} not divisible by heads={heads}")
        head_dim = inner // heads
        kind = match.group("kind")
        sliced = module.out_features if SLICE_DIM[kind] == "row" else module.in_features
        if sliced != heads * head_dim:
            raise ValueError(f"{canonical}: sliced axis {sliced} != {heads}*{head_dim}")
        projections.append(
            Projection(
                name=canonical,
                kind=kind,
                attn=int(match.group("attn")),
                num_heads=heads,
                head_dim=head_dim,
                in_features=module.in_features,
                out_features=module.out_features,
            )
        )
    if not projections:
        raise ValueError("No attention projections found (expected diffusers attn{n}.to_{q,k,v,out.0})")
    return projections


@dataclass(frozen=True)
class LoRAPlan:
    """One LoRA per matrix; ``heads=None`` means the whole matrix (standard LoRA)."""

    module_name: str
    projection: Projection
    heads: tuple[int, ...] | None


def plans_full(projections: Sequence[Projection]) -> list[LoRAPlan]:
    return [LoRAPlan(p.name, p, None) for p in projections]


def plans_from_units(projections: Sequence[Projection], unit_names: Iterable[str]) -> list[LoRAPlan]:
    """Group head units ``<module>@h<i>`` by module -> one sliced LoRA per matrix.

    A module whose every head is selected becomes a plain full LoRA (numerically
    identical, see tests), and a bare module name selects the whole matrix.
    """
    by_name = {p.name: p for p in projections}
    grouped: dict[str, set[int] | None] = {}
    for unit in unit_names:
        module, head = parse_unit_name(unit)
        if module not in by_name:
            raise KeyError(f"Selected unit {unit!r} does not exist in this UNet")
        if head is None:
            grouped[module] = None
            continue
        if not 0 <= head < by_name[module].num_heads:
            raise ValueError(f"{unit}: head index out of range [0, {by_name[module].num_heads})")
        if module in grouped and grouped[module] is None:
            continue
        grouped.setdefault(module, set()).add(head)
    plans = []
    for p in projections:  # module order, not selection order: checkpoint-stable
        if p.name not in grouped:
            continue
        heads = grouped[p.name]
        if heads is not None and len(heads) == p.num_heads:
            heads = None
        plans.append(LoRAPlan(p.name, p, None if heads is None else tuple(sorted(heads))))
    if not plans:
        raise ValueError("Selection is empty")
    return plans
