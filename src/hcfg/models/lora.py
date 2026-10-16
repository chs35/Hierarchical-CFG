r"""LoRA on whole attention matrices *and* on slices of them (CA-LoRA heads).

One module, :class:`LoRALinear`, covers all three LoRA fine-tuners so they share
exactly the same numerics:

* vanilla LoRA / DreamBooth LoRA: every head of every attention projection
  (``heads=None``) -> ``out = W0 x + s·B A x``.
* style CA-LoRA: only the selected heads of a matrix are trainable. The paper's
  sliced LoRA (CA-LoRA App. A.2, Alg. 3-4) cuts **one** fused weight and adapts
  only the selected heads' part of it:

  **type B** (``to_q/k/v``, heads own output rows)::

      A ∈ R^{r × d_in}, B ∈ R^{(|S|·d_h) × r}
      out = W0 x ;  out[..., rows(S)] += s · B A x

  **type A** (``to_out.0``, heads own input columns)::

      A ∈ R^{r × (|S|·d_h)}, B ∈ R^{d_out × r}
      out = W0 x + s · B A x[..., cols(S)]

  So ΔW is *structurally* zero outside the selected heads: unselected rows/cols of
  B/A do not exist (no masking, no "hope it stays small"), and the trainable
  parameter count is exactly what the head budget paid for. ``r`` is the module
  rank (64), shared across the selected heads of the matrix.

Precision: the frozen base layer is stored in ``weight_dtype`` (bf16 on A100),
``lora_A``/``lora_B`` are fp32 master weights. Under bf16 autocast both matmuls
run on tensor cores; the optimizer still updates fp32 weights.

Adapter on/off (needed by the Base branches of diagnostics / alternative
samplers): ``module.scale = 0`` returns ``base_layer(x)`` *bit-exactly*, without
touching the weights, so ε(FT,·) == ε(Base,·) holds exactly (tests/test_lora.py).
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from hcfg.models.heads import LoRAPlan, Projection

FORMAT = "hcfg-lora-v1"


def head_slice_index(heads: Sequence[int], head_dim: int) -> torch.Tensor:
    """``([0, 2], d_h=3) -> [0,1,2, 6,7,8]`` — rows (type B) / cols (type A) of the heads."""
    return torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in heads]).long()


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        heads: Sequence[int] | None = None,
        num_heads: int = 1,
        head_dim: int | None = None,
        slice_dim: str = "row",
        dropout: float = 0.0,
        init: str = "gaussian",
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if slice_dim not in ("row", "col"):
            raise ValueError(f"slice_dim must be row/col, got {slice_dim!r}")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.scale = 1.0
        """Adapter scale; 0 disables the adapter exactly (see module docstring)."""
        self.merged = False
        self.slice_dim = slice_dim
        self.num_heads = int(num_heads)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = base_layer.in_features, base_layer.out_features
        sliced_axis = out_f if slice_dim == "row" else in_f
        if heads is not None:
            heads = tuple(int(h) for h in heads)
            if not heads or list(heads) != sorted(set(heads)):
                raise ValueError(f"heads must be non-empty, sorted and unique, got {heads}")
            if head_dim is None or sliced_axis != num_heads * head_dim:
                raise ValueError(f"sliced axis {sliced_axis} != num_heads*head_dim ({num_heads}*{head_dim})")
            if heads[0] < 0 or heads[-1] >= num_heads:
                raise ValueError(f"heads {heads} outside [0, {num_heads})")
            if len(heads) == num_heads:
                heads = None  # whole matrix: plain LoRA, skip the gather/scatter
        self.head_dim = int(head_dim) if head_dim is not None else sliced_axis // max(num_heads, 1)
        self.sliced = heads is not None

        width = len(heads) * self.head_dim if heads is not None else sliced_axis
        if slice_dim == "row":
            a_shape, b_shape = (rank, in_f), (width, rank)
        else:
            a_shape, b_shape = (rank, width), (out_f, rank)
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.empty(a_shape, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(b_shape, device=device, dtype=torch.float32))
        all_heads = tuple(range(self.num_heads)) if heads is None else heads
        # Persistent: the checkpoint itself records which heads it adapts.
        self.register_buffer("lora_heads", torch.tensor(all_heads, dtype=torch.long, device=device))
        self.register_buffer(
            "_index",
            head_slice_index(all_heads, self.head_dim).to(device) if self.sliced else torch.empty(0, dtype=torch.long, device=device),
            persistent=False,
        )
        self.reset_parameters(init)

    def reset_parameters(self, init: str = "gaussian") -> None:
        """B = 0, so attaching never changes the output."""
        if init == "gaussian":
            nn.init.normal_(self.lora_A, std=1.0 / self.rank)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    # nn.Linear look-alike attributes (diffusers reads e.g. attn.to_q.weight.dtype)
    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def heads(self) -> tuple[int, ...]:
        return tuple(int(h) for h in self.lora_heads.tolist())

    @property
    def lora_type(self) -> str:
        return "B" if self.slice_dim == "row" else "A"

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base = self.base_layer(x)
        if self.merged or self.scale == 0.0:
            return base
        s = self.scaling * self.scale
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        x = self.lora_dropout(x)
        if not self.sliced:
            return base + (F.linear(F.linear(x, a), b) * s).to(base.dtype)
        if self.slice_dim == "col":  # type A: read the selected heads' input columns only
            delta = F.linear(F.linear(x.index_select(-1, self._index), a), b)
            return base + (delta * s).to(base.dtype)
        # type B: write the selected heads' output rows only
        delta = F.linear(F.linear(x, a), b)
        return base.index_add(-1, self._index, (delta * s).to(base.dtype))

    def delta_weight(self) -> torch.Tensor:
        """``s·B A`` shaped like the slice (== full matrix when not sliced), fp32."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def full_delta_weight(self) -> torch.Tensor:
        """ΔW embedded in the full weight shape (zeros outside the selected heads)."""
        weight = self.base_layer.weight
        full = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        delta = self.delta_weight()
        if not self.sliced:
            return delta
        axis = 0 if self.slice_dim == "row" else 1
        return full.index_add(axis, self._index, delta)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ΔW into the base weight (only for the all-FT sampler; Base branches need unmerged)."""
        if self.merged:
            return
        weight = self.base_layer.weight
        weight.data += (self.full_delta_weight() * self.scale).to(weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        weight = self.base_layer.weight
        weight.data -= (self.full_delta_weight() * self.scale).to(weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        heads = "all" if not self.sliced else f"{list(self.heads)}/{self.num_heads}"
        return f"type={self.lora_type}, heads={heads}, d_h={self.head_dim}, r={self.rank}, alpha={self.alpha:g}"


# --------------------------------------------------------------------------------------
# Attaching / toggling
# --------------------------------------------------------------------------------------


def _replace(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and attr.isdigit():
        parent[int(attr)] = new
    else:
        setattr(parent, attr, new)


def attach_lora(
    unet: nn.Module,
    plans: Sequence[LoRAPlan],
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    init: str = "gaussian",
) -> list[LoRALinear]:
    """Wrap each planned projection; base weights are frozen, only A/B train."""
    attached = []
    for plan in plans:
        base = unet.get_submodule(plan.module_name)
        if isinstance(base, LoRALinear):
            raise RuntimeError(f"{plan.module_name} already has a LoRA")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{plan.module_name} is {type(base).__name__}, not nn.Linear")
        base.requires_grad_(False)
        p: Projection = plan.projection
        module = LoRALinear(
            base,
            rank=rank,
            alpha=alpha,
            heads=plan.heads,
            num_heads=p.num_heads,
            head_dim=p.head_dim,
            slice_dim=p.slice_dim,
            dropout=dropout,
            init=init,
        )
        _replace(unet, plan.module_name, module)
        attached.append(module)
    return attached


def iter_lora(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for _, m in iter_lora(model) for p in (m.lora_A, m.lora_B)]


def set_adapter_scale(model: nn.Module, scale: float) -> None:
    for _, module in iter_lora(model):
        if module.merged:
            raise RuntimeError("Cannot rescale a merged LoRA; unmerge first (Base branches need it unmerged)")
        module.scale = float(scale)


@contextlib.contextmanager
def adapter_scale(model: nn.Module, scale: float):
    """Temporarily set every LoRA's scale; ``adapter_scale(unet, 0)`` == Base UNet."""
    previous = {name: m.scale for name, m in iter_lora(model)}
    set_adapter_scale(model, scale)
    try:
        yield
    finally:
        for name, module in iter_lora(model):
            module.scale = previous[name]


def disable_adapter(model: nn.Module):
    return adapter_scale(model, 0.0)


def merge_lora(model: nn.Module) -> None:
    for _, module in iter_lora(model):
        module.merge()


def unmerge_lora(model: nn.Module) -> None:
    for _, module in iter_lora(model):
        module.unmerge()


def lora_summary(model: nn.Module) -> dict:
    modules = list(iter_lora(model))
    trainable = sum(m.lora_A.numel() + m.lora_B.numel() for _, m in modules)
    heads = sum(len(m.heads) for _, m in modules)
    types = {"A": 0, "B": 0}
    for _, m in modules:
        types[m.lora_type] += 1
    return {
        "num_modules": len(modules),
        "num_sliced_modules": sum(m.sliced for _, m in modules),
        "num_heads_adapted": heads,
        "trainable_params": trainable,
        "lora_types": types,
    }


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------


def save_lora(model: nn.Module, path: str | Path, *, metadata: dict | None = None) -> None:
    tensors: dict[str, torch.Tensor] = {}
    rank = alpha = None
    for name, module in iter_lora(model):
        tensors[f"{name}.lora_A"] = module.lora_A.detach().float().cpu().contiguous()
        tensors[f"{name}.lora_B"] = module.lora_B.detach().float().cpu().contiguous()
        tensors[f"{name}.lora_heads"] = module.lora_heads.cpu().contiguous()
        rank, alpha = module.rank, module.alpha
    if not tensors:
        raise ValueError("Model has no LoRA modules to save")
    meta = {"format": FORMAT, "rank": str(rank), "alpha": str(alpha)}
    meta["extra"] = json.dumps(metadata or {}, ensure_ascii=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)


def read_lora_metadata(path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not an {FORMAT} checkpoint")
    return {"rank": int(meta["rank"]), "alpha": float(meta["alpha"]), **json.loads(meta.get("extra", "{}"))}


def load_lora(unet: nn.Module, path: str | Path, projections: Sequence[Projection]) -> list[LoRALinear]:
    """Rebuild the adapters from the checkpoint alone (heads come from its tensors)."""
    meta = read_lora_metadata(path)
    state = load_file(str(path))
    by_name = {p.name: p for p in projections}
    modules = sorted({k.rsplit(".", 1)[0] for k in state})
    plans = []
    for name in modules:
        if name not in by_name:
            raise KeyError(f"Checkpoint module {name!r} not in this UNet")
        proj = by_name[name]
        heads = tuple(state[f"{name}.lora_heads"].tolist())
        plans.append(LoRAPlan(name, proj, None if len(heads) == proj.num_heads else heads))
    attached = attach_lora(unet, plans, rank=meta["rank"], alpha=meta["alpha"])
    for name, module in iter_lora(unet):
        module.lora_A.data.copy_(state[f"{name}.lora_A"])
        module.lora_B.data.copy_(state[f"{name}.lora_B"])
    return attached
