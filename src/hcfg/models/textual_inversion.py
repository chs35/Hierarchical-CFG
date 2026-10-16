"""Textual Inversion for SDXL's two text encoders.

The new token's vectors live in a separate fp32 ``nn.Parameter`` and are spliced
into the frozen token-embedding lookup, rather than un-freezing the whole
embedding matrix (where AdamW weight decay or any stray gradient would silently
move the *other* tokens). Consequences, both pinned by tests:

* UNet untouched and every pre-existing token untouched, so any prompt without
  the placeholder — in particular ∅ — encodes bit-identically to the Base model:
  ``Diff_Null == 0`` exactly.
* The checkpoint is just the new vectors per encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

FORMAT = "hcfg-ti-v1"


class ExtendedEmbedding(nn.Module):
    """``nn.Embedding`` of the original vocabulary + trainable rows for new ids."""

    def __init__(self, base: nn.Embedding, init_vectors: torch.Tensor):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.num_base = base.num_embeddings
        self.new_embeddings = nn.Parameter(init_vectors.detach().clone().float())

    @property
    def weight(self) -> torch.Tensor:  # transformers inspects ``embedding.weight``
        return self.base.weight

    @property
    def num_embeddings(self) -> int:
        return self.num_base + self.new_embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.base.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_new = input_ids >= self.num_base
        out = self.base(input_ids.clamp(max=self.num_base - 1))
        if not bool(is_new.any()):
            return out
        new = self.new_embeddings.to(out.dtype)[(input_ids - self.num_base).clamp(min=0)]
        return torch.where(is_new.unsqueeze(-1), new, out)


def placeholder_tokens(token: str, num_vectors: int) -> list[str]:
    return [token] if num_vectors == 1 else [f"{token}_{i}" for i in range(num_vectors)]


def expand_prompt(prompt: str, placeholder: str, token: str, num_vectors: int) -> str:
    """Prompt-file notation (``S*``) -> the registered token string(s)."""
    return prompt.replace(placeholder, " ".join(placeholder_tokens(token, num_vectors)))


def _embedding_module(text_encoder: nn.Module) -> nn.Module:
    # get/set_input_embeddings: stable across transformers 4.x and 5.x layouts
    return text_encoder.get_input_embeddings()


def install_textual_inversion(
    tokenizers: list,
    text_encoders: list[nn.Module],
    *,
    token: str,
    num_vectors: int = 1,
    initializer: str | None = None,
    vectors: list[torch.Tensor] | None = None,
) -> list[ExtendedEmbedding]:
    """Add the token(s) to every tokenizer and splice trainable rows into every encoder.

    Initialised from ``initializer``'s embedding (mean over its sub-tokens) unless
    explicit ``vectors`` (from a checkpoint) are given.
    """
    tokens = placeholder_tokens(token, num_vectors)
    modules = []
    for index, (tokenizer, encoder) in enumerate(zip(tokenizers, text_encoders)):
        base = _embedding_module(encoder)
        if isinstance(base, ExtendedEmbedding):
            raise RuntimeError("Textual inversion already installed on this encoder")
        added = tokenizer.add_tokens(tokens)
        if added != len(tokens):
            raise ValueError(f"Tokenizer {index} already contains {tokens}")
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if ids != list(range(base.num_embeddings, base.num_embeddings + len(tokens))):
            raise RuntimeError(f"Unexpected new token ids {ids} (vocab size {base.num_embeddings})")
        if vectors is not None:
            init = vectors[index]
        else:
            init_ids = tokenizer(initializer, add_special_tokens=False).input_ids
            if not init_ids:
                raise ValueError(f"Initializer {initializer!r} tokenizes to nothing")
            mean = base.weight[init_ids].float().mean(dim=0)
            init = mean.unsqueeze(0).repeat(len(tokens), 1)
        module = ExtendedEmbedding(base, init.to(base.weight.device))
        encoder.set_input_embeddings(module)
        assert encoder.get_input_embeddings() is module
        modules.append(module)
    return modules


def ti_parameters(text_encoders: list[nn.Module]) -> list[nn.Parameter]:
    params = []
    for encoder in text_encoders:
        module = _embedding_module(encoder)
        if isinstance(module, ExtendedEmbedding):
            params.append(module.new_embeddings)
    return params


def save_textual_inversion(text_encoders: list[nn.Module], path: str | Path, *, token: str, metadata: dict | None = None) -> None:
    tensors = {}
    for index, encoder in enumerate(text_encoders):
        module = _embedding_module(encoder)
        tensors[f"text_encoder_{index}"] = module.new_embeddings.detach().float().cpu().contiguous()
    meta = {"format": FORMAT, "token": token, "num_vectors": str(tensors["text_encoder_0"].shape[0])}
    meta["extra"] = json.dumps(metadata or {}, ensure_ascii=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)


def load_textual_inversion(tokenizers: list, text_encoders: list[nn.Module], path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not an {FORMAT} checkpoint")
    state = load_file(str(path))
    vectors = [state[f"text_encoder_{i}"] for i in range(len(text_encoders))]
    install_textual_inversion(
        tokenizers, text_encoders, token=meta["token"], num_vectors=int(meta["num_vectors"]), vectors=vectors
    )
    return {"token": meta["token"], "num_vectors": int(meta["num_vectors"]), **json.loads(meta.get("extra", "{}"))}
