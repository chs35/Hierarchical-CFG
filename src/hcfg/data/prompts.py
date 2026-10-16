"""Prompt sets: c (domain), c_refine = c + suffix, the DreamBooth prior prompt and c_control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hcfg.config import PromptConfig, resolve_path

DOMAIN_KEY = "domain"
CONTROL_KEY = "control"


@dataclass(frozen=True)
class PromptSet:
    name: str
    domain: str
    prior: str | None
    refine_suffixes: dict[str, str]
    targets: dict[str, dict] = field(default_factory=dict)
    control: str | None = None
    """c_control (diagnostics only): unrelated prompt; not part of :attr:`keys`."""

    def refine(self, key: str) -> str:
        """``c_refine`` for ``key``; always ``c`` + suffix, so c is a prefix of c_refine."""
        return self.domain + self.refine_suffixes[key]

    def prompt(self, key: str) -> str:
        """``domain`` -> c, ``control`` -> c_control, any refine key -> c_refine."""
        if key == DOMAIN_KEY:
            return self.domain
        if key == CONTROL_KEY:
            if self.control is None:
                raise KeyError("No 'control' prompt in the prompt file")
            return self.control
        if key not in self.refine_suffixes:
            raise KeyError(f"Unknown prompt key {key!r}; known: {[DOMAIN_KEY, *self.refine_suffixes]}")
        return self.refine(key)

    @property
    def keys(self) -> list[str]:
        return [DOMAIN_KEY, *self.refine_suffixes]


def load_prompt_set(config: PromptConfig) -> PromptSet:
    path: Path = resolve_path(config.file)
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    sets = data["sets"]
    if config.set not in sets:
        raise KeyError(f"Prompt set {config.set!r} not in {path}; available {sorted(sets)}")
    entry = sets[config.set]
    return PromptSet(
        name=config.set,
        domain=entry["domain"],
        prior=entry.get("prior"),
        refine_suffixes=dict(data.get("refine", {})),
        targets=dict(data.get("targets", {})),
        control=data.get("control"),
    )
