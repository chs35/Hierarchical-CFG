"""Small JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
