"""Console + file logging shared by every script."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    # Windows consoles default to cp949/cp1252 and choke on "∅"/"ε" in log lines.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
