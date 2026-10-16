"""Shared argument parsing for scripts/*.py."""

from __future__ import annotations

import argparse

from hcfg.config import Config, load_config
from hcfg.utils.logging import setup_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config (see configs/)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. train.max_steps=100")
    return parser


def load_from_args(args: argparse.Namespace, log_name: str) -> Config:
    config = load_config(args.config, args.set)
    setup_logging(config.run_dir / f"{log_name}.log")
    return config
