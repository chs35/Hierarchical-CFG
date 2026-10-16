"""Overlay the diagnostics of several runs and copy their summaries into results/.

    python scripts/plot_diagnostics.py                                   # runs/*/diagnostics/summary.json
    python scripts/plot_diagnostics.py --summaries runs/a/diagnostics/summary.json runs/b/diagnostics/summary.json
"""

import argparse
import shutil

from hcfg.config import resolve_path
from hcfg.diagnostics.plot import plot_summaries
from hcfg.utils.io import load_json
from hcfg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summaries", nargs="*", default=None)
    parser.add_argument("--out", default="results/diagnostics")
    args = parser.parse_args()
    setup_logging()
    paths = [resolve_path(p) for p in args.summaries] if args.summaries else sorted(resolve_path("runs").glob("*/diagnostics/summary.json"))
    if not paths:
        raise SystemExit("No summary.json found")
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [load_json(p) for p in paths]
    for path, summary in zip(paths, summaries):
        shutil.copyfile(path, out / f"{summary['meta'].get('run', path.parent.parent.name)}_summary.json")
    for figure in plot_summaries(summaries, out):
        print(figure)


if __name__ == "__main__":
    main()
