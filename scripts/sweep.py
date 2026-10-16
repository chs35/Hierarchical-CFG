"""Hierarchical-CFG (w1, w2) grid of a fine-tuner (report §4.2): c_refine keys x seeds x N images per cell.

    python scripts/sweep.py --config configs/vanilla_lora.yaml                    # hcfg.variant=basic, 5x5 grid
    python scripts/sweep.py --config configs/vanilla_lora.yaml --set hcfg.variant=alt_domain
    python scripts/sweep.py --config configs/*.yaml --check-disk                  # room for all runs?
    # Diff_Null split (report §7): explicit [w1, w2, λ] cells -> hcfg/basic_split/
    python scripts/sweep.py --config configs/ca_lora_style.yaml --set "hcfg.cells=[[7,2,0],[7,4,0],[7,4,-1],[7,4,5]]"

Output: runs/<name>/hcfg/<variant>/w1_<a>_w2_<b>[_lam_<λ>]/<key>/seed<s>/<i>.png (+ manifest.json per cell,
usable with scripts/evaluate.py-style evaluation). Re-running the same command resumes.
"""

import argparse
import sys
from pathlib import Path

from hcfg.config import load_config
from hcfg.diagnostics.storage import GB, ensure_persistent
from hcfg.generation.sweep import FINAL_CHECKPOINT, check_only, run_sweep
from hcfg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True, help="YAML config (several only with --check-disk)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides")
    parser.add_argument("--checkpoint", default=None, help="fine-tuner checkpoint (default: the run's final one)")
    parser.add_argument("--out-name", default=None, help="sweep directory name under runs/<name>/hcfg/")
    parser.add_argument("--check-disk", action="store_true", help="only check persistence and free space, then exit")
    args = parser.parse_args()
    configs = [load_config(path, args.set) for path in args.config]
    if args.check_disk:
        setup_logging()
        info = check_only(configs, args.out_name)
        print(f"OK: {sum(info['images_to_generate'].values())} images, need {info['required'] / GB:.2f} GB incl. margin, "
              f"{info['free'] / GB:.2f} GB free")
        return
    if len(configs) != 1:
        sys.exit("Pass exactly one --config (several are only allowed with --check-disk)")
    config = configs[0]
    ensure_persistent(config.run_dir, config.storage)  # before the log file is created
    setup_logging(config.run_dir / "sweep.log")
    run_sweep(config, checkpoint=Path(args.checkpoint or config.run_dir / FINAL_CHECKPOINT), out_name=args.out_name)


if __name__ == "__main__":
    main()
