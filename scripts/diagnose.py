"""Diagnostics of a fine-tuner (report §3): Diff_Null, G_domain, G_refine vs G_refine^Base per t.

    python scripts/diagnose.py --config configs/vanilla_lora.yaml \
        --set data.image_dir=/workspace/data/cityscapes/leftImg8bit/train data.image_list=configs/data/train10.txt
    python scripts/diagnose.py --config configs/vanilla_lora.yaml --set ... --save-tensors      # + fp16 ε tensors
    python scripts/diagnose.py --config configs/vanilla_lora.yaml --set ... --resume            # after an interruption
    python scripts/diagnose.py --config configs/*.yaml --set ... --save-tensors --check-disk     # room for all runs?

Default checkpoint: runs/<name>/finetuner-final.safetensors -> runs/<name>/diagnostics/.
Output must be on the persistent volume (RunPod: /workspace) and fit in it; both
are checked before the model is loaded.
"""

import argparse
import json
import sys
from pathlib import Path

from hcfg.config import load_config
from hcfg.diagnostics.run import FINAL_CHECKPOINT, default_out_name, diagnose_run
from hcfg.diagnostics.storage import GB, check_disk_space, ensure_persistent, estimate_output_bytes
from hcfg.utils.logging import setup_logging

SAVE_TENSORS_DEFAULT = "diagnostics.save_tensors=[eps_ft_null, eps_base_null, g_domain]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True, help="YAML config (several only with --check-disk)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides")
    parser.add_argument("--checkpoint", default=None, help="fine-tuner checkpoint (default: the run's final one)")
    parser.add_argument("--out-name", default=None, help="output directory name under the run dir")
    parser.add_argument("--sources", nargs="+", choices=["forward", "trajectory"], default=None)
    parser.add_argument("--save-tensors", action="store_true", help=f"shorthand for {SAVE_TENSORS_DEFAULT}")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run in the same directory")
    parser.add_argument("--check-disk", action="store_true", help="only check persistence and free space, then exit")
    args = parser.parse_args()
    overrides = ([SAVE_TENSORS_DEFAULT] if args.save_tensors else []) + list(args.set)
    configs = [load_config(path, overrides) for path in args.config]

    if args.check_disk:
        setup_logging()
        total: dict[str, int] = {}
        for config in configs:
            sources = args.sources or config.diagnostics.sources
            out_dir = config.run_dir / (args.out_name or "diagnostics")
            ensure_persistent(out_dir, config.storage)
            estimate = estimate_output_bytes(config, sources)
            print(json.dumps({config.name: {k: round(v / GB, 3) for k, v in estimate.items()}}))
            for key, value in estimate.items():
                total[key] = total.get(key, 0) + value
        info = check_disk_space(configs[0].run_dir.parent, total, configs[0].storage)
        print(f"OK: {len(configs)} run(s) need {info['required'] / GB:.2f} GB incl. margin, {info['free'] / GB:.2f} GB free")
        return
    if len(configs) != 1:
        sys.exit("Pass exactly one --config (several are only allowed with --check-disk)")
    config = configs[0]
    checkpoint = args.checkpoint or config.run_dir / FINAL_CHECKPOINT
    out_dir = config.run_dir / (args.out_name or default_out_name(Path(checkpoint)))
    ensure_persistent(out_dir, config.storage)  # before the log file is created
    setup_logging(config.run_dir / "diagnose.log")
    diagnose_run(config, checkpoint=checkpoint, out_name=args.out_name, sources=args.sources, resume=args.resume)


if __name__ == "__main__":
    main()
