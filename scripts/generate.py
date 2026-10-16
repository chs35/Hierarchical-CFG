"""Generate the evaluation images of a run with standard CFG (c and every c_refine).

    python scripts/generate.py --config configs/vanilla_lora.yaml            # runs/<name>/finetuner-final.safetensors
    python scripts/generate.py --config configs/vanilla_lora.yaml --checkpoint runs/vanilla_lora/checkpoint-004000.safetensors
    python scripts/generate.py --config configs/base.yaml --set name=base --base   # Base-model control
"""

from hcfg.cli import base_parser, load_from_args
from hcfg.generation.run import generate_for_run


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--checkpoint", default=None, help="fine-tuner checkpoint (default: the run's final one)")
    parser.add_argument("--base", action="store_true", help="sample the Base model (no fine-tuner)")
    parser.add_argument("--samples-name", default="samples")
    parser.add_argument("--keys", nargs="*", default=None, help="subset of prompt keys")
    args = parser.parse_args()
    config = load_from_args(args, "generate")
    checkpoint = None if args.base else (args.checkpoint or config.run_dir / "finetuner-final.safetensors")
    generate_for_run(config, checkpoint=checkpoint, samples_name=args.samples_name, keys=args.keys)


if __name__ == "__main__":
    main()
