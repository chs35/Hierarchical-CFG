"""Evaluate a generated sample set: CLIP-score, CMMD, zero-shot weather/vehicle, OWLv2 vehicle+colour.

    python scripts/evaluate.py --config configs/vanilla_lora.yaml --set eval.reference_dir=/data/cityscapes_val500
"""

from hcfg.cli import base_parser, load_from_args
from hcfg.eval.evaluate import evaluate_samples


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--samples-name", default="samples")
    args = parser.parse_args()
    config = load_from_args(args, "evaluate")
    evaluate_samples(config, config.run_dir / args.samples_name)


if __name__ == "__main__":
    main()
