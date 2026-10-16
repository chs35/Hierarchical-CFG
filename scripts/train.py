"""Train one fine-tuner.

    python scripts/train.py --config configs/vanilla_lora.yaml --set data.image_dir=/data/cityscapes_train10
"""

from hcfg.cli import base_parser, load_from_args
from hcfg.train.trainer import train


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = load_from_args(args, "train")
    train(config)


if __name__ == "__main__":
    main()
