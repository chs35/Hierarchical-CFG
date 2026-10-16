"""GPU verification (A100): real SDXL, exact identities, tensor cores, short runs of every stage.

    # quick (~5 min): environment, exactness on real SDXL, tensor-core profile
    python scripts/gpu_check.py --stages env exact tensorcore

    # full (~30-40 min): + 60-step training of all five fine-tuners, generation, evaluation, CA
    python scripts/gpu_check.py --image-dir /data/cityscapes_train10 --reference-dir /data/cityscapes_val500 --compare-fp32

Without --image-dir, synthetic 2048x1024 images are used (speed/plumbing only).
Writes runs/gpu_check/report.json; exit code 1 if any check FAILs.
See docs/REPRODUCE.md (GPU verification) for how to read the report.
"""

import argparse
import sys

from hcfg.gpu_check import FINETUNER_CONFIGS, STAGES, run_gpu_check
from hcfg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="runs/gpu_check")
    parser.add_argument("--image-dir", default=None, help="the fixed 10 training images (default: synthetic)")
    parser.add_argument("--reference-dir", default=None, help="real Cityscapes images, enables the CMMD check")
    parser.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    parser.add_argument("--steps", type=int, default=60, help="training steps per fine-tuner in the train stage")
    parser.add_argument("--finetuners", nargs="*", default=list(FINETUNER_CONFIGS), choices=list(FINETUNER_CONFIGS))
    parser.add_argument("--compare-fp32", action="store_true", help="also time an fp32/no-TF32 vanilla LoRA baseline")
    args = parser.parse_args()
    setup_logging(f"{args.out}/gpu_check.log")
    report = run_gpu_check(
        out_dir=args.out,
        image_dir=args.image_dir,
        reference_dir=args.reference_dir,
        stages=args.stages,
        steps=args.steps,
        finetuners=args.finetuners,
        compare_fp32=args.compare_fp32,
    )
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
