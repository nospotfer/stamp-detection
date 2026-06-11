#!/usr/bin/env python
"""Fine-tune RT-DETRv2 on the merged stamp dataset.

Examples:
    python scripts/train.py
    python scripts/train.py --set train.lr=5e-5 aug.strength=0.8 train.run_name=exp2
    python scripts/train.py --set wandb.enabled=false train.epochs=2  # smoke test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.training.run import run_training
from stamp_detection.utils import load_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config, args.overrides)
    run_training(cfg)


if __name__ == "__main__":
    main()
