#!/usr/bin/env python
"""Download all Roboflow candidate datasets and merge them into one single-class
COCO dataset under datasets/processed/.

First run prints any class names missing from configs/dataset_class_map.yaml;
fill the map (stamp subclasses -> stamp, everything else -> DROP) and rerun
with --skip-download.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.data.download import download_all
from stamp_detection.data.merge import merge
from stamp_detection.utils import load_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--skip-download", action="store_true",
                        help="Only run the merge step on already-downloaded data")
    parser.add_argument("--force", action="store_true",
                        help="Re-download datasets even if present")
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config, args.overrides)

    if not args.skip_download:
        download_all(cfg.data.datasets_file, cfg.data.raw_dir, force=args.force)

    ok = merge(cfg.data)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
