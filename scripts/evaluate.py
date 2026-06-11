#!/usr/bin/env python
"""Compute COCO detection metrics (mAP, mAP50, mAP75, AR) for a checkpoint on a split.

Example:
    python scripts/evaluate.py --checkpoint runs/rtdetrv2-stamp/best --split test
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.inference import iter_split, load_checkpoint, predict
from stamp_detection.training.metrics import METRIC_NAMES
from stamp_detection.utils import load_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config, args.overrides)
    model, image_processor, device = load_checkpoint(args.checkpoint)

    split_dir = Path(cfg.data.processed_dir) / args.split
    coco_gt = COCO(str(split_dir / "_annotations.coco.json"))

    detections = []
    n = 0
    for _, image, _, image_id in iter_split(cfg.data.processed_dir, args.split):
        pred = predict(model, image_processor, image, cfg.data.image_size,
                       threshold=0.01, device=device)
        for box, score in zip(pred["boxes"], pred["scores"]):
            x1, y1, x2, y2 = box
            detections.append({"image_id": image_id, "category_id": 0,
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
        n += 1
        if n % 100 == 0:
            print(f"[evaluate] {n} images...")

    if not detections:
        print("[evaluate] no detections produced")
        return

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    metrics = {name: float(coco_eval.stats[i]) for i, name in enumerate(METRIC_NAMES)}
    print(f"\n[evaluate] {args.checkpoint} on {args.split} ({n} images):")
    for k, v in metrics.items():
        print(f"  {k:12s} {v:.4f}")

    out = Path(args.checkpoint) / f"metrics_{args.split}.json"
    with contextlib.suppress(OSError):
        out.write_text(json.dumps(metrics, indent=2))
        print(f"[evaluate] saved -> {out}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
