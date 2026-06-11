#!/usr/bin/env python
"""Qualitative analysis: draw GT vs predicted boxes over a split and bucket the
results into clean / has-false-negative / has-false-positive folders.

Example:
    python scripts/qualitative.py --checkpoint runs/rtdetrv2-stamp/best \
        --split test --threshold 0.5 --wandb
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.inference import (
    iter_split, load_checkpoint, match_detections, predict, xywh_to_xyxy,
)
from stamp_detection.utils import load_env
from stamp_detection.visualization import draw_detections, side_by_side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="Output dir (default: <checkpoint>/qualitative_<split>)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Two panels (GT | predictions) instead of one overlay")
    parser.add_argument("--wandb", action="store_true",
                        help="Also log a gallery table to W&B")
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config, args.overrides)
    model, image_processor, device = load_checkpoint(args.checkpoint)

    out_root = Path(args.out or Path(args.checkpoint) / f"qualitative_{args.split}")
    buckets = {"tp_clean": 0, "with_fn": 0, "with_fp": 0}
    for b in buckets:
        (out_root / b).mkdir(parents=True, exist_ok=True)

    wandb_run = None
    table = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                               job_type="qualitative",
                               name=f"qualitative-{args.split}")
        table = wandb.Table(columns=["image", "bucket", "n_gt", "n_pred", "tp", "fp", "fn"])

    rows = []
    for i, (path, image, gt_xywh, _) in enumerate(
            iter_split(cfg.data.processed_dir, args.split)):
        if args.max_images is not None and i >= args.max_images:
            break
        pred = predict(model, image_processor, image, cfg.data.image_size,
                       threshold=args.threshold, device=device)
        gt_xyxy = [xywh_to_xyxy(b) for b in gt_xywh]
        tp, fp, fn = match_detections(pred["boxes"], gt_xyxy)
        bucket = "with_fn" if fn else ("with_fp" if fp else "tp_clean")
        buckets[bucket] += 1

        render = (side_by_side if args.side_by_side else draw_detections)(
            image, pred_boxes=pred["boxes"], pred_scores=pred["scores"],
            gt_boxes=gt_xyxy,
        )
        out_path = out_root / bucket / f"{path.stem}.jpg"
        render.save(out_path, quality=90)
        rows.append({"file": path.name, "bucket": bucket, "n_gt": len(gt_xyxy),
                     "n_pred": len(pred["boxes"]), "tp": tp, "fp": fp, "fn": fn})
        if table is not None:
            import wandb

            table.add_data(wandb.Image(render), bucket, len(gt_xyxy),
                           len(pred["boxes"]), tp, fp, fn)

    with open(out_root / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "bucket", "n_gt", "n_pred",
                                               "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(buckets.values())
    print(f"[qualitative] {total} images -> {out_root}")
    for b, n in buckets.items():
        print(f"  {b}: {n} ({n / max(total, 1):.1%})")

    if wandb_run is not None:
        import wandb

        wandb.log({f"qualitative/{args.split}": table})
        wandb.finish()


if __name__ == "__main__":
    main()
