#!/usr/bin/env python
"""Render ground-truth samples of the processed dataset, grouped by source
Roboflow dataset, to visually inspect annotation quality after the merge.

Writes individual renders plus one contact-sheet grid per source.

Example:
    python scripts/visualize_dataset.py --per-source 8
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.visualization import draw_detections

THUMB = 420  # contact-sheet cell size in px


def collect_by_source(processed_dir: Path, splits: list[str]) -> dict[str, list[dict]]:
    by_source = defaultdict(list)
    for split in splits:
        split_dir = processed_dir / split
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        manifest = json.loads((split_dir / "manifest.json").read_text())
        anns_by_image = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_image[a["image_id"]].append(a["bbox"])
        for img in coco["images"]:
            source = manifest[img["file_name"]]["source"]
            by_source[source].append({
                "path": split_dir / img["file_name"],
                "split": split,
                "gt": anns_by_image.get(img["id"], []),
            })
    return by_source


def contact_sheet(renders: list[Image.Image], labels: list[str], cols: int) -> Image.Image:
    rows = (len(renders) + cols - 1) // cols
    pad, header = 6, 22
    sheet = Image.new("RGB", (cols * (THUMB + pad) + pad,
                              rows * (THUMB + header + pad) + pad), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for i, (render, label) in enumerate(zip(renders, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (THUMB + pad)
        y = pad + r * (THUMB + header + pad)
        thumb = render.copy()
        thumb.thumbnail((THUMB, THUMB))
        draw.text((x + 2, y + 4), label, fill=(255, 255, 255))
        sheet.paste(thumb, (x + (THUMB - thumb.width) // 2,
                            y + header + (THUMB - thumb.height) // 2))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--per-source", type=int, default=8)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--out", default=None,
                        help="Output dir (default: <processed_dir>/preview)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    processed_dir = Path(cfg.data.processed_dir)
    out_root = Path(args.out or processed_dir / "preview")
    rng = random.Random(args.seed)

    by_source = collect_by_source(processed_dir, args.splits)
    for source, items in sorted(by_source.items()):
        # bias the sample towards annotated images but always include a negative
        positives = [it for it in items if it["gt"]]
        negatives = [it for it in items if not it["gt"]]
        n_neg = min(1 if negatives else 0, args.per_source)
        sample = rng.sample(positives, min(args.per_source - n_neg, len(positives)))
        sample += rng.sample(negatives, n_neg)

        source_dir = out_root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        renders, labels = [], []
        for it in sample:
            with Image.open(it["path"]) as im:
                gt_xyxy = [[x, y, x + w, y + h] for x, y, w, h in it["gt"]]
                render = draw_detections(im, gt_boxes=gt_xyxy)
            render.save(source_dir / f"{it['split']}_{it['path'].name}", quality=88)
            renders.append(render)
            labels.append(f"{it['split']}/{it['path'].stem[:12]}  ({len(it['gt'])} boxes)")
        sheet = contact_sheet(renders, labels, cols=4)
        sheet.save(out_root / f"{source}.jpg", quality=88)
        print(f"[preview] {source}: {len(sample)} samples "
              f"({len(positives)} positives / {len(negatives)} negatives available) "
              f"-> {out_root / (source + '.jpg')}")


if __name__ == "__main__":
    main()
