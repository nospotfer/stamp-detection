#!/usr/bin/env python
"""Import a local YOLO-format dataset (Roboflow yolov*-style export) into
datasets/raw/ in the same COCO layout the Roboflow downloads use, so
prepare_data.py / merge treat it exactly like any other source.

The dataset directory is MOVED (cut/paste) into datasets/raw/<slug>/ to avoid
duplicate copies on disk; YOLO label txts are kept alongside for provenance.

Expected input layout (standard Roboflow YOLO export):
    <path>/data.yaml                 # names list + roboflow workspace/project
    <path>/{train,valid,test}/images/*.jpg
    <path>/{train,valid,test}/labels/*.txt

Example:
    python scripts/import_local_dataset.py --path ~/datasets/Stamp_detection_v8i_yolov11
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_split_to_coco(split_dir: Path, class_names: list[str]) -> dict:
    """Build a COCO dict from <split>/{images,labels} after images were moved
    up into <split>/ itself."""
    images, annotations = [], []
    ann_id = 0
    image_files = sorted(p for p in split_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS)
    for img_id, img_path in enumerate(image_files):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img_path.name,
                       "width": w, "height": h})
        label_file = split_dir / "labels" / (img_path.stem + ".txt")
        if not label_file.exists():
            continue
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:5])
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cls,
                "bbox": [round(x, 2), round(y, 2), round(bw * w, 2), round(bh * h, 2)],
                "area": round(bw * w * bh * h, 2), "iscrowd": 0,
            })
            ann_id += 1
    return {
        "info": {"description": f"Imported local YOLO dataset ({split_dir.name})"},
        "categories": [{"id": i, "name": n, "supercategory": "none"}
                       for i, n in enumerate(class_names)],
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--path", required=True, help="Local YOLO dataset directory")
    parser.add_argument("--slug", default=None,
                        help="Target name under datasets/raw/ "
                             "(default: <workspace>__<project> from data.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    src = Path(args.path).expanduser().resolve()
    data_yaml = src / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found - not a YOLO-format export?")
    meta = yaml.safe_load(data_yaml.read_text())
    class_names = list(meta["names"])

    rf = meta.get("roboflow", {})
    slug = args.slug or (f"{rf['workspace']}__{rf['project']}"
                         if rf.get("workspace") else src.name.lower())
    dst = Path(cfg.data.raw_dir) / slug
    if dst.exists():
        raise SystemExit(f"{dst} already exists - remove it first or pass --slug")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[import] moving {src} -> {dst}")
    shutil.move(str(src), str(dst))

    for split_dir in sorted(d for d in dst.iterdir() if (d / "images").is_dir()):
        for img in sorted((split_dir / "images").iterdir()):
            shutil.move(str(img), str(split_dir / img.name))
        (split_dir / "images").rmdir()
        coco = yolo_split_to_coco(split_dir, class_names)
        (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
        print(f"[import] {slug}/{split_dir.name}: {len(coco['images'])} images, "
              f"{len(coco['annotations'])} boxes, classes={class_names}")

    print(f"[import] done. Add '{slug}' to {cfg.data.class_map_file} and rerun:\n"
          f"  python scripts/prepare_data.py --skip-download")


if __name__ == "__main__":
    main()
