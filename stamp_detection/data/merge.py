"""Merge raw Roboflow datasets into one single-class COCO dataset.

Steps: ingest all raw splits -> remap classes (subclasses of stamp -> "stamp",
other classes dropped) -> sanitize boxes -> dedup (sha256 exact + dHash near-dup)
-> cap negatives -> deterministic re-split -> emit processed/{train,val,test}.

Class mapping lives in configs/dataset_class_map.yaml:

    hrsdyolo__stamp-yctn7:
      stamp: stamp
    swp-3jks1__stamp-shape:
      round: stamp
      signature: DROP

Any class name found in the data but missing from the map is a hard error, so
new/unexpected classes always get a human decision.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from stamp_detection.config import DataConfig
from stamp_detection.utils import dhash, hamming, sha256_file

Image.MAX_IMAGE_PIXELS = None  # some scans are huge; we trust our own data

MIN_BOX_SIDE = 2.0
MIN_BOX_AREA = 4.0


@dataclass
class Record:
    """One image with its remapped stamp annotations."""

    source: str  # dataset slug
    path: Path
    width: int
    height: int
    boxes: list[list[float]] = field(default_factory=list)  # xywh abs
    sha256: str = ""
    dhash: int = 0


def _load_class_map(path: str | Path) -> dict[str, dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _ingest_source(source_dir: Path, class_map: dict[str, str] | None,
                   unmapped: dict[str, set]) -> list[Record]:
    """Read every split of one raw dataset; remap/drop classes; sanitize boxes."""
    records = []
    slug = source_dir.name
    for ann_file in sorted(source_dir.glob("*/_annotations.coco.json")):
        coco = json.loads(ann_file.read_text())
        # Roboflow exports often include a category 0 supercategory placeholder
        # (e.g. "stamps" with the project name) that has no annotations of its own;
        # we only error on names that actually appear in annotations.
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        used_cat_ids = {a["category_id"] for a in coco["annotations"]}
        anns_by_image = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_image[a["image_id"]].append(a)

        for name in (cat_names[cid] for cid in used_cat_ids):
            if class_map is None or name not in class_map:
                unmapped[slug].add(name)
        if unmapped.get(slug):
            continue  # collecting class names for the report; no records this pass

        for img in coco["images"]:
            img_path = ann_file.parent / img["file_name"]
            if not img_path.exists():
                continue
            w, h = img["width"], img["height"]
            boxes = []
            for a in anns_by_image.get(img["id"], []):
                if class_map[cat_names[a["category_id"]]] == "DROP":
                    continue
                x, y, bw, bh = a["bbox"]
                # clip to image bounds
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
                if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE \
                        or (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                    continue
                boxes.append([x1, y1, x2 - x1, y2 - y1])
            records.append(Record(source=slug, path=img_path, width=w, height=h, boxes=boxes))
    return records


def _dedup(records: list[Record], dhash_threshold: int) -> tuple[list[Record], dict]:
    """Exact (sha256) then near-duplicate (dHash) dedup.

    Within a duplicate group keep the record with most boxes, then largest area.
    """
    for r in records:
        r.sha256 = sha256_file(r.path)
        with Image.open(r.path) as im:
            r.dhash = dhash(im)

    def better(a: Record, b: Record) -> Record:
        ka = (len(a.boxes), a.width * a.height)
        kb = (len(b.boxes), b.width * b.height)
        return a if ka >= kb else b

    # pass 1: exact
    by_sha: dict[str, Record] = {}
    exact_dups = 0
    for r in records:
        if r.sha256 in by_sha:
            exact_dups += 1
            by_sha[r.sha256] = better(by_sha[r.sha256], r)
        else:
            by_sha[r.sha256] = r
    survivors = list(by_sha.values())

    # pass 2: near-duplicate via dHash hamming distance. O(n^2) on unique images
    # is fine at this scale (<20k); bucket by hash prefix to cut comparisons.
    survivors.sort(key=lambda r: r.sha256)
    near_dups = 0
    kept: list[Record] = []
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in survivors:
        merged = False
        for i, k in enumerate(kept):
            if hamming(r.dhash, k.dhash) <= dhash_threshold:
                pair_counts[tuple(sorted((r.source, k.source)))] += 1
                kept[i] = better(k, r)
                near_dups += 1
                merged = True
                break
        if not merged:
            kept.append(r)

    report = {
        "input_images": len(records),
        "exact_duplicates_removed": exact_dups,
        "near_duplicates_removed": near_dups,
        "kept": len(kept),
        "near_dup_source_pairs": {f"{a} <-> {b}": n for (a, b), n in sorted(pair_counts.items())},
    }
    return kept, report


def _split(records: list[Record], cfg: DataConfig) -> dict[str, list[Record]]:
    """Deterministic split after dedup: shuffle by seed over sha256-sorted records."""
    ordered = sorted(records, key=lambda r: r.sha256)
    rng = random.Random(cfg.seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = round(n * cfg.train_fraction)
    n_val = round(n * cfg.val_fraction)
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }


def _emit_split(split: str, records: list[Record], out_root: Path) -> dict:
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations, manifest = [], [], {}
    ann_id = 0
    for img_id, r in enumerate(records):
        fname = f"{r.sha256[:16]}{r.path.suffix.lower()}"
        shutil.copy2(r.path, split_dir / fname)
        images.append({"id": img_id, "file_name": fname, "width": r.width, "height": r.height})
        manifest[fname] = {"source": r.source, "original": r.path.name}
        for box in r.boxes:
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 0,
                "bbox": [round(v, 2) for v in box],
                "area": round(box[2] * box[3], 2), "iscrowd": 0,
            })
            ann_id += 1
    coco = {
        "info": {"description": f"Merged single-class stamp dataset ({split})"},
        "categories": [{"id": 0, "name": "stamp", "supercategory": "none"}],
        "images": images,
        "annotations": annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"images": len(images), "boxes": len(annotations)}


def merge(cfg: DataConfig) -> bool:
    """Run the full merge. Returns True on success, False if unmapped classes were found."""
    raw_dir = Path(cfg.raw_dir)
    class_maps = _load_class_map(cfg.class_map_file)
    unmapped: dict[str, set] = defaultdict(set)

    all_records: list[Record] = []
    per_source: dict[str, dict] = {}
    for source_dir in sorted(d for d in raw_dir.iterdir() if d.is_dir()):
        recs = _ingest_source(source_dir, class_maps.get(source_dir.name), unmapped)
        all_records.extend(recs)
        per_source[source_dir.name] = {
            "images": len(recs),
            "boxes": sum(len(r.boxes) for r in recs),
        }

    if any(unmapped.values()):
        print("\n[merge] ERROR: unmapped class names found. Add them to "
              f"{cfg.class_map_file} (map to 'stamp' or 'DROP'):\n")
        for slug, names in sorted(unmapped.items()):
            if names:
                print(f"  {slug}:")
                for n in sorted(names):
                    print(f"    {n}: stamp   # or DROP")
        return False

    print("\n[merge] ingested per source:")
    for slug, s in per_source.items():
        print(f"  {slug}: {s['images']} images, {s['boxes']} boxes")

    deduped, dedup_report = _dedup(all_records, cfg.dedup_dhash_threshold)
    print(f"\n[merge] dedup: {dedup_report['input_images']} -> {dedup_report['kept']} "
          f"({dedup_report['exact_duplicates_removed']} exact, "
          f"{dedup_report['near_duplicates_removed']} near dups removed)")
    for pair, n in dedup_report["near_dup_source_pairs"].items():
        print(f"    {pair}: {n}")

    # cap annotation-free negatives (real "no stamp on invoice" case, but don't flood)
    positives = [r for r in deduped if r.boxes]
    negatives = [r for r in deduped if not r.boxes]
    max_neg = round(cfg.negative_keep_ratio * len(positives))
    if len(negatives) > max_neg:
        rng = random.Random(cfg.seed)
        negatives = rng.sample(sorted(negatives, key=lambda r: r.sha256), max_neg)
    print(f"[merge] {len(positives)} positives, keeping {len(negatives)} negatives "
          f"(cap ratio {cfg.negative_keep_ratio})")

    splits = _split(positives + negatives, cfg)
    out_root = Path(cfg.processed_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    stats = {}
    for split, recs in splits.items():
        stats[split] = _emit_split(split, recs, out_root)
    summary = {"per_source": per_source, "dedup": dedup_report, "splits": stats}
    (out_root / "merge_report.json").write_text(json.dumps(summary, indent=2))
    print("\n[merge] final splits:")
    for split, s in stats.items():
        print(f"  {split}: {s['images']} images, {s['boxes']} boxes")
    print(f"[merge] report -> {out_root / 'merge_report.json'}")
    return True
