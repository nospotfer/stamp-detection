"""COCO dataset wrapper feeding RT-DETRv2: PIL -> albumentations -> HF image processor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None


class CocoStampDataset(Dataset):
    """One processed split (datasets/processed/<split>).

    __getitem__ returns the dict the HF Trainer/collate expect:
    {"pixel_values": FloatTensor[3,H,W], "labels": {class_labels, boxes, ...}}.
    The image processor is called with do_resize/do_pad off because albumentations
    already produced the square model input; it converts COCO xywh-abs annotations
    to the normalized cxcywh targets RT-DETR trains on.
    """

    def __init__(self, split_dir: str | Path, transforms, image_processor,
                 max_samples: int | None = None):
        self.split_dir = Path(split_dir)
        self.transforms = transforms
        self.image_processor = image_processor
        coco = json.loads((self.split_dir / "_annotations.coco.json").read_text())
        anns_by_image = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)
        self.items = [
            {"image": img, "annotations": anns_by_image.get(img["id"], [])}
            for img in coco["images"]
        ]
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        img_info = item["image"]
        with Image.open(self.split_dir / img_info["file_name"]) as im:
            image = np.asarray(im.convert("RGB"))

        bboxes = [a["bbox"] for a in item["annotations"]]
        category_ids = [a["category_id"] for a in item["annotations"]]
        out = self.transforms(image=image, bboxes=bboxes, category_ids=category_ids)

        annotations = {
            "image_id": img_info["id"],
            "annotations": [
                {"image_id": img_info["id"], "category_id": cid, "bbox": list(box),
                 "area": float(box[2] * box[3]), "iscrowd": 0}
                for box, cid in zip(out["bboxes"], out["category_ids"])
            ],
        }
        encoding = self.image_processor(
            images=out["image"],
            annotations=annotations,
            return_tensors="pt",
            do_resize=False,
            do_pad=False,
        )
        labels = {k: v for k, v in encoding["labels"][0].items()}
        # keep the pre-resize size around for COCO eval / postprocessing
        labels["orig_size"] = torch.tensor([img_info["height"], img_info["width"]])
        return {"pixel_values": encoding["pixel_values"][0], "labels": labels}


def collate_fn(batch: list[dict]) -> dict:
    """Stack pixel values; labels stay a list of dicts (what RTDetrV2 forward expects)."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }
