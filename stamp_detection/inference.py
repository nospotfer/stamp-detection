"""Checkpoint loading and split inference shared by evaluate.py and qualitative.py."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

Image.MAX_IMAGE_PIXELS = None


def load_checkpoint(checkpoint: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    return model, image_processor, device


def iter_split(processed_dir: str | Path, split: str):
    """Yield (image_path, PIL image, gt boxes [xywh abs], image_id) for a processed split."""
    split_dir = Path(processed_dir) / split
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a["bbox"])
    for img in coco["images"]:
        path = split_dir / img["file_name"]
        with Image.open(path) as im:
            yield path, im.convert("RGB"), anns_by_image.get(img["id"], []), img["id"]


@torch.no_grad()
def predict(model, image_processor, image: Image.Image, image_size: int,
            threshold: float, device: str) -> dict:
    """Run one image; returns {"boxes": xyxy abs on original image, "scores"}."""
    w, h = image.size
    resized = image.resize((image_size, image_size))
    inputs = image_processor(images=resized, return_tensors="pt",
                             do_resize=False, do_pad=False)
    outputs = model(pixel_values=inputs["pixel_values"].to(device))
    result = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=torch.tensor([[h, w]]),
        use_focal_loss=True,
    )[0]
    return {
        "boxes": result["boxes"].cpu().tolist(),
        "scores": result["scores"].cpu().tolist(),
    }


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(pred_boxes: list, gt_boxes_xyxy: list, iou_thr: float = 0.5):
    """Greedy matching by IoU; returns (n_tp, n_fp, n_fn)."""
    unmatched_gt = list(range(len(gt_boxes_xyxy)))
    tp = 0
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j in unmatched_gt:
            iou = box_iou(pb, gt_boxes_xyxy[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr:
            tp += 1
            unmatched_gt.remove(best_j)
    fp = len(pred_boxes) - tp
    fn = len(unmatched_gt)
    return tp, fp, fn
