"""COCO mAP/AR metrics for the HF Trainer.

Requires TrainingArguments(eval_do_concat_batches=False) so compute_metrics
receives per-batch (logits, pred_boxes) and label-dict lists instead of a
broken concatenation of variable-length structures.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_NAMES = [
    "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large",
]


def build_compute_metrics(image_processor, threshold: float = 0.01):
    """Returns a compute_metrics closure accumulating COCO detections across batches."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        gt_images, gt_anns, dt = [], [], []
        ann_id = 0
        for batch_preds, batch_labels in zip(predictions, labels):
            # batch_preds: tuple of arrays; [1] = logits, [2] = pred_boxes
            # (index [0] is loss-related when labels are passed)
            logits, pred_boxes = batch_preds[1], batch_preds[2]
            target_sizes = torch.tensor(
                np.stack([t["orig_size"] for t in batch_labels])
            )

            class _Out:
                pass

            out = _Out()
            out.logits = torch.from_numpy(logits)
            out.pred_boxes = torch.from_numpy(pred_boxes)
            results = image_processor.post_process_object_detection(
                out, threshold=threshold, target_sizes=target_sizes,
                use_focal_loss=True,
            )

            for target, result in zip(batch_labels, results):
                # scalar under transformers 5.1, 1-element array under 5.11
                image_id = int(np.asarray(target["image_id"]).reshape(-1)[0])
                h, w = (int(v) for v in target["orig_size"])
                gt_images.append({"id": image_id, "width": w, "height": h})
                # boxes in targets are normalized cxcywh on the resized image;
                # convert back to absolute xywh on the original size for COCO
                boxes = torch.from_numpy(np.asarray(target["boxes"])).reshape(-1, 4)
                cx, cy, bw, bh = boxes.unbind(-1)
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                for xi, yi, wi, hi in zip(x, y, bw * w, bh * h):
                    gt_anns.append({
                        "id": ann_id, "image_id": image_id, "category_id": 0,
                        "bbox": [float(xi), float(yi), float(wi), float(hi)],
                        "area": float(wi * hi), "iscrowd": 0,
                    })
                    ann_id += 1
                for score, label, box in zip(
                        result["scores"], result["labels"], result["boxes"]):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    dt.append({
                        "image_id": image_id, "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
                    })

        if not gt_anns:
            return {name: 0.0 for name in METRIC_NAMES}

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = {
                "images": gt_images,
                "annotations": gt_anns,
                "categories": [{"id": 0, "name": "stamp"}],
            }
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(dt) if dt else COCO()
            if not dt:
                coco_dt.dataset = {"images": gt_images, "annotations": [],
                                   "categories": [{"id": 0, "name": "stamp"}]}
                coco_dt.createIndex()
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        return {name: float(coco_eval.stats[i]) for i, name in enumerate(METRIC_NAMES)}

    return compute_metrics
