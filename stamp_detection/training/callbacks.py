"""Trainer callbacks: W&B validation-image panels and Optuna pruning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import TrainerCallback

from stamp_detection.config import ExperimentConfig

Image.MAX_IMAGE_PIXELS = None


class WandbValImagesCallback(TrainerCallback):
    """After each evaluation, log a fixed panel of validation images with
    predicted and ground-truth boxes (interactive wandb box overlays)."""

    def __init__(self, cfg: ExperimentConfig, image_processor, model):
        self.cfg = cfg
        self.image_processor = image_processor
        self.model = model
        self._eval_count = 0
        self._samples = self._pick_samples()

    def _pick_samples(self) -> list[dict]:
        split_dir = Path(self.cfg.data.processed_dir) / "val"
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        anns_by_image = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a["bbox"])
        rng = np.random.RandomState(self.cfg.data.seed)
        images = sorted(coco["images"], key=lambda x: x["file_name"])
        # prefer images that contain stamps so the panel is informative
        with_boxes = [i for i in images if anns_by_image.get(i["id"])]
        pool = with_boxes if len(with_boxes) >= self.cfg.wandb.n_val_images else images
        idx = rng.choice(len(pool), size=min(self.cfg.wandb.n_val_images, len(pool)),
                         replace=False)
        return [
            {"path": split_dir / pool[i]["file_name"],
             "gt": anns_by_image.get(pool[i]["id"], [])}
            for i in sorted(idx)
        ]

    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs):
        import wandb

        if wandb.run is None:
            return
        self._eval_count += 1
        if (self._eval_count - 1) % self.cfg.wandb.log_images_every_n_evals != 0:
            return

        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        panel = []
        size = self.cfg.data.image_size
        for sample in self._samples:
            with Image.open(sample["path"]) as im:
                image = im.convert("RGB")
            w, h = image.size
            resized = image.resize((size, size))
            inputs = self.image_processor(images=resized, return_tensors="pt",
                                          do_resize=False, do_pad=False)
            outputs = self.model(pixel_values=inputs["pixel_values"].to(device))
            result = self.image_processor.post_process_object_detection(
                outputs, threshold=self.cfg.wandb.image_log_threshold,
                target_sizes=torch.tensor([[h, w]]), use_focal_loss=True,
            )[0]

            box_data_pred = [
                {"position": {"minX": float(b[0]) / w, "minY": float(b[1]) / h,
                              "maxX": float(b[2]) / w, "maxY": float(b[3]) / h},
                 "class_id": 0, "scores": {"conf": float(s)},
                 "box_caption": f"stamp {float(s):.2f}"}
                for b, s in zip(result["boxes"], result["scores"])
            ]
            box_data_gt = [
                {"position": {"minX": x / w, "minY": y / h,
                              "maxX": (x + bw) / w, "maxY": (y + bh) / h},
                 "class_id": 0, "box_caption": "stamp"}
                for x, y, bw, bh in sample["gt"]
            ]
            panel.append(wandb.Image(image, boxes={
                "predictions": {"box_data": box_data_pred, "class_labels": {0: "stamp"}},
                "ground_truth": {"box_data": box_data_gt, "class_labels": {0: "stamp"}},
            }))
        wandb.log({"val/detections": panel, "val/detections_global_step": state.global_step})
        if was_training:
            self.model.train()


class OptunaPruningCallback(TrainerCallback):
    """Report eval_map to Optuna each evaluation; stop the trial when pruned."""

    def __init__(self, trial, metric: str = "eval_map"):
        self.trial = trial
        self.metric = metric
        self.pruned = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric not in metrics:
            return
        self.trial.report(metrics[self.metric], step=round(state.epoch or 0))
        if self.trial.should_prune():
            self.pruned = True
            control.should_training_stop = True
