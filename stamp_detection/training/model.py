"""Model and optimizer construction for RT-DETRv2 fine-tuning."""

from __future__ import annotations

import torch
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

from stamp_detection.config import ExperimentConfig


def build_model(cfg: ExperimentConfig) -> RTDetrV2ForObjectDetection:
    model = RTDetrV2ForObjectDetection.from_pretrained(
        cfg.model.checkpoint,
        num_labels=cfg.model.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
        ignore_mismatched_sizes=True,  # class head is re-initialized 80 -> 1
    )
    if cfg.model.freeze_backbone:
        for p in model.model.backbone.parameters():
            p.requires_grad = False
    return model


def build_image_processor(cfg: ExperimentConfig):
    # RT-DETR rescales to [0,1] without ImageNet normalization; keep defaults.
    return AutoImageProcessor.from_pretrained(
        cfg.model.checkpoint,
        use_fast=True,
        size={"height": cfg.data.image_size, "width": cfg.data.image_size},
    )


def build_optimizer(model: torch.nn.Module, cfg: ExperimentConfig) -> torch.optim.AdamW:
    """AdamW with the DETR-family recipe: backbone at lr * mult, no decay on norm/bias."""
    decay, no_decay, bb_decay, bb_no_decay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "backbone" in name
        skip_decay = param.ndim <= 1 or name.endswith(".bias")
        if is_backbone:
            (bb_no_decay if skip_decay else bb_decay).append(param)
        else:
            (no_decay if skip_decay else decay).append(param)
    backbone_lr = cfg.train.lr * cfg.train.backbone_lr_mult
    groups = [
        {"params": decay, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "lr": cfg.train.lr, "weight_decay": 0.0},
        {"params": bb_decay, "lr": backbone_lr, "weight_decay": cfg.train.weight_decay},
        {"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.999))
