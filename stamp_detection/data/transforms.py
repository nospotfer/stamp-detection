"""Albumentations pipelines for stamps on scanned documents.

Design notes:
- Geometric fills use white (255): the background of a document is paper, never black.
- No vertical flip, and horizontal flip defaults to p=0 — stamps contain text and
  mirrored glyphs do not occur in real scans (see README for discussion).
- Photometric augs model the real degradation axes of document pipelines:
  scanner exposure, ink color (blue/red/purple/black), grayscale scans, low DPI,
  JPEG re-compression, fax-grade rescans.
- Magnitudes scale linearly with cfg.strength in [0, 1]; defaults documented at 0.5.
- Final Resize to the square model input lives here, so the HF image processor
  runs with do_resize=False.
"""

from __future__ import annotations

import albumentations as A

from stamp_detection.config import AugConfig


def _bbox_params(cfg: AugConfig) -> A.BboxParams:
    return A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_visibility=cfg.min_visibility,
        clip=True,
    )


def build_train_transforms(cfg: AugConfig, image_size: int) -> A.Compose:
    s = cfg.strength / 0.5  # 1.0 at the documented default strength
    transforms = [
        A.Affine(
            rotate=(-cfg.rotate_deg * s, cfg.rotate_deg * s),
            scale=(1 - (1 - cfg.scale_low) * s, 1 + (cfg.scale_high - 1) * s),
            translate_percent=(-cfg.translate_pct * s, cfg.translate_pct * s),
            fill=255,
            p=0.7 if cfg.strength > 0 else 0.0,
        ),
        A.Perspective(scale=(0.02, 0.02 + 0.03 * s), fill=255, p=cfg.perspective_p),
        A.RandomSizedBBoxSafeCrop(
            height=image_size, width=image_size, erosion_rate=0.1,
            p=cfg.bbox_safe_crop_p,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=cfg.brightness_contrast * s,
            contrast_limit=cfg.brightness_contrast * s,
            p=cfg.brightness_contrast_p,
        ),
        A.HueSaturationValue(
            hue_shift_limit=round(cfg.hue_shift * s),
            sat_shift_limit=round(cfg.sat_shift * s),
            val_shift_limit=round(cfg.val_shift * s),
            p=cfg.hsv_p,
        ),
        A.ToGray(p=cfg.gray_p),
        A.GaussianBlur(blur_limit=(3, 5), p=cfg.blur_p),
        A.GaussNoise(std_range=(0.02, min(0.02 + 0.06 * s, 0.3)), p=cfg.noise_p),
        A.ImageCompression(
            quality_range=(cfg.jpeg_quality_low, 95), p=cfg.jpeg_p,
        ),
        A.Downscale(scale_range=(0.5, 0.9), p=cfg.downscale_p),
        A.HorizontalFlip(p=cfg.hflip_p),
        A.Resize(height=image_size, width=image_size),
    ]
    if cfg.strength == 0:
        transforms = [A.Resize(height=image_size, width=image_size)]
    return A.Compose(transforms, bbox_params=_bbox_params(cfg))


def build_eval_transforms(cfg: AugConfig, image_size: int) -> A.Compose:
    # Square resize without padding mirrors the official RT-DETR recipe.
    return A.Compose(
        [A.Resize(height=image_size, width=image_size)],
        bbox_params=_bbox_params(cfg),
    )
