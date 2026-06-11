"""Configuration system: nested dataclasses loaded from YAML with dotted CLI overrides.

All scripts share one ExperimentConfig. Usage:

    cfg = load_config("configs/default.yaml", overrides=["train.lr=5e-5", "aug.strength=0.8"])
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "datasets/raw"
    processed_dir: str = "datasets/processed"
    datasets_file: str = "datasets.txt"
    class_map_file: str = "configs/dataset_class_map.yaml"
    image_size: int = 640
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    # test fraction is the remainder
    negative_keep_ratio: float = 0.1
    dedup_dhash_threshold: int = 3
    # Roboflow exports bake augmented copies (rotation/flip/noise) of each
    # original into the train split. False (default) keeps one canonical copy
    # per original; True keeps them all but splits group-aware (no leakage).
    keep_baked_augmentations: bool = False
    seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass
class ModelConfig:
    checkpoint: str = "PekingU/rtdetr_v2_r50vd"  # RT-DETRv2-L
    num_labels: int = 1
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    epochs: int = 72
    batch_size: int = 8
    grad_accum: int = 2
    eval_batch_size: int = 16
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 0.1
    bf16: bool = True
    early_stopping_patience: int = 10
    metric_for_best: str = "eval_map"
    num_workers: int = 8
    output_dir: str = "runs"
    run_name: str | None = None
    save_total_limit: int = 2
    seed: int = 42


@dataclass
class AugConfig:
    strength: float = 0.5
    hflip_p: float = 0.0  # stamps contain text; mirrored glyphs don't occur in real scans
    rotate_deg: float = 5.0
    scale_low: float = 0.85
    scale_high: float = 1.15
    translate_pct: float = 0.05
    perspective_p: float = 0.2
    bbox_safe_crop_p: float = 0.3
    brightness_contrast: float = 0.25
    brightness_contrast_p: float = 0.5
    hue_shift: int = 12
    sat_shift: int = 25
    val_shift: int = 10
    hsv_p: float = 0.4
    gray_p: float = 0.15
    blur_p: float = 0.2
    noise_p: float = 0.2
    jpeg_p: float = 0.3
    jpeg_quality_low: int = 50
    downscale_p: float = 0.1
    min_visibility: float = 0.3


@dataclass
class WandbConfig:
    enabled: bool = True
    entity: str = "aiaccount"
    project: str = "stamp-detection"
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    log_images_every_n_evals: int = 1
    n_val_images: int = 16
    image_log_threshold: float = 0.3


@dataclass
class HPOConfig:
    study_name: str = "stamp-rtdetrv2"
    storage: str = "sqlite:///runs/hpo.db"
    n_trials: int = 30
    trial_epochs: int = 10
    max_train_samples: int | None = 3000
    pruner: str = "asha"  # asha | median | none


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)

    @property
    def id2label(self) -> dict[int, str]:
        return {0: "stamp"}

    @property
    def label2id(self) -> dict[str, int]:
        return {"stamp": 0}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(value: str, target_type: Any) -> Any:
    """Coerce a CLI string to the type of the dataclass field it overrides."""
    if value.lower() in ("none", "null"):
        return None
    if target_type is bool or isinstance(target_type, bool):
        return value.lower() in ("1", "true", "yes")
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(path: str | Path | None = "configs/default.yaml",
                overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML config and apply 'a.b=value' overrides."""
    cfg = ExperimentConfig()
    if path is not None and Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for section_name, section_val in raw.items():
            if not hasattr(cfg, section_name):
                raise KeyError(f"Unknown config section '{section_name}' in {path}")
            section = getattr(cfg, section_name)
            for key, value in (section_val or {}).items():
                if not hasattr(section, key):
                    raise KeyError(f"Unknown config key '{section_name}.{key}' in {path}")
                setattr(section, key, value)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be 'a.b=value', got '{ov}'")
        dotted, value = ov.split("=", 1)
        parts = dotted.split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be 'section.key', got '{dotted}'")
        section_name, key = parts
        section = getattr(cfg, section_name)
        if not hasattr(section, key):
            raise KeyError(f"Unknown config key '{dotted}'")
        current = getattr(section, key)
        setattr(section, key, _coerce(value, type(current) if current is not None else str))
    return cfg


def add_config_args(parser) -> None:
    """Attach the shared --config/--set arguments to an argparse parser."""
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="Config overrides, e.g. --set train.lr=5e-5 aug.strength=0.8")
