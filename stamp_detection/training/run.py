"""Shared training entrypoint used by scripts/train.py and scripts/hpo.py."""

from __future__ import annotations

import os
from pathlib import Path

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from stamp_detection.config import ExperimentConfig
from stamp_detection.data.dataset import CocoStampDataset, collate_fn
from stamp_detection.data.transforms import build_eval_transforms, build_train_transforms
from stamp_detection.training.callbacks import WandbValImagesCallback
from stamp_detection.training.metrics import build_compute_metrics
from stamp_detection.training.model import build_image_processor, build_model, build_optimizer
from stamp_detection.utils import set_seed


def setup_wandb(cfg: ExperimentConfig, run_name: str | None = None) -> None:
    """Configure W&B via env vars; Trainer's WandbCallback does the init."""
    if not cfg.wandb.enabled:
        os.environ["WANDB_MODE"] = "disabled"
        return
    os.environ.pop("WANDB_MODE", None)
    os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.wandb.group:
        os.environ["WANDB_RUN_GROUP"] = cfg.wandb.group
    if run_name:
        os.environ["WANDB_NAME"] = run_name


def run_training(cfg: ExperimentConfig, extra_callbacks: list | None = None,
                 run_name: str | None = None) -> dict:
    """Train RT-DETRv2 with the given config; returns final eval metrics.

    The best checkpoint (by cfg.train.metric_for_best) is loaded at the end and
    saved to <output_dir>/<run_name>/best.
    """
    set_seed(cfg.train.seed)
    run_name = run_name or cfg.train.run_name or "rtdetrv2-stamp"
    output_dir = Path(cfg.train.output_dir) / run_name
    setup_wandb(cfg, run_name)

    image_processor = build_image_processor(cfg)
    model = build_model(cfg)

    processed = Path(cfg.data.processed_dir)
    train_ds = CocoStampDataset(
        processed / "train",
        build_train_transforms(cfg.aug, cfg.data.image_size),
        image_processor,
        max_samples=cfg.data.max_train_samples,
    )
    val_ds = CocoStampDataset(
        processed / "val",
        build_eval_transforms(cfg.aug, cfg.data.image_size),
        image_processor,
        max_samples=cfg.data.max_eval_samples,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.eval_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler,
        max_grad_norm=cfg.train.max_grad_norm,
        bf16=cfg.train.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.train.metric_for_best,
        greater_is_better=True,
        logging_steps=20,
        dataloader_num_workers=cfg.train.num_workers,
        remove_unused_columns=False,  # or Trainer strips the label dicts
        eval_do_concat_batches=False,  # keep per-batch nested outputs for COCO eval
        report_to=["wandb"] if cfg.wandb.enabled else [],
        seed=cfg.train.seed,
    )

    optimizer = build_optimizer(model, cfg)
    callbacks = []
    if cfg.train.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg.train.early_stopping_patience))
    if cfg.wandb.enabled and cfg.wandb.n_val_images > 0:
        callbacks.append(WandbValImagesCallback(cfg, image_processor, model))
    callbacks.extend(extra_callbacks or [])

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=image_processor,
        compute_metrics=build_compute_metrics(image_processor),
        optimizers=(optimizer, None),  # scheduler built by Trainer from args
        callbacks=callbacks,
    )

    trainer.train()

    # transformers 5.x saves RT-DETR checkpoints with legacy key names and
    # Trainer._load_best_model raw-loads them without the rename mapping,
    # leaving most transformer/head weights at their last-epoch values.
    # Reload the best checkpoint through from_pretrained, which does remap.
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        from transformers import RTDetrV2ForObjectDetection

        trainer.model = RTDetrV2ForObjectDetection.from_pretrained(best_ckpt).to(
            trainer.args.device)
        for cb in callbacks:
            if isinstance(cb, WandbValImagesCallback):
                cb.model = trainer.model
        print(f"[train] reloaded best checkpoint: {best_ckpt}")

    metrics = trainer.evaluate()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    image_processor.save_pretrained(str(best_dir))
    print(f"[train] best model saved to {best_dir}")
    print(f"[train] final metrics: " +
          ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("eval_ma")))

    if cfg.wandb.enabled:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    return metrics
