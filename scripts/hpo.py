#!/usr/bin/env python
"""Hyperparameter optimization with Optuna (TPE sampler + ASHA pruning).

Each trial trains for hpo.trial_epochs on an optional subsample of the train set
and reports eval_map per epoch; pruned trials stop early. Every trial is its own
W&B run grouped under hpo-<study_name>. Results are stored in SQLite, so the
study is resumable: rerun the same command to continue.

Example:
    python scripts/hpo.py --set hpo.n_trials=30
"""

import argparse
import copy
import sys
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.training.callbacks import OptunaPruningCallback
from stamp_detection.training.run import run_training
from stamp_detection.utils import load_env


def suggest(trial: optuna.Trial, base_cfg) -> object:
    cfg = copy.deepcopy(base_cfg)
    cfg.train.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    cfg.train.backbone_lr_mult = trial.suggest_float("backbone_lr_mult", 0.01, 1.0, log=True)
    cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    cfg.train.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.10)
    cfg.aug.strength = trial.suggest_float("aug_strength", 0.0, 1.0)
    # trial-specific shortening
    cfg.train.epochs = cfg.hpo.trial_epochs
    cfg.train.early_stopping_patience = 0  # pruner handles early termination
    if cfg.hpo.max_train_samples:
        cfg.data.max_train_samples = cfg.hpo.max_train_samples
    cfg.wandb.group = f"hpo-{cfg.hpo.study_name}"
    cfg.wandb.tags = [*cfg.wandb.tags, "hpo", f"trial-{trial.number}"]
    cfg.wandb.n_val_images = 0  # skip image panels during HPO
    cfg.train.save_total_limit = 1
    return cfg


def make_objective(base_cfg):
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest(trial, base_cfg)
        pruning_cb = OptunaPruningCallback(trial, metric=cfg.train.metric_for_best)
        run_name = f"hpo-{cfg.hpo.study_name}-t{trial.number:03d}"
        cfg.train.run_name = run_name
        metrics = run_training(cfg, extra_callbacks=[pruning_cb], run_name=run_name)
        if pruning_cb.pruned:
            raise optuna.TrialPruned()
        return metrics[cfg.train.metric_for_best]

    return objective


def build_pruner(kind: str) -> optuna.pruners.BasePruner:
    if kind == "asha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=3)
    if kind == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=2)
    return optuna.pruners.NopPruner()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config, args.overrides)
    Path("runs").mkdir(exist_ok=True)

    study = optuna.create_study(
        study_name=cfg.hpo.study_name,
        storage=cfg.hpo.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.train.seed),
        pruner=build_pruner(cfg.hpo.pruner),
        load_if_exists=True,
    )
    study.optimize(make_objective(cfg), n_trials=cfg.hpo.n_trials)

    best = study.best_trial
    print(f"\n[hpo] best trial #{best.number}: eval_map={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k} = {v}")

    out_dir = Path("runs") / f"hpo-{cfg.hpo.study_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_yaml = out_dir / "best.yaml"
    best_yaml.write_text(yaml.safe_dump({
        "train": {
            "lr": best.params["lr"],
            "backbone_lr_mult": best.params["backbone_lr_mult"],
            "weight_decay": best.params["weight_decay"],
            "warmup_ratio": best.params["warmup_ratio"],
        },
        "aug": {"strength": best.params["aug_strength"]},
    }))
    print(f"\n[hpo] best params -> {best_yaml}")
    print("[hpo] full training command:")
    print(
        "  python scripts/train.py --set "
        f"train.lr={best.params['lr']:.3e} "
        f"train.backbone_lr_mult={best.params['backbone_lr_mult']:.3f} "
        f"train.weight_decay={best.params['weight_decay']:.3e} "
        f"train.warmup_ratio={best.params['warmup_ratio']:.3f} "
        f"aug.strength={best.params['aug_strength']:.2f}"
    )


if __name__ == "__main__":
    main()
