# CLAUDE.md

RT-DETRv2-L fine-tuning pipeline for single-class stamp detection in scanned
documents. See README.md for the full workflow.

## Environment

- Use the **`optipix`** conda env: `conda run -n optipix python ...`
  (outside contributors: create one per README setup).
- Secrets live in `.env` (`ROBOFLOW_API_KEY`, `WANDB_API_KEY`) — never commit it.
- W&B entity `aiaccount`, project `stamp-detection` (set in `configs/default.yaml`).
- Single RTX 3090 (24 GB); defaults are sized for it.

## Commands

```bash
conda run -n optipix python scripts/prepare_data.py              # download + merge
conda run -n optipix python scripts/train.py                     # full training
conda run -n optipix python scripts/hpo.py                       # Optuna search
conda run -n optipix python scripts/evaluate.py    --checkpoint runs/<run>/best --split test
conda run -n optipix python scripts/qualitative.py --checkpoint runs/<run>/best --split test
conda run -n optipix python scripts/export_onnx.py --checkpoint runs/<run>/best
```

Smoke test (fast, no W&B):

```bash
conda run -n optipix python scripts/train.py --set train.epochs=2 train.batch_size=2 \
    data.max_train_samples=64 data.max_eval_samples=32 wandb.enabled=false train.num_workers=2
```

## Conventions

- **Config**: one schema in `stamp_detection/config.py`, defaults in
  `configs/default.yaml`, every script accepts `--config` and
  `--set section.key=value ...` overrides. Add new knobs to the dataclass first.
- Dataset class-name decisions live in `configs/dataset_class_map.yaml`
  (`<class>: stamp` or `<class>: DROP`); the merge hard-fails on unmapped names
  by design.
- `datasets/`, `runs/`, `wandb/` are gitignored — never commit data, checkpoints
  or ONNX files.
- transformers 5.x notes baked into this repo: `eval_strategy` (not
  `evaluation_strategy`), `processing_class=` (not `tokenizer=`),
  `remove_unused_columns=False` and `eval_do_concat_batches=False` are required
  for object detection; the RT-DETR image processor rescales to [0,1] with no
  ImageNet normalization.
- Best checkpoint of a run: `runs/<run_name>/best/`.
