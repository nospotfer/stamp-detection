# Stamp Detection with RT-DETRv2

Train an [RT-DETRv2-L](https://huggingface.co/docs/transformers/en/model_doc/rt_detr_v2)
(`PekingU/rtdetr_v2_r50vd`) model to detect **stamps** in scanned documents
(mostly invoices). Single detection class, end-to-end pipeline:

- dataset download from Roboflow Universe + unification into one single-class COCO dataset
- training with HuggingFace `Trainer` (bf16, cosine LR, backbone LR multiplier, early stopping)
- document-specific augmentations (albumentations)
- COCO mAP evaluation + Weights & Biases logging with interactive detection panels
- hyperparameter optimization with Optuna (TPE + ASHA pruning)
- qualitative analysis with rendered detections, bucketed by failure type
- ONNX export with automatic PyTorch-parity verification + latency benchmark

## Setup

```bash
git clone <this repo> && cd stamp-detection

conda create -n stamp-detection python=3.12 -y
conda activate stamp-detection

# install torch for your CUDA version first (see https://pytorch.org/get-started/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cp .env.example .env   # then fill in your keys
```

`.env` keys:

| Key | Where to get it |
|---|---|
| `ROBOFLOW_API_KEY` | [Roboflow account settings](https://app.roboflow.com/settings/api) — needed to download datasets |
| `WANDB_API_KEY` | [wandb.ai/authorize](https://wandb.ai/authorize) — or run with `--set wandb.enabled=false` |

W&B entity/project are configured in `configs/default.yaml` (`wandb.entity`, `wandb.project`).

## 1. Prepare data

Candidate datasets (Roboflow Universe URLs) live in `datasets.txt`.

```bash
python scripts/prepare_data.py
```

This downloads the latest COCO export of every dataset into `datasets/raw/` and
merges them into `datasets/processed/{train,val,test}` with a single category,
`stamp`. The merge:

1. **Remaps classes** per `configs/dataset_class_map.yaml`: stamp subclasses
   (round/rectangular/oval/...) become `stamp`; other classes (signature, logo, ...)
   are dropped. Unknown class names abort the merge and are printed in
   copy-pasteable YAML form — map them and rerun with `--skip-download`.
2. **Sanitizes** boxes (clipped to image bounds, degenerate boxes removed).
3. **Deduplicates** across datasets — exact (SHA-256) plus near-duplicate (dHash)
   matching, since several candidate datasets are forks of the same images.
4. **Drops baked-in augmented copies**: Roboflow exports bake train-time
   augmentations (rotations with black fill, flips, noise) into the images,
   with up to 18 copies of the same document. Only one canonical copy per
   original is kept (preferring un-augmented ones, detected via the `.rf.`
   filename stem and black-corner heuristics); we re-apply better augmentations
   at train time instead. Set `data.keep_baked_augmentations=true` to keep them.
5. **Caps negatives**: images without stamps are kept (real "invoice without
   stamp" case) but limited to `data.negative_keep_ratio` of the positives.
6. **Re-splits** 80/10/10 deterministically *after* dedup and group-aware (all
   copies of one original stay in the same split), so no duplicate or augmented
   sibling of a train image can leak into val or test.

A `merge_report.json` with per-source/dedup/split statistics is written to
`datasets/processed/`.

**Local datasets** (e.g. an old Roboflow YOLO export on disk) can be unified
into the same pipeline — the directory is *moved* under `datasets/raw/` and its
labels converted to COCO, after which the merge treats it like any other source:

```bash
python scripts/import_local_dataset.py --path ~/datasets/my_yolo_export
# add the printed slug to configs/dataset_class_map.yaml, then:
python scripts/prepare_data.py --skip-download
```

**Inspecting the result**: render ground-truth samples grouped per source
dataset (one contact sheet each under `datasets/processed/preview/`):

```bash
python scripts/visualize_dataset.py --per-source 8
```

## 2. Train

```bash
python scripts/train.py
# any config value can be overridden on the CLI:
python scripts/train.py --set train.lr=5e-5 aug.strength=0.8 train.run_name=exp2
```

Defaults (see `configs/default.yaml`): bf16, batch 8 × grad-accum 2 (effective 16),
AdamW lr 1e-4 with backbone at 0.1×, cosine schedule with 5% warmup, grad clip 0.1,
up to 72 epochs with early stopping (patience 10) on validation mAP. Fits
comfortably on a single 24 GB GPU at 640×640.

Every epoch logs COCO metrics (`eval_map`, `eval_map_50`, `eval_map_75`, AR) to
W&B, plus an interactive panel of validation images with predicted and
ground-truth boxes. The best checkpoint is saved to
`runs/<run_name>/best/`.

## 3. Hyperparameter optimization

```bash
python scripts/hpo.py            # 30 trials x 10 epochs by default
```

Optuna (TPE sampler) searches `lr`, `backbone_lr_mult`, `weight_decay`,
`warmup_ratio` and `aug.strength`; ASHA pruning kills weak trials after 2
evaluations. The study is stored in `runs/hpo.db` (rerun the command to resume),
each trial is a W&B run grouped under `hpo-<study_name>`, and the best
parameters are written to `runs/hpo-<study>/best.yaml` together with the exact
`train.py` command to launch the full run.

## 4. Evaluate & inspect

```bash
python scripts/evaluate.py    --checkpoint runs/rtdetrv2-stamp/best --split test
python scripts/qualitative.py --checkpoint runs/rtdetrv2-stamp/best --split test --wandb
```

`qualitative.py` renders ground truth (green) vs predictions (red + score) and
buckets images into `tp_clean/`, `with_fn/` (missed stamps) and `with_fp/`
(spurious detections) for fast failure browsing, with an `index.csv` summary.
`--side-by-side` renders GT and predictions as separate panels; `--wandb` logs
a gallery table.

## 5. Export to ONNX

```bash
python scripts/export_onnx.py    --checkpoint runs/rtdetrv2-stamp/best
python scripts/benchmark_onnx.py --onnx runs/rtdetrv2-stamp/best/model.onnx \
                                 --checkpoint runs/rtdetrv2-stamp/best
```

Export is dynamic-batch / static 640×640 and automatically verifies parity
against PyTorch on real validation images. Two flavors:

- **raw** (default): outputs `logits [B,300,1]` and `pred_boxes [B,300,4]`
  (normalized cxcywh).
- **`--with-postprocessing`**: the graph takes an extra `orig_target_sizes [B,2]`
  (height, width) input and directly outputs absolute `boxes` (xyxy), `scores`
  and `labels`.

Consuming the **raw** model:

```python
import numpy as np, onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("model.onnx")
image = Image.open("invoice.jpg").convert("RGB")
W, H = image.size
x = np.asarray(image.resize((640, 640)), np.float32).transpose(2, 0, 1)[None] / 255.0
logits, boxes = session.run(None, {"pixel_values": x})

scores = 1 / (1 + np.exp(-logits[0, :, 0]))   # sigmoid: focal-loss head, NOT softmax
keep = scores > 0.5                            # RT-DETR is end-to-end: NO NMS needed
cx, cy, w, h = boxes[0, keep].T
xyxy = np.stack([(cx - w/2) * W, (cy - h/2) * H, (cx + w/2) * W, (cy + h/2) * H], 1)
```

Note: the model expects pixels rescaled to `[0, 1]` **without** ImageNet
normalization (RT-DETR convention).

**TF32 caveat (ONNX Runtime CUDA):** on Ampere+ GPUs ORT enables TF32 by
default, which adds ~1e-2 matmul noise — enough to reshuffle RT-DETR's
internal top-300 query selection between runs/devices. Detection quality is
unaffected, but if you need bit-stable outputs (e.g. regression tests), create
the session with `("CUDAExecutionProvider", {"use_tf32": "0"})`. The exporter's
built-in parity check does this automatically.

## Augmentations

Tuned for stamps on scanned documents (`stamp_detection/data/transforms.py`);
one knob, `aug.strength` ∈ [0, 1], scales all magnitudes:

| Augmentation | Why |
|---|---|
| Affine (rotate ±5°, scale, translate, **white** fill) | scan skew; document background is paper, never black |
| Perspective (p=0.2) | photographed rather than flatbed-scanned documents |
| BBox-safe random crop | scale variance without ever cropping a stamp away |
| Brightness/contrast | scanner exposure variance |
| Hue/saturation shift | ink colors: blue, red, purple, black |
| ToGray (p=0.15) | grayscale/B&W scans are common for invoices |
| Gaussian blur + noise | low-DPI scans, sensor noise |
| JPEG compression (q 50–95) | re-compression artifacts in document pipelines |
| Downscale (p=0.1) | fax-grade rescans |
| Horizontal flip | **disabled by default**: stamps contain text, and mirrored glyphs never occur in real scans, so hflip teaches a false invariance. Vertical flip is never used. If recall on symmetric round stamps lags, `--set aug.hflip_p=0.3` is a safe experiment. |

## Repo layout

```
configs/            default.yaml (all knobs) + dataset_class_map.yaml
stamp_detection/    package: config, data pipeline, training, inference, drawing
scripts/            prepare_data / train / hpo / evaluate / qualitative / export_onnx / benchmark_onnx
datasets/           (gitignored) raw/ + processed/
runs/               (gitignored) checkpoints, HPO study, exports
```

## License

See [LICENSE](LICENSE). The candidate datasets come from
[Roboflow Universe](https://universe.roboflow.com) — check each dataset's own
license before commercial use.
