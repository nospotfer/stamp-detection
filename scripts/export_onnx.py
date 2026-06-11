#!/usr/bin/env python
"""Export a trained RT-DETRv2 checkpoint to ONNX and verify parity against PyTorch.

Two modes:
- raw (default): outputs logits [B,300,1] and pred_boxes [B,300,4] (normalized
  cxcywh). Consumers apply sigmoid (focal-loss head - NOT softmax), threshold,
  convert cxcywh -> xyxy and scale by original image size. RT-DETR is end-to-end:
  NO NMS is needed.
- --with-postprocessing: wraps the model so the graph itself outputs absolute
  xyxy boxes, scores and labels given an extra orig_target_sizes [B,2] input.

Input is dynamic-batch, static 640x640 spatial (anchors derive from feature-map
shapes; re-export for a different input size).

Example:
    python scripts/export_onnx.py --checkpoint runs/rtdetrv2-stamp/best
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.inference import load_checkpoint
from stamp_detection.utils import load_env


class PostprocessWrapper(torch.nn.Module):
    """Embeds sigmoid + top-k + cxcywh->abs-xyxy so consumers get final boxes."""

    def __init__(self, model, num_top_queries: int = 300):
        super().__init__()
        self.model = model
        self.num_top_queries = num_top_queries

    def forward(self, pixel_values: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values)
        logits, boxes = outputs.logits, outputs.pred_boxes
        # cxcywh (normalized) -> xyxy (absolute)
        cxcy, wh = boxes[..., :2], boxes[..., 2:]
        xyxy = torch.cat([cxcy - wh / 2, cxcy + wh / 2], dim=-1)
        scale = orig_target_sizes.flip(-1).repeat(1, 2).unsqueeze(1).to(xyxy.dtype)
        xyxy = xyxy * scale
        scores_all = logits.sigmoid()  # focal-loss head
        scores, flat_idx = scores_all.flatten(1).topk(self.num_top_queries, dim=1)
        num_classes = logits.shape[2]
        labels = flat_idx % num_classes
        query_idx = flat_idx // num_classes
        boxes_out = xyxy.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, 4))
        return boxes_out, scores, labels


def export(model, out_path: Path, image_size: int, with_post: bool,
           opset: int, use_dynamo: bool) -> None:
    model = model.cpu().eval().float()
    dummy_pixels = torch.randn(1, 3, image_size, image_size)
    if with_post:
        wrapper = PostprocessWrapper(model)
        args = (dummy_pixels, torch.tensor([[image_size, image_size]], dtype=torch.int64))
        input_names = ["pixel_values", "orig_target_sizes"]
        output_names = ["boxes", "scores", "labels"]
        dynamic_axes = {"pixel_values": {0: "batch"}, "orig_target_sizes": {0: "batch"},
                        "boxes": {0: "batch"}, "scores": {0: "batch"},
                        "labels": {0: "batch"}}
    else:
        class RawWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                out = self.m(pixel_values=pixel_values)
                return out.logits, out.pred_boxes

        wrapper = RawWrapper(model)
        args = (dummy_pixels,)
        input_names = ["pixel_values"]
        output_names = ["logits", "pred_boxes"]
        dynamic_axes = {"pixel_values": {0: "batch"}, "logits": {0: "batch"},
                        "pred_boxes": {0: "batch"}}

    torch.onnx.export(
        wrapper, args, str(out_path),
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset,
        do_constant_folding=True, dynamo=use_dynamo,
    )
    import onnx

    onnx.checker.check_model(str(out_path))
    print(f"[export] ONNX model -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB, opset {opset}, "
          f"{'with' if with_post else 'raw'} postprocessing)")


def _raw_to_dets(logits, boxes_cxcywh, score_thr):
    """(B, Q, C) logits + normalized cxcywh -> per-image [(score, xyxy), ...]."""
    scores = 1.0 / (1.0 + np.exp(-logits.max(-1)))
    out = []
    for b in range(scores.shape[0]):
        keep = scores[b] > score_thr
        cx, cy, w, h = boxes_cxcywh[b][keep].T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        out.append(list(zip(scores[b][keep], xyxy)))
    return out


def _detections_match(torch_dets, ort_dets, d_score: float = 5e-3,
                      min_iou: float = 0.95) -> bool:
    """Permutation-invariant check: every torch detection must have an ORT
    counterpart with near-identical score and box (IoU-based). Raw query order
    is not comparable: RT-DETR's internal top-300 selection reorders under tiny
    numeric noise without changing the detections."""
    from stamp_detection.inference import box_iou

    all_ok = True
    for img_i, (t_dets, o_dets) in enumerate(zip(torch_dets, ort_dets)):
        unmatched = list(range(len(o_dets)))
        misses = 0
        for ts, tb in t_dets:
            hit = next((j for j in unmatched
                        if abs(float(ts) - float(o_dets[j][0])) < d_score
                        and box_iou(list(tb), list(o_dets[j][1])) > min_iou), None)
            if hit is None:
                misses += 1
            else:
                unmatched.remove(hit)
        if misses or len(t_dets) != len(o_dets):
            all_ok = False
        print(f"[parity]   image {img_i}: torch {len(t_dets)} dets / "
              f"ort {len(o_dets)} dets, unmatched {misses}")
    return all_ok


def parity_check(model, onnx_path: Path, processed_dir: Path, image_size: int,
                 with_post: bool, n_images: int = 8) -> None:
    """Compare torch fp32 vs onnxruntime on real validation images."""
    import onnxruntime as ort

    val_dir = processed_dir / "val"
    image_paths = sorted(val_dir.glob("*.jpg"))[:n_images] or \
        sorted(p for p in val_dir.iterdir() if p.suffix in (".jpg", ".jpeg", ".png"))[:n_images]
    if not image_paths:
        print("[parity] WARNING: no val images found, using random tensor")
        pixels = torch.rand(2, 3, image_size, image_size)
    else:
        arrays = []
        for p in image_paths:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB").resize((image_size, image_size)))
            arrays.append(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)
        pixels = torch.from_numpy(np.stack(arrays))

    # TF32 (default on Ampere+ in ORT CUDA) adds ~1e-2 matmul noise, enough to
    # flip RT-DETR's near-tied top-300 query selection and scramble the output
    # order. Disable it for the parity check; see README for deployment notes.
    providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"[parity] ORT providers: {session.get_providers()}")

    model = model.cpu().eval().float()
    with torch.no_grad():
        torch_out = model(pixel_values=pixels)

    if with_post:
        sizes = np.full((len(pixels), 2), image_size, dtype=np.int64)
        ort_boxes, ort_scores, _ = session.run(
            None, {"pixel_values": pixels.numpy(), "orig_target_sizes": sizes})
        wrapper = PostprocessWrapper(model)
        with torch.no_grad():
            t_boxes, t_scores, _ = wrapper(pixels, torch.from_numpy(sizes))
        # compare as detection sets; boxes already absolute xyxy
        thr = 0.1

        def post_dets(scores, boxes):
            return [[(s, b) for s, b in zip(scores[i], boxes[i]) if s > thr]
                    for i in range(len(scores))]

        ok = _detections_match(post_dets(t_scores.numpy(), t_boxes.numpy()),
                               post_dets(ort_scores, ort_boxes))
    else:
        ort_logits, ort_boxes = session.run(None, {"pixel_values": pixels.numpy()})
        d_logits = float(np.abs(ort_logits - torch_out.logits.numpy()).max())
        d_boxes = float(np.abs(ort_boxes - torch_out.pred_boxes.numpy()).max())
        print(f"[parity] raw tensors (query order): max |dLogits|={d_logits:.2e}, "
              f"max |dBoxes|={d_boxes:.2e} (informational; near-tied queries may permute)")
        ok = _detections_match(
            _raw_to_dets(torch_out.logits.numpy(), torch_out.pred_boxes.numpy(), 0.1),
            _raw_to_dets(ort_logits, ort_boxes, 0.1))

    if ok:
        print("[parity] PASS: ONNX output matches PyTorch")
    else:
        raise SystemExit("[parity] FAIL: ONNX output deviates from PyTorch beyond tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None,
                        help="Output path (default: <checkpoint>/model.onnx)")
    parser.add_argument("--with-postprocessing", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo", action="store_true",
                        help="Use the torch.export-based ONNX exporter")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config, args.overrides)
    model, _, _ = load_checkpoint(args.checkpoint, device="cpu")

    out_path = Path(args.out or Path(args.checkpoint) / "model.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(model, out_path, cfg.data.image_size, args.with_postprocessing,
           args.opset, args.dynamo)
    if not args.skip_parity:
        parity_check(model, out_path, Path(cfg.data.processed_dir),
                     cfg.data.image_size, args.with_postprocessing)


if __name__ == "__main__":
    main()
