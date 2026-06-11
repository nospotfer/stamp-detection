#!/usr/bin/env python
"""Latency benchmark: ONNX Runtime (CUDA and CPU) vs PyTorch.

Example:
    python scripts/benchmark_onnx.py --onnx runs/rtdetrv2-stamp/best/model.onnx \
        --checkpoint runs/rtdetrv2-stamp/best
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stamp_detection.config import add_config_args, load_config
from stamp_detection.inference import load_checkpoint
from stamp_detection.utils import load_env

WARMUP = 10


def stats(times_ms: list[float]) -> str:
    t = np.asarray(times_ms)
    return f"mean {t.mean():7.2f} ms | p50 {np.percentile(t, 50):7.2f} | p95 {np.percentile(t, 95):7.2f}"


def bench_ort(onnx_path: str, provider: str, batch: int, size: int,
              iters: int) -> list[float] | None:
    import onnxruntime as ort

    if provider not in ort.get_available_providers():
        return None
    session = ort.InferenceSession(onnx_path, providers=[provider])
    feed = {"pixel_values": np.random.rand(batch, 3, size, size).astype(np.float32)}
    if any(i.name == "orig_target_sizes" for i in session.get_inputs()):
        feed["orig_target_sizes"] = np.full((batch, 2), size, dtype=np.int64)
    for _ in range(WARMUP):
        session.run(None, feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return times


@torch.no_grad()
def bench_torch(model, batch: int, size: int, device: str, bf16: bool,
                iters: int) -> list[float]:
    model = model.to(device).eval()
    pixels = torch.rand(batch, 3, size, size, device=device)
    ctx = torch.autocast(device, dtype=torch.bfloat16) if bf16 else torch.no_grad()
    with ctx:
        for _ in range(WARMUP):
            model(pixel_values=pixels)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(pixel_values=pixels)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Also benchmark the PyTorch model from this checkpoint")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 4, 8])
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config, args.overrides)
    size = cfg.data.image_size

    model = None
    if args.checkpoint:
        model, _, _ = load_checkpoint(args.checkpoint, device="cpu")

    for batch in args.batch_sizes:
        print(f"\n=== batch {batch} @ {size}x{size} ===")
        for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
            times = bench_ort(args.onnx, provider, batch, size, args.iters)
            if times is None:
                print(f"  ort {provider:25s} unavailable")
                continue
            ips = batch / (np.mean(times) / 1000)
            print(f"  ort {provider:25s} {stats(times)} | {ips:7.1f} img/s")
        if model is not None and torch.cuda.is_available():
            for bf16 in (True, False):
                times = bench_torch(model, batch, size, "cuda", bf16, args.iters)
                ips = batch / (np.mean(times) / 1000)
                name = f"torch cuda {'bf16' if bf16 else 'fp32'}"
                print(f"  {name:29s} {stats(times)} | {ips:7.1f} img/s")


if __name__ == "__main__":
    main()
