"""Shared helpers: env loading, seeding, image hashing."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load .env from the repo root (keys: ROBOFLOW_API_KEY, WANDB_API_KEY, ...)."""
    load_dotenv(REPO_ROOT / ".env")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable {name}. "
            f"Add it to {REPO_ROOT / '.env'} (see .env.example)."
        )
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash: robust to resizing/re-encoding, used for near-dup detection."""
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return int.from_bytes(np.packbits(diff.flatten()).tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
