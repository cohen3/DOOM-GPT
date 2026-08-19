from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


WIDTH = 64
HEIGHT = 32


def target_gray(image: Image.Image | np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Convert an RGB/frame image into the shared grayscale float target in [0, 1]."""
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    return np.asarray(img.convert("L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def save_target(path: Path, values: np.ndarray) -> None:
    """Save a shared target array as fixed-scale 8-bit grayscale PNG."""
    Image.fromarray(np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8), mode="L").save(path)


def load_target(path: Path) -> np.ndarray:
    """Load a processed target PNG as float grayscale values in [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
