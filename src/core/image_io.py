from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(corrected, dtype=np.uint8)


def resize_long_side(image: np.ndarray, max_long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def create_preview(image: np.ndarray, max_side: int = 960) -> np.ndarray:
    return resize_long_side(image, max_side)
