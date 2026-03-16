from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ExportService:
    def save_png(self, path: Path, image: np.ndarray, alpha_mask: np.ndarray) -> None:
        rgba = np.dstack(
            [
                image.astype(np.uint8),
                np.clip(alpha_mask * 255.0, 0, 255).astype(np.uint8),
            ]
        )
        Image.fromarray(rgba, mode="RGBA").save(path, format="PNG", optimize=True)

    def save_jpg(self, path: Path, image: np.ndarray, quality: int = 95) -> None:
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(
            path,
            format="JPEG",
            quality=max(70, min(100, int(quality))),
            subsampling=0,
            optimize=True,
        )
