from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class BackgroundComposer:
    @staticmethod
    def compose(
        image: np.ndarray,
        alpha_mask: np.ndarray,
        bg_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        alpha = np.clip(alpha_mask.astype(np.float32), 0.0, 1.0)[..., None]
        background = np.full_like(image, bg_rgb, dtype=np.uint8)
        output = image.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
        return np.clip(output, 0, 255).astype(np.uint8)


@dataclass(slots=True)
class EdgeRefiner:
    level: int = 50

    def refine(self, image: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
        del image
        alpha = np.clip(alpha_mask.astype(np.float32), 0.0, 1.0)
        binary = (alpha > 0.05).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = (labels == largest).astype(np.uint8)
        kernel = max(1, int(round(self.level / 25)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
        alpha = np.where(binary > 0, alpha, 0.0).astype(np.float32)
        blur_size = max(1, kernel * 2 + 1)
        alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), sigmaX=0)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)
