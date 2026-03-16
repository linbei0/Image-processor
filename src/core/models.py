from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class BackgroundColor(str, Enum):
    WHITE = "white"
    BLUE = "blue"
    RED = "red"

    @property
    def rgb(self) -> tuple[int, int, int]:
        return {
            BackgroundColor.WHITE: (255, 255, 255),
            BackgroundColor.BLUE: (67, 142, 219),
            BackgroundColor.RED: (224, 62, 62),
        }[self]


class ExportFormat(str, Enum):
    PNG = "png"
    JPG = "jpg"


class HardwareMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compose_result_mask(
    auto_mask: np.ndarray,
    manual_contour_mask: np.ndarray | None,
    manual_brush_add_mask: np.ndarray | None,
    manual_brush_erase_mask: np.ndarray | None,
    legacy_manual_mask: np.ndarray | None,
) -> np.ndarray:
    if legacy_manual_mask is not None and all(
        value is None for value in (manual_contour_mask, manual_brush_add_mask, manual_brush_erase_mask)
    ):
        return legacy_manual_mask.astype(np.float32)
    base = manual_contour_mask.astype(np.float32).copy() if manual_contour_mask is not None else auto_mask.astype(np.float32).copy()
    if manual_brush_add_mask is not None:
        base = np.maximum(base, manual_brush_add_mask.astype(np.float32))
    if manual_brush_erase_mask is not None:
        base = np.where(manual_brush_erase_mask > 0.5, 0.0, base)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


@dataclass(slots=True)
class ProcessingRequest:
    input_path: Path
    bg_color: BackgroundColor = BackgroundColor.BLUE
    output_format: ExportFormat = ExportFormat.PNG
    edge_refine_level: int = 50
    hardware_mode: HardwareMode = HardwareMode.AUTO

    def __post_init__(self) -> None:
        if self.input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("仅支持 JPG/JPEG/PNG 图片文件")
        self.edge_refine_level = int(max(0, min(100, self.edge_refine_level)))


@dataclass(slots=True)
class ProcessingResult:
    preview_image: np.ndarray
    final_image: np.ndarray
    alpha_mask: np.ndarray
    elapsed_ms: float
    provider: str
    source_image: np.ndarray | None = None
    manual_mask: np.ndarray | None = None
    manual_contour_mask: np.ndarray | None = None
    manual_brush_add_mask: np.ndarray | None = None
    manual_brush_erase_mask: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def active_mask(self) -> np.ndarray:
        return compose_result_mask(
            auto_mask=self.alpha_mask,
            manual_contour_mask=self.manual_contour_mask,
            manual_brush_add_mask=self.manual_brush_add_mask,
            manual_brush_erase_mask=self.manual_brush_erase_mask,
            legacy_manual_mask=self.manual_mask,
        )
