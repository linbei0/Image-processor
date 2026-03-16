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
    warnings: list[str] = field(default_factory=list)

    @property
    def active_mask(self) -> np.ndarray:
        return self.manual_mask if self.manual_mask is not None else self.alpha_mask
