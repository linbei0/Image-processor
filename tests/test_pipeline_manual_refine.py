from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from core.manual_refine import ManualEditResult
from core.models import BackgroundColor, ExportFormat, HardwareMode, ProcessingRequest
from core.pipeline import ProcessingPipeline


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def get_available_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def load(self, hardware_mode: HardwareMode) -> str:
        return "CPUExecutionProvider"

    def warmup(self) -> None:
        return None

    def infer(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        alpha = np.zeros(image.shape[:2], dtype=np.float32)
        alpha[4:-4, 4:-4] = 1.0
        return alpha


def create_sample_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), (20, 30, 40))
    image.save(path)


def test_pipeline_applies_manual_mask_without_rerunning_engine(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    create_sample_image(image_path)
    engine = FakeEngine()
    pipeline = ProcessingPipeline(engine=engine)
    request = ProcessingRequest(
        input_path=image_path,
        bg_color=BackgroundColor.BLUE,
        output_format=ExportFormat.PNG,
        hardware_mode=HardwareMode.AUTO,
    )
    base = pipeline.process(request)
    manual_mask = np.zeros_like(base.alpha_mask)
    manual_mask[8:-8, 8:-8] = 1.0

    updated = pipeline.apply_manual_mask(request, base, manual_mask)

    assert engine.calls == 1
    assert updated.manual_mask is not None
    assert updated.active_mask.shape == base.alpha_mask.shape
    assert updated.active_mask[16, 16] > 0.9
    assert updated.active_mask[0, 0] == 0.0


def test_pipeline_applies_manual_edit_result_without_rerunning_engine(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    create_sample_image(image_path)
    engine = FakeEngine()
    pipeline = ProcessingPipeline(engine=engine)
    request = ProcessingRequest(
        input_path=image_path,
        bg_color=BackgroundColor.RED,
        output_format=ExportFormat.PNG,
        hardware_mode=HardwareMode.AUTO,
    )
    base = pipeline.process(request)
    contour_mask = np.zeros_like(base.alpha_mask)
    contour_mask[6:-6, 6:-6] = 1.0
    add_mask = np.zeros_like(base.alpha_mask)
    add_mask[1:5, 1:5] = 1.0
    erase_mask = np.zeros_like(base.alpha_mask)
    erase_mask[12:18, 12:18] = 1.0
    active_mask = np.where(erase_mask > 0.5, 0.0, np.maximum(contour_mask, add_mask)).astype(np.float32)
    edit_result = ManualEditResult(
        auto_mask=base.alpha_mask,
        contour_mask=contour_mask,
        brush_add_mask=add_mask,
        brush_erase_mask=erase_mask,
        active_mask=active_mask,
    )

    updated = pipeline.apply_manual_edit(request, base, edit_result)

    assert engine.calls == 1
    assert updated.manual_contour_mask is not None
    assert updated.manual_brush_add_mask is not None
    assert updated.manual_brush_erase_mask is not None
