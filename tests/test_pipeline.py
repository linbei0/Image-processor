from pathlib import Path

import numpy as np
from PIL import Image

from core.pipeline import ProcessingPipeline
from core.models import BackgroundColor, ExportFormat, HardwareMode, ProcessingRequest


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
        alpha[2:-2, 2:-2] = 1.0
        return alpha


def create_sample_image(path: Path) -> None:
    image = Image.new("RGB", (24, 24), (10, 20, 30))
    image.save(path)


def test_pipeline_processes_image_and_reuses_cached_alpha(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    create_sample_image(image_path)
    engine = FakeEngine()
    pipeline = ProcessingPipeline(engine=engine)

    first = pipeline.process(
        ProcessingRequest(
            input_path=image_path,
            bg_color=BackgroundColor.BLUE,
            output_format=ExportFormat.PNG,
            hardware_mode=HardwareMode.AUTO,
        )
    )
    second = pipeline.process(
        ProcessingRequest(
            input_path=image_path,
            bg_color=BackgroundColor.RED,
            output_format=ExportFormat.JPG,
            hardware_mode=HardwareMode.AUTO,
        )
    )

    assert engine.calls == 1
    assert first.provider == "CPUExecutionProvider"
    assert first.final_image.shape == (24, 24, 3)
    assert second.final_image.shape == (24, 24, 3)
    assert first.preview_image.shape[0] <= 1024
