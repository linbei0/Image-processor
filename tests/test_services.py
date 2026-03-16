from pathlib import Path

import numpy as np
from PIL import Image

from core.export import ExportService
from core.image_ops import BackgroundComposer, EdgeRefiner
from core.models import BackgroundColor


def test_background_composer_blends_foreground_and_background() -> None:
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    alpha = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)

    output = BackgroundComposer.compose(image, alpha, BackgroundColor.WHITE.rgb)

    assert tuple(output[0, 0]) == (255, 0, 0)
    assert tuple(output[1, 0]) == (255, 255, 255)
    assert tuple(output[0, 1]) == (127, 255, 127)


def test_edge_refiner_preserves_range_and_shape() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    alpha = np.zeros((16, 16), dtype=np.float32)
    alpha[4:12, 4:12] = 1.0

    refined = EdgeRefiner(level=60).refine(image, alpha)

    assert refined.shape == alpha.shape
    assert refined.dtype == np.float32
    assert refined.min() >= 0.0
    assert refined.max() <= 1.0


def test_export_service_saves_png_and_jpg(tmp_path: Path) -> None:
    image = np.full((10, 10, 3), 200, dtype=np.uint8)
    alpha = np.full((10, 10), 0.5, dtype=np.float32)
    exporter = ExportService()

    png_path = tmp_path / "result.png"
    jpg_path = tmp_path / "result.jpg"
    exporter.save_png(png_path, image, alpha)
    exporter.save_jpg(jpg_path, image, quality=92)

    png = Image.open(png_path)
    jpg = Image.open(jpg_path)

    assert png.mode == "RGBA"
    assert jpg.mode == "RGB"
