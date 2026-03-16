from pathlib import Path

import pytest

from core.config import AppConfig
from core.models import BackgroundColor, ExportFormat, HardwareMode, ProcessingRequest


def test_processing_request_rejects_non_image_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "note.txt"
    bad_file.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="仅支持"):
        ProcessingRequest(input_path=bad_file)


def test_processing_request_accepts_jpg_defaults(tmp_path: Path) -> None:
    image_file = tmp_path / "portrait.jpg"
    image_file.write_bytes(b"fake")

    request = ProcessingRequest(input_path=image_file)

    assert request.bg_color is BackgroundColor.BLUE
    assert request.output_format is ExportFormat.PNG
    assert request.hardware_mode is HardwareMode.AUTO
    assert request.edge_refine_level == 50


def test_app_config_uses_appdata_style_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))

    config = AppConfig.default()

    assert config.app_dir == tmp_path / "IDPhotoBackgroundTool"
    assert config.model_dir == config.app_dir / "models"
    assert config.cache_dir == config.app_dir / "cache"
