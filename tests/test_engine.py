from pathlib import Path

import pytest

from core.config import AppConfig
from core.models import HardwareMode
from inference.engine import MODEL_FILE_NAME, ModNetOnnxEngine


def test_engine_raises_clear_error_when_model_missing(tmp_path: Path) -> None:
    config = AppConfig(
        app_dir=tmp_path,
        model_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        settings_file=tmp_path / "settings.json",
        last_export_file=tmp_path / "last_export.txt",
    )
    missing_model = tmp_path / "models" / MODEL_FILE_NAME
    engine = ModNetOnnxEngine(config=config, model_path=missing_model)

    with pytest.raises(FileNotFoundError, match=MODEL_FILE_NAME):
        engine.load(HardwareMode.CPU)


def test_engine_gpu_mode_requires_directml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = AppConfig(
        app_dir=tmp_path,
        model_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        settings_file=tmp_path / "settings.json",
        last_export_file=tmp_path / "last_export.txt",
    )
    engine = ModNetOnnxEngine(config=config)
    monkeypatch.setattr(engine, "get_available_providers", lambda: ["CPUExecutionProvider"])

    with pytest.raises(RuntimeError, match="DirectML"):
        engine._resolve_providers(HardwareMode.GPU)
