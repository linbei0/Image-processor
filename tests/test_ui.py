from pathlib import Path

import os
import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from core.models import BackgroundColor, HardwareMode, ProcessingResult
from core.settings import AppSettingsStore
from ui.window import IdPhotoWindow


class DummyEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path


class FakePipeline:
    def __init__(self, model_path: Path, process_error: Exception | None = None) -> None:
        self.engine = DummyEngine(model_path)
        self.process_error = process_error
        self.process_calls = []
        self.export_calls = []

    def process(self, request):
        self.process_calls.append(request)
        if self.process_error is not None:
            raise self.process_error

        fill = np.full((24, 24, 3), request.bg_color.rgb, dtype=np.uint8)
        alpha = np.ones((24, 24), dtype=np.float32)
        provider = "DmlExecutionProvider" if request.hardware_mode is HardwareMode.GPU else "CPUExecutionProvider"
        return ProcessingResult(
            preview_image=fill.copy(),
            final_image=fill,
            alpha_mask=alpha,
            elapsed_ms=12.5,
            provider=provider,
        )

    def export(self, path: Path, result: ProcessingResult, quality: int = 95) -> None:
        self.export_calls.append((path, result, quality))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"exported")


def create_image(path: Path) -> None:
    Image.new("RGB", (40, 50), (120, 130, 140)).save(path)


@pytest.fixture()
def settings_store(tmp_path: Path) -> AppSettingsStore:
    return AppSettingsStore(tmp_path / "settings.json")


def test_window_shows_explicit_model_missing_notice(tmp_path: Path, qtbot, settings_store: AppSettingsStore) -> None:
    pipeline = FakePipeline(
        tmp_path / "missing-model.onnx",
        process_error=FileNotFoundError("未找到模型文件，请先运行 scripts/download_model.py"),
    )
    window = IdPhotoWindow(pipeline=pipeline, settings_store=settings_store)
    qtbot.addWidget(window)
    window.show()

    assert window.model_notice_label.isVisible()
    assert "模型未准备好" in window.model_notice_label.text()
    assert not window.export_button.isEnabled()


def test_import_runs_processing_and_updates_previews(
    tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch, settings_store: AppSettingsStore
) -> None:
    model_path = tmp_path / "modnet.onnx"
    model_path.write_bytes(b"ready")
    image_path = tmp_path / "portrait.png"
    create_image(image_path)
    pipeline = FakePipeline(model_path)
    window = IdPhotoWindow(pipeline=pipeline, settings_store=settings_store)
    qtbot.addWidget(window)
    window.show()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(image_path), "PNG"))

    qtbot.mouseClick(window.import_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: len(pipeline.process_calls) == 1, timeout=3000)
    qtbot.waitUntil(lambda: window.export_button.isEnabled(), timeout=3000)

    assert window.current_input_path == image_path
    assert window.original_preview.pixmap() is not None
    assert window.result_preview.pixmap() is not None
    assert "处理完成" in window.statusBar().currentMessage()
    assert pipeline.process_calls[0].bg_color is BackgroundColor.BLUE


def test_adjusting_controls_reprocesses_with_latest_options(
    tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch, settings_store: AppSettingsStore
) -> None:
    model_path = tmp_path / "modnet.onnx"
    model_path.write_bytes(b"ready")
    image_path = tmp_path / "portrait.png"
    create_image(image_path)
    pipeline = FakePipeline(model_path)
    window = IdPhotoWindow(pipeline=pipeline, settings_store=settings_store)
    qtbot.addWidget(window)
    window.show()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(image_path), "PNG"))
    qtbot.mouseClick(window.import_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: len(pipeline.process_calls) == 1, timeout=3000)

    qtbot.mouseClick(window.advanced_toggle, Qt.LeftButton)
    assert window.advanced_panel.isVisible()

    qtbot.mouseClick(window.color_buttons[BackgroundColor.RED], Qt.LeftButton)
    window.edge_slider.setValue(72)
    window.hardware_mode_combo.setCurrentIndex(window.hardware_mode_combo.findData(HardwareMode.GPU))

    qtbot.waitUntil(lambda: len(pipeline.process_calls) >= 3, timeout=3000)
    latest = pipeline.process_calls[-1]

    assert latest.bg_color is BackgroundColor.RED
    assert latest.edge_refine_level == 72
    assert latest.hardware_mode is HardwareMode.GPU
    assert "DmlExecutionProvider" in window.statusBar().currentMessage()


def test_export_uses_pipeline_and_persists_last_export_dir(
    tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch, settings_store: AppSettingsStore
) -> None:
    model_path = tmp_path / "modnet.onnx"
    model_path.write_bytes(b"ready")
    image_path = tmp_path / "portrait.png"
    create_image(image_path)
    export_path = tmp_path / "exports" / "portrait_result.png"
    pipeline = FakePipeline(model_path)
    window = IdPhotoWindow(pipeline=pipeline, settings_store=settings_store)
    qtbot.addWidget(window)
    window.show()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(image_path), "PNG"))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(export_path), "PNG"))

    qtbot.mouseClick(window.import_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: window.export_button.isEnabled(), timeout=3000)

    qtbot.mouseClick(window.export_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: len(pipeline.export_calls) == 1, timeout=3000)

    saved_path, _, quality = pipeline.export_calls[0]
    assert saved_path == export_path
    assert export_path.exists()
    assert quality == 95
    assert settings_store.load().last_export_dir == export_path.parent
