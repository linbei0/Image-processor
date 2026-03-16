from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.models import BackgroundColor, ExportFormat, HardwareMode, ProcessingRequest, ProcessingResult
from core.settings import AppSettingsStore


class PreviewLabel(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._source_pixmap: QPixmap | None = None
        self.setObjectName(f"{title}_preview")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(280, 360)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QLabel { background: #f7f9fc; border: 1px solid #cfd7e6; border-radius: 12px; color: #43516a; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_placeholder(self, text: str) -> None:
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def set_image(self, image: QImage) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self.setText("")
        self._refresh_scaled_pixmap()

    def set_rgb_array(self, image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("预览图必须是 RGB 三通道数组")
        contiguous = np.ascontiguousarray(image)
        height, width, _ = contiguous.shape
        qimage = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_RGB888).copy()
        self.set_image(qimage)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        available_width = max(1, self.width() - 24)
        available_height = max(1, self.height() - 24)
        scaled = self._source_pixmap.scaled(
            available_width,
            available_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class WorkerSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()


class WorkerTask(QRunnable):
    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # pragma: no cover - Qt thread boundary
            self.signals.failed.emit(exc)
        else:
            self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit()


class PipelineCoordinator(QObject):
    busy_changed = Signal(bool)
    process_started = Signal(object)
    process_succeeded = Signal(object, object)
    process_failed = Signal(object, str)
    export_started = Signal(object)
    export_succeeded = Signal(object)
    export_failed = Signal(object, str)

    def __init__(self, pipeline, thread_pool: QThreadPool | None = None) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.thread_pool = thread_pool or QThreadPool.globalInstance()
        self._busy = False
        self._pending_request: ProcessingRequest | None = None
        self._active_tasks: set[WorkerTask] = set()

    def process(self, request: ProcessingRequest) -> None:
        if self._busy:
            self._pending_request = request
            return
        self._start_process(request)

    def export(self, path: Path, result: ProcessingResult, quality: int = 95) -> bool:
        if self._busy:
            return False
        self._set_busy(True)
        self.export_started.emit(path)
        task = WorkerTask(self._run_export, path, result, quality)
        self._active_tasks.add(task)
        task.signals.succeeded.connect(self.export_succeeded.emit)
        task.signals.failed.connect(lambda exc: self.export_failed.emit(path, self._format_error(exc)))
        task.signals.finished.connect(lambda task=task: self._finish_export(task))
        self.thread_pool.start(task)
        return True

    def _start_process(self, request: ProcessingRequest) -> None:
        self._set_busy(True)
        self.process_started.emit(request)
        task = WorkerTask(self.pipeline.process, request)
        self._active_tasks.add(task)
        task.signals.succeeded.connect(lambda result: self.process_succeeded.emit(request, result))
        task.signals.failed.connect(lambda exc: self.process_failed.emit(request, self._format_error(exc)))
        task.signals.finished.connect(lambda task=task: self._finish_process(task))
        self.thread_pool.start(task)

    def _finish_process(self, task: WorkerTask) -> None:
        self._active_tasks.discard(task)
        if self._pending_request is not None:
            next_request = self._pending_request
            self._pending_request = None
            self._start_process(next_request)
            return
        self._set_busy(False)

    def _finish_export(self, task: WorkerTask) -> None:
        self._active_tasks.discard(task)
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        if self._busy == busy:
            return
        self._busy = busy
        self.busy_changed.emit(busy)

    def _run_export(self, path: Path, result: ProcessingResult, quality: int) -> Path:
        self.pipeline.export(path, result, quality=quality)
        return path

    @staticmethod
    def _format_error(exc: Exception) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        if isinstance(exc, FileNotFoundError):
            return f"模型未准备好：{message}"
        return message


class IdPhotoWindow(QMainWindow):
    def __init__(
        self,
        pipeline,
        settings_store: AppSettingsStore | None = None,
        thread_pool: QThreadPool | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.settings_store = settings_store or AppSettingsStore(Path.cwd() / "settings.json")
        self.coordinator = PipelineCoordinator(pipeline=pipeline, thread_pool=thread_pool)
        self.current_input_path: Path | None = None
        self.current_result: ProcessingResult | None = None
        self._busy = False

        self.setWindowTitle("证件照换底工具")
        self.resize(1200, 760)
        self._build_ui()
        self._connect_signals()
        self._refresh_model_notice()
        self._update_result_placeholder()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        self.model_notice_label = QLabel()
        self.model_notice_label.setObjectName("model_notice_label")
        self.model_notice_label.setWordWrap(True)
        self.model_notice_label.setStyleSheet(
            "QLabel { background: #fff1d6; color: #7a4a00; border: 1px solid #f1c27d; border-radius: 10px; padding: 10px 12px; }"
        )
        self.model_notice_label.hide()
        root.addWidget(self.model_notice_label)

        header_layout = QHBoxLayout()
        title = QLabel("Windows 本地证件照换底")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: 700; color: #102347; }")
        subtitle = QLabel("导入照片后自动调用 ProcessingPipeline 进行抠图与换底。")
        subtitle.setStyleSheet("QLabel { color: #5f6f88; }")
        title_block = QVBoxLayout()
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_layout.addLayout(title_block)
        header_layout.addStretch(1)

        self.import_button = QPushButton("导入图片")
        self.import_button.setObjectName("import_button")
        self.import_button.setMinimumHeight(40)
        self.export_button = QPushButton("导出结果")
        self.export_button.setObjectName("export_button")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        header_layout.addWidget(self.import_button)
        header_layout.addWidget(self.export_button)
        root.addLayout(header_layout)

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(16)
        preview_layout.addWidget(self._create_preview_group("原图", "请选择 JPG / PNG 证件照"))
        preview_layout.addWidget(self._create_preview_group("结果", "导入图片后自动处理并显示结果"))
        root.addLayout(preview_layout, 1)

        controls_group = QGroupBox("调整")
        controls_group.setStyleSheet("QGroupBox { font-weight: 700; color: #17305c; }")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(12)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("背景色"))
        color_row.addStretch(1)
        self.bg_button_group = QButtonGroup(self)
        self.bg_button_group.setExclusive(True)
        self.color_buttons: dict[BackgroundColor, QToolButton] = {}
        for color in (BackgroundColor.WHITE, BackgroundColor.BLUE, BackgroundColor.RED):
            button = self._create_color_button(color)
            self.bg_button_group.addButton(button)
            self.color_buttons[color] = button
            color_row.addWidget(button)
        self.color_buttons[BackgroundColor.BLUE].setChecked(True)
        controls_layout.addLayout(color_row)

        slider_layout = QGridLayout()
        slider_layout.addWidget(QLabel("边缘柔化"), 0, 0)
        self.edge_slider = QSlider(Qt.Horizontal)
        self.edge_slider.setObjectName("edge_slider")
        self.edge_slider.setRange(0, 100)
        self.edge_slider.setValue(50)
        self.edge_value_label = QLabel("50")
        slider_layout.addWidget(self.edge_slider, 0, 1)
        slider_layout.addWidget(self.edge_value_label, 0, 2)
        controls_layout.addLayout(slider_layout)

        self.advanced_toggle = QPushButton("高级设置")
        self.advanced_toggle.setObjectName("advanced_toggle")
        self.advanced_toggle.setCheckable(True)
        controls_layout.addWidget(self.advanced_toggle)

        self.advanced_panel = QFrame()
        self.advanced_panel.setObjectName("advanced_panel")
        advanced_layout = QGridLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(12, 12, 12, 12)
        advanced_layout.setHorizontalSpacing(12)
        advanced_layout.setVerticalSpacing(10)
        advanced_layout.addWidget(QLabel("导出格式"), 0, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("PNG（保留透明边缘）", ExportFormat.PNG)
        self.output_format_combo.addItem("JPG（兼容常见证件照提交）", ExportFormat.JPG)
        advanced_layout.addWidget(self.output_format_combo, 0, 1)
        advanced_layout.addWidget(QLabel("推理模式"), 1, 0)
        self.hardware_mode_combo = QComboBox()
        self.hardware_mode_combo.addItem("自动", HardwareMode.AUTO)
        self.hardware_mode_combo.addItem("仅 CPU", HardwareMode.CPU)
        self.hardware_mode_combo.addItem("DirectML GPU", HardwareMode.GPU)
        advanced_layout.addWidget(self.hardware_mode_combo, 1, 1)
        self.advanced_panel.hide()
        controls_layout.addWidget(self.advanced_panel)

        root.addWidget(controls_group)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("请选择一张证件照开始处理。")

    def _create_preview_group(self, title: str, placeholder: str) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        preview = PreviewLabel(title.lower())
        preview.set_placeholder(placeholder)
        if title == "原图":
            self.original_preview = preview
        else:
            self.result_preview = preview
        layout.addWidget(preview)
        return group

    def _create_color_button(self, color: BackgroundColor) -> QToolButton:
        label = {
            BackgroundColor.WHITE: "白",
            BackgroundColor.BLUE: "蓝",
            BackgroundColor.RED: "红",
        }[color]
        r, g, b = color.rgb
        button = QToolButton(self)
        button.setText(label)
        button.setCheckable(True)
        button.setFixedSize(52, 52)
        button.setStyleSheet(
            "QToolButton {"
            f"background: rgb({r}, {g}, {b});"
            "border: 2px solid transparent; border-radius: 14px; font-size: 16px; font-weight: 700;"
            f"color: {'#17305c' if color is BackgroundColor.WHITE else '#ffffff'};"
            "}"
            "QToolButton:checked { border-color: #102347; }"
        )
        return button

    def _connect_signals(self) -> None:
        self.import_button.clicked.connect(self.choose_image)
        self.export_button.clicked.connect(self.export_result)
        self.bg_button_group.buttonClicked.connect(lambda _: self.schedule_process())
        self.edge_slider.valueChanged.connect(self._on_edge_value_changed)
        self.advanced_toggle.toggled.connect(self._toggle_advanced_panel)
        self.output_format_combo.currentIndexChanged.connect(lambda _: self.schedule_process())
        self.hardware_mode_combo.currentIndexChanged.connect(lambda _: self.schedule_process())

        self.coordinator.busy_changed.connect(self._set_busy_state)
        self.coordinator.process_started.connect(self._on_process_started)
        self.coordinator.process_succeeded.connect(self._on_process_succeeded)
        self.coordinator.process_failed.connect(self._on_process_failed)
        self.coordinator.export_started.connect(self._on_export_started)
        self.coordinator.export_succeeded.connect(self._on_export_succeeded)
        self.coordinator.export_failed.connect(self._on_export_failed)

    def _on_edge_value_changed(self, value: int) -> None:
        self.edge_value_label.setText(str(value))
        self.schedule_process()

    def _toggle_advanced_panel(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)

    def choose_image(self) -> None:
        start_dir = self._suggest_open_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择证件照",
            str(start_dir),
            "图片文件 (*.png *.jpg *.jpeg)",
        )
        if not file_path:
            return

        image_path = Path(file_path)
        try:
            self._load_original_preview(image_path)
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 8000)
            self.result_preview.set_placeholder(str(exc))
            return

        self.current_input_path = image_path
        self.current_result = None
        self.export_button.setEnabled(False)
        self._update_result_placeholder("正在处理图片，请稍候...")
        self.schedule_process()

    def schedule_process(self) -> None:
        if self.current_input_path is None:
            return
        request = self._build_request()
        self.coordinator.process(request)

    def export_result(self) -> None:
        if self.current_result is None or self.current_input_path is None:
            return
        export_format = ExportFormat(self.output_format_combo.currentData())
        start_dir = self._suggest_export_dir()
        suggested = start_dir / f"{self.current_input_path.stem}_{self._selected_bg_color().value}.{export_format.value}"
        filters = "PNG (*.png)" if export_format is ExportFormat.PNG else "JPG (*.jpg)"
        file_path, _ = QFileDialog.getSaveFileName(self, "导出结果", str(suggested), filters)
        if not file_path:
            return
        self.coordinator.export(Path(file_path), self.current_result, quality=95)

    def _build_request(self) -> ProcessingRequest:
        if self.current_input_path is None:
            raise RuntimeError("导入图片后才能处理")
        return ProcessingRequest(
            input_path=self.current_input_path,
            bg_color=self._selected_bg_color(),
            output_format=ExportFormat(self.output_format_combo.currentData()),
            edge_refine_level=self.edge_slider.value(),
            hardware_mode=HardwareMode(self.hardware_mode_combo.currentData()),
        )

    def _selected_bg_color(self) -> BackgroundColor:
        for color, button in self.color_buttons.items():
            if button.isChecked():
                return color
        return BackgroundColor.BLUE

    def _load_original_preview(self, path: Path) -> None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(f"无法读取图片文件：{path}")
        self.original_preview.set_image(image)

    def _refresh_model_notice(self, explicit_message: str | None = None) -> None:
        message = explicit_message or self._detect_model_notice()
        if not message:
            self.model_notice_label.hide()
            return
        self.model_notice_label.setText(message)
        self.model_notice_label.show()

    def _detect_model_notice(self) -> str | None:
        engine = getattr(self.pipeline, "engine", None)
        model_path = getattr(engine, "model_path", None)
        if isinstance(model_path, Path) and not model_path.exists():
            return f"模型未准备好：未找到 {model_path}。请先运行 scripts/download_model.py 下载 MODNet ONNX 模型。"
        return None

    def _set_busy_state(self, busy: bool) -> None:
        self._busy = busy
        self.import_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.current_result is not None)
        for button in self.color_buttons.values():
            button.setEnabled(not busy)
        self.edge_slider.setEnabled(not busy)
        self.advanced_toggle.setEnabled(not busy)
        self.output_format_combo.setEnabled(not busy)
        self.hardware_mode_combo.setEnabled(not busy)

    def _on_process_started(self, request: ProcessingRequest) -> None:
        self.statusBar().showMessage(
            f"正在处理：背景 {request.bg_color.value} / 边缘柔化 {request.edge_refine_level}",
            0,
        )

    def _on_process_succeeded(self, request: ProcessingRequest, result: ProcessingResult) -> None:
        self.current_result = result
        self.result_preview.set_rgb_array(result.preview_image)
        self.export_button.setEnabled(True)
        self._refresh_model_notice()
        self.statusBar().showMessage(
            f"处理完成 | Provider: {result.provider} | 耗时: {result.elapsed_ms:.1f} ms",
            10000,
        )

    def _on_process_failed(self, request: ProcessingRequest, message: str) -> None:
        self.current_result = None
        self.export_button.setEnabled(False)
        self._update_result_placeholder(message)
        self._refresh_model_notice(message if message.startswith("模型未准备好") else None)
        self.statusBar().showMessage(message, 12000)

    def _on_export_started(self, path: Path) -> None:
        self.statusBar().showMessage(f"正在导出到 {path} ...", 0)

    def _on_export_succeeded(self, path: Path) -> None:
        self.settings_store.save_last_export_dir(path.parent)
        self.statusBar().showMessage(f"导出完成：{path}", 10000)

    def _on_export_failed(self, path: Path, message: str) -> None:
        self.statusBar().showMessage(f"导出失败：{message}", 12000)

    def _update_result_placeholder(self, text: str | None = None) -> None:
        if text:
            self.result_preview.set_placeholder(text)
            return
        notice = self._detect_model_notice()
        if notice:
            self.result_preview.set_placeholder(notice)
            return
        self.result_preview.set_placeholder("导入图片后自动处理并显示结果")

    def _suggest_open_dir(self) -> Path:
        if self.current_input_path is not None:
            return self.current_input_path.parent
        stored = self.settings_store.load().last_export_dir
        if stored is not None:
            return stored
        return Path.home()

    def _suggest_export_dir(self) -> Path:
        stored = self.settings_store.load().last_export_dir
        if stored is not None:
            return stored
        if self.current_input_path is not None:
            return self.current_input_path.parent
        return Path.home()
