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
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.manual_refine import ManualRefineSession
from core.models import BackgroundColor, ExportFormat, HardwareMode, ProcessingRequest, ProcessingResult
from core.settings import AppSettingsStore
from ui.contour_editor import ContourEditorDialog


class PreviewLabel(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._source_pixmap: QPixmap | None = None
        self.setObjectName(f"{title}_preview")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(280, 360)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QLabel { background: #f7f9fc; border: 1px solid #cfd7e6; border-radius: 12px; color: #43516a; }"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_placeholder(self, text: str) -> None:
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def set_image(self, image: QImage) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self.setText("")
        self._refresh_scaled_pixmap()

    def set_rgb_array(self, image: np.ndarray) -> None:
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
        scaled = self._source_pixmap.scaled(
            max(1, self.width() - 24),
            max(1, self.height() - 24),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
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
        except Exception as exc:  # pragma: no cover
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
        task = WorkerTask(self.pipeline.export, path, result, quality)
        self._active_tasks.add(task)
        task.signals.succeeded.connect(lambda _: self.export_succeeded.emit(path))
        task.signals.failed.connect(lambda exc: self.export_failed.emit(path, self._format_error(exc)))
        task.signals.finished.connect(lambda task=task: self._finish(task))
        self.thread_pool.start(task)
        return True

    def _start_process(self, request: ProcessingRequest) -> None:
        self._set_busy(True)
        self.process_started.emit(request)
        task = WorkerTask(self.pipeline.process, request)
        self._active_tasks.add(task)
        task.signals.succeeded.connect(lambda result: self.process_succeeded.emit(request, result))
        task.signals.failed.connect(lambda exc: self.process_failed.emit(request, self._format_error(exc)))
        task.signals.finished.connect(lambda task=task: self._finish(task))
        self.thread_pool.start(task)

    def _finish(self, task: WorkerTask) -> None:
        self._active_tasks.discard(task)
        if self._pending_request is not None and not self._busy:
            return
        if self._pending_request is not None:
            next_request = self._pending_request
            self._pending_request = None
            self._set_busy(False)
            self.process(next_request)
            return
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        if self._busy == busy:
            return
        self._busy = busy
        self.busy_changed.emit(busy)

    @staticmethod
    def _format_error(exc: Exception) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        if isinstance(exc, FileNotFoundError):
            return f"模型未准备好：{message}"
        return message


class IdPhotoWindow(QMainWindow):
    editor_dialog_class = ContourEditorDialog

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
        self.resize(1240, 820)
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
        self.model_notice_label.setWordWrap(True)
        self.model_notice_label.setStyleSheet(
            "QLabel { background: #fff1d6; color: #7a4a00; border: 1px solid #f1c27d; border-radius: 10px; padding: 10px 12px; }"
        )
        self.model_notice_label.hide()
        root.addWidget(self.model_notice_label)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("Windows 本地证件照换底工具")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: 700; color: #102347; }")
        subtitle = QLabel("自动抠图、标准换底，并支持手动轮廓微调")
        subtitle.setStyleSheet("QLabel { color: #5f6f88; }")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch(1)

        self.import_button = QPushButton("导入图片")
        self.import_button.setMinimumHeight(40)
        self.manual_refine_button = QPushButton("手动微调")
        self.manual_refine_button.setMinimumHeight(40)
        self.manual_refine_button.setEnabled(False)
        self.export_button = QPushButton("导出结果")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        header.addWidget(self.import_button)
        header.addWidget(self.manual_refine_button)
        header.addWidget(self.export_button)
        root.addLayout(header)

        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self._create_preview_group("原图", "请选择 JPG / PNG 证件照"))
        preview_layout.addWidget(self._create_preview_group("结果", "导入图片后自动处理并显示结果"))
        root.addLayout(preview_layout, 1)

        controls_group = QGroupBox("调整")
        controls_layout = QVBoxLayout(controls_group)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("背景色"))
        color_row.addStretch(1)
        self.bg_button_group = QButtonGroup(self)
        self.bg_button_group.setExclusive(True)
        self.color_buttons: dict[BackgroundColor, QToolButton] = {}
        for color in (BackgroundColor.WHITE, BackgroundColor.BLUE, BackgroundColor.RED):
            button = QToolButton()
            button.setText({
                BackgroundColor.WHITE: "白色",
                BackgroundColor.BLUE: "蓝色",
                BackgroundColor.RED: "红色",
            }[color])
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setMinimumWidth(86)
            button.setStyleSheet(
                "QToolButton { border: 1px solid #cfd7e6; border-radius: 18px; padding: 8px 14px; background: white; }"
                "QToolButton:checked { background: #102347; color: white; border-color: #102347; }"
            )
            if color is BackgroundColor.BLUE:
                button.setChecked(True)
            self.color_buttons[color] = button
            self.bg_button_group.addButton(button)
            color_row.addWidget(button)
        controls_layout.addLayout(color_row)

        advanced_row = QHBoxLayout()
        advanced_row.addWidget(QLabel("高级设置"))
        advanced_row.addStretch(1)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("展开")
        self.advanced_toggle.setCheckable(True)
        advanced_row.addWidget(self.advanced_toggle)
        controls_layout.addLayout(advanced_row)

        self.advanced_panel = QWidget()
        advanced_layout = QGridLayout(self.advanced_panel)
        advanced_layout.addWidget(QLabel("边缘柔化"), 0, 0)
        self.edge_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_slider.setRange(0, 100)
        self.edge_slider.setValue(50)
        self.edge_value_label = QLabel("50")
        advanced_layout.addWidget(self.edge_slider, 0, 1)
        advanced_layout.addWidget(self.edge_value_label, 0, 2)

        advanced_layout.addWidget(QLabel("导出格式"), 1, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("PNG", ExportFormat.PNG)
        self.output_format_combo.addItem("JPG", ExportFormat.JPG)
        advanced_layout.addWidget(self.output_format_combo, 1, 1)

        advanced_layout.addWidget(QLabel("硬件模式"), 1, 2)
        self.hardware_mode_combo = QComboBox()
        self.hardware_mode_combo.addItem("自动", HardwareMode.AUTO)
        self.hardware_mode_combo.addItem("仅 CPU", HardwareMode.CPU)
        self.hardware_mode_combo.addItem("DirectML GPU", HardwareMode.GPU)
        advanced_layout.addWidget(self.hardware_mode_combo, 1, 3)
        self.advanced_panel.hide()
        controls_layout.addWidget(self.advanced_panel)

        root.addWidget(controls_group)

        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)
        self.setCentralWidget(central)

    def _create_preview_group(self, title: str, placeholder: str) -> QWidget:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        preview = PreviewLabel(title)
        preview.set_placeholder(placeholder)
        layout.addWidget(preview)
        if title == "原图":
            self.original_preview = preview
        else:
            self.result_preview = preview
        return box

    def _connect_signals(self) -> None:
        self.import_button.clicked.connect(self.choose_image)
        self.manual_refine_button.clicked.connect(self.open_manual_refine)
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
        self.advanced_toggle.setText("收起" if expanded else "展开")
        self.advanced_panel.setVisible(expanded)

    def choose_image(self) -> None:
        start_dir = self._suggest_open_dir()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择证件照", str(start_dir), "Images (*.jpg *.jpeg *.png)")
        if not file_path:
            return
        image_path = Path(file_path)
        self.current_input_path = image_path
        self.current_result = None
        self.manual_refine_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self._load_original_preview(image_path)
        self._update_result_placeholder("正在处理图片，请稍候...")
        self.schedule_process()

    def schedule_process(self) -> None:
        if self.current_input_path is None:
            return
        request = self._build_request()
        if self.current_result is not None and self.current_result.manual_mask is not None:
            updated = self.pipeline.apply_manual_mask(request, self.current_result, self.current_result.manual_mask)
            self._on_process_succeeded(request, updated)
            return
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

    def open_manual_refine(self) -> None:
        if self.current_result is None:
            return
        try:
            session = ManualRefineSession.from_mask(self.current_result.active_mask)
        except ValueError as exc:
            QMessageBox.warning(self, "无法进入手动微调", str(exc))
            return
        image = self.current_result.source_image
        if image is None and self.current_input_path is not None:
            reader = QImageReader(str(self.current_input_path))
            reader.setAutoTransform(True)
            qimage = reader.read()
            if not qimage.isNull():
                ptr = qimage.convertToFormat(QImage.Format.Format_RGB888)
                buffer = ptr.bits().tobytes(ptr.width() * ptr.height() * 3)
                image = np.frombuffer(buffer, dtype=np.uint8).reshape(ptr.height(), ptr.width(), 3).copy()
        dialog = self.editor_dialog_class(session=session, image=image, parent=self)
        if dialog.exec():
            self._apply_manual_refine(dialog.build_manual_mask())

    def _apply_manual_refine(self, manual_mask: np.ndarray) -> None:
        if self.current_input_path is None or self.current_result is None:
            return
        request = self._build_request()
        updated = self.pipeline.apply_manual_mask(request, self.current_result, manual_mask)
        self._on_process_succeeded(request, updated)
        self.statusBar().showMessage("已应用手动微调", 6000)

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
        if message:
            self.model_notice_label.setText(message)
            self.model_notice_label.show()
        else:
            self.model_notice_label.hide()

    def _detect_model_notice(self) -> str | None:
        model_path = getattr(getattr(self.pipeline, "engine", None), "model_path", None)
        if isinstance(model_path, Path) and not model_path.exists():
            return f"模型未准备好：未找到 {model_path}。请先运行 scripts/download_model.py 下载 MODNet ONNX 模型。"
        return None

    def _set_busy_state(self, busy: bool) -> None:
        self._busy = busy
        self.import_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.current_result is not None)
        self.manual_refine_button.setEnabled(not busy and self.current_result is not None)
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
        self.manual_refine_button.setEnabled(True)
        self._refresh_model_notice()
        detail = "（已手动微调）" if result.manual_mask is not None else ""
        self.statusBar().showMessage(
            f"处理完成{detail} | Provider: {result.provider} | 耗时: {result.elapsed_ms:.1f} ms",
            10000,
        )

    def _on_process_failed(self, request: ProcessingRequest, message: str) -> None:
        del request
        self.current_result = None
        self.export_button.setEnabled(False)
        self.manual_refine_button.setEnabled(False)
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
        return stored if stored is not None else Path.home()

    def _suggest_export_dir(self) -> Path:
        stored = self.settings_store.load().last_export_dir
        if stored is not None:
            return stored
        if self.current_input_path is not None:
            return self.current_input_path.parent
        return Path.home()
