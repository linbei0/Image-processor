from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.manual_refine import ManualRefineSession


def _rgb_to_qimage(image: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(image)
    height, width = contiguous.shape[:2]
    return QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_RGB888).copy()


def _mask_overlay(mask: np.ndarray) -> QPixmap:
    alpha = np.clip(mask * 110.0, 0, 110).astype(np.uint8)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 1] = 180
    rgba[..., 2] = 255
    rgba[..., 3] = alpha
    qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


class BrushCursorItem(QGraphicsEllipseItem):
    def __init__(self) -> None:
        super().__init__()
        self.setRect(-10, -10, 20, 20)
        self.setPen(QPen(QColor("#f8fafc"), 1.5, Qt.PenStyle.DashLine))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setZValue(30)
        self.hide()

    def update_cursor(self, point: QPointF, radius: int) -> None:
        diameter = radius * 2
        self.setRect(-radius, -radius, diameter, diameter)
        self.setPos(point)
        self.show()


class ControlPointItem(QGraphicsEllipseItem):
    def __init__(self, index: int, editor: "ContourEditorView", point: np.ndarray) -> None:
        super().__init__(-5, -5, 10, 10)
        self.index = index
        self.editor = editor
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#0f172a"), 1.5))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPos(float(point[0]), float(point[1]))
        self.setZValue(20)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.editor.edit_mode != "contour":
            event.ignore()
            return
        self.editor.begin_handle_move(self.index)
        super().mousePressEvent(event)
        self.editor.selected_index = self.index

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        self.editor.finish_handle_move(self.index, self.pos())

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange:
            rect = self.editor.sceneRect()
            value = QPointF(
                min(max(value.x(), rect.left()), rect.right()),
                min(max(value.y(), rect.top()), rect.bottom()),
            )
            return value
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            self.editor.preview_handle_move(self.index, self.pos())
        return super().itemChange(change, value)


class ContourEditorView(QGraphicsView):
    mode_changed = Signal(str)

    def __init__(self, session: ManualRefineSession, image: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.image = image
        self.selected_index: int | None = None
        self._moving_handle_index: int | None = None
        self._panning = False
        self._painting = False
        self._did_initial_fit = False
        self._user_changed_view = False
        self._last_refresh_at = -1.0
        self._refresh_interval_seconds = 1 / 60
        self.edit_mode = "contour"
        self.current_brush_mode = "add"
        self.brush_radius = 24
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#111827"))
        self.setSceneRect(0, 0, image.shape[1], image.shape[0])

        self._image_item = QGraphicsPixmapItem(QPixmap.fromImage(_rgb_to_qimage(image)))
        self._overlay_pixmap = _mask_overlay(session.active_mask)
        self._overlay_item = QGraphicsPixmapItem(self._overlay_pixmap)
        self._overlay_item.setOpacity(0.55)
        self._path_item = QGraphicsPathItem()
        self._path_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        self._path_item.setZValue(10)
        self._cursor_item = BrushCursorItem()

        self._scene.addItem(self._image_item)
        self._scene.addItem(self._overlay_item)
        self._scene.addItem(self._path_item)
        self._scene.addItem(self._cursor_item)

        self._handle_items: list[ControlPointItem] = []
        self._rebuild_items()
        self._fit_scene_to_view()

    def set_edit_mode(self, mode: str) -> None:
        self.edit_mode = mode
        show_handles = mode == "contour"
        for item in self._handle_items:
            item.setVisible(show_handles)
        self._path_item.setVisible(show_handles)
        if mode != "brush":
            self._cursor_item.hide()
        self.mode_changed.emit(mode)

    def set_brush_mode(self, mode: str) -> None:
        self.current_brush_mode = mode

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius = max(1, int(radius))

    def begin_handle_move(self, index: int) -> None:
        self._moving_handle_index = index
        self.session.push_history()

    def preview_handle_move(self, index: int, pos: QPointF) -> None:
        if self._moving_handle_index != index:
            return
        self.session.current_points[index] = np.array([pos.x(), pos.y()], dtype=np.float32)
        self._refresh_visuals()

    def finish_handle_move(self, index: int, pos: QPointF) -> None:
        self._moving_handle_index = None
        self.session.current_points[index] = np.array([pos.x(), pos.y()], dtype=np.float32)
        self._refresh_visuals()

    def insert_point_at_scene_pos(self, scene_pos: QPointF) -> None:
        points = self.session.current_points
        best_index = 0
        best_distance = float("inf")
        p = np.array([scene_pos.x(), scene_pos.y()], dtype=np.float32)
        for idx in range(len(points)):
            a = points[idx]
            b = points[(idx + 1) % len(points)]
            ab = b - a
            denom = float(np.dot(ab, ab)) or 1.0
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            projection = a + ab * t
            distance = float(np.linalg.norm(projection - p))
            if distance < best_distance:
                best_distance = distance
                best_index = idx
        self.session.insert_point(best_index, (scene_pos.x(), scene_pos.y()))
        self._rebuild_items()

    def delete_selected_point(self) -> None:
        if self.selected_index is None:
            return
        self.session.delete_point(self.selected_index)
        self.selected_index = None
        self._rebuild_items()

    def undo(self) -> None:
        if self.session.undo():
            self._rebuild_items()

    def redo(self) -> None:
        if self.session.redo():
            self._rebuild_items()

    def reset(self) -> None:
        self.session.reset()
        self._rebuild_items()

    def _maybe_refresh_visuals(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or (now - self._last_refresh_at) >= self._refresh_interval_seconds:
            self._refresh_visuals()
            self._last_refresh_at = now

    def wheelEvent(self, event) -> None:  # noqa: N802
        self._user_changed_view = True
        if self.edit_mode == "brush" and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_brush_radius(self.brush_radius + (2 if event.angleDelta().y() > 0 else -2))
            self._cursor_item.update_cursor(self.mapToScene(event.position().toPoint()), self.brush_radius)
            event.accept()
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Delete and self.edit_mode == "contour":
            self.delete_selected_point()
            return
        if event.key() == Qt.Key.Key_BracketLeft:
            self.set_brush_radius(self.brush_radius - 2)
            return
        if event.key() == Qt.Key.Key_BracketRight:
            self.set_brush_radius(self.brush_radius + 2)
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        scene_pos = self.mapToScene(event.position().toPoint())
        item = self.itemAt(event.position().toPoint())
        if self.edit_mode == "brush" and event.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self.session.begin_brush_stroke()
            self._apply_brush_at_scene_pos(scene_pos, force_full=False)
            return
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(item, ControlPointItem):
            self._user_changed_view = True
            self._panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        scene_pos = self.mapToScene(event.position().toPoint())
        if self.edit_mode == "brush":
            self._cursor_item.update_cursor(scene_pos, self.brush_radius)
            if self._painting:
                self._apply_brush_at_scene_pos(scene_pos, force_full=False)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._painting and event.button() == Qt.MouseButton.LeftButton:
            self._painting = False
            self.session.end_brush_stroke()
            self._maybe_refresh_visuals(force=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_item.hide()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self.edit_mode == "contour":
            self.insert_point_at_scene_pos(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._did_initial_fit:
            self._fit_scene_to_view()
            self._did_initial_fit = True

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._did_initial_fit and not self._user_changed_view:
            self._fit_scene_to_view()

    def _rebuild_items(self) -> None:
        for item in self._handle_items:
            self._scene.removeItem(item)
        self._handle_items.clear()
        for index, point in enumerate(self.session.current_points):
            handle = ControlPointItem(index=index, editor=self, point=point)
            self._scene.addItem(handle)
            self._handle_items.append(handle)
        self._refresh_visuals()

    def _fit_scene_to_view(self) -> None:
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _refresh_visuals(self) -> None:
        points = self.session.current_points
        path = QPainterPath(QPointF(float(points[0][0]), float(points[0][1])))
        for point in points[1:]:
            path.lineTo(float(point[0]), float(point[1]))
        path.closeSubpath()
        self._path_item.setPath(path)
        try:
            self._overlay_pixmap = _mask_overlay(self.session.active_mask)
            self._overlay_item.setPixmap(self._overlay_pixmap)
            self._path_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        except ValueError:
            self._path_item.setPen(QPen(QColor("#ef4444"), 2.0))

    def _apply_brush_at_scene_pos(self, scene_pos: QPointF, force_full: bool = False) -> None:
        dirty_rect = self.session.apply_brush(
            (scene_pos.x(), scene_pos.y()),
            self.brush_radius,
            self.current_brush_mode,
        )
        if force_full:
            self._maybe_refresh_visuals(force=True)
            return
        self._refresh_overlay_region(dirty_rect)

    def _refresh_overlay_region(self, rect: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return
        patch_mask = self.session.active_mask_region(rect)
        patch_pixmap = _mask_overlay(patch_mask)
        painter = QPainter(self._overlay_pixmap)
        painter.drawPixmap(x0, y0, patch_pixmap)
        painter.end()
        self._overlay_item.setPixmap(self._overlay_pixmap)


class ContourEditorDialog(QDialog):
    def __init__(self, session: ManualRefineSession, image: np.ndarray | None, parent=None) -> None:
        super().__init__(parent)
        if image is None:
            raise ValueError("手动微调需要原始图像")
        self.session = session
        self.setWindowTitle("手动微调")
        self.resize(1160, 860)

        layout = QVBoxLayout(self)
        toolbar = QToolBar()
        self.contour_mode_button = QToolButton()
        self.contour_mode_button.setText("轮廓模式")
        self.contour_mode_button.setCheckable(True)
        self.contour_mode_button.setChecked(True)
        self.brush_mode_button = QToolButton()
        self.brush_mode_button.setText("笔刷模式")
        self.brush_mode_button.setCheckable(True)
        toolbar.addWidget(self.contour_mode_button)
        toolbar.addWidget(self.brush_mode_button)
        toolbar.addSeparator()

        self.add_brush_button = QToolButton()
        self.add_brush_button.setText("补前景")
        self.add_brush_button.setCheckable(True)
        self.add_brush_button.setChecked(True)
        self.erase_brush_button = QToolButton()
        self.erase_brush_button.setText("擦背景")
        self.erase_brush_button.setCheckable(True)
        toolbar.addWidget(self.add_brush_button)
        toolbar.addWidget(self.erase_brush_button)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("笔刷大小"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(4, 80)
        self.brush_slider.setValue(24)
        self.brush_slider.setFixedWidth(160)
        toolbar.addWidget(self.brush_slider)
        self.brush_label = QLabel("24 px")
        toolbar.addWidget(self.brush_label)
        toolbar.addSeparator()

        undo_action = toolbar.addAction("撤销")
        redo_action = toolbar.addAction("重做")
        reset_action = toolbar.addAction("恢复自动结果")
        layout.addWidget(toolbar)

        tips = QLabel("轮廓模式：拖点修大轮廓。笔刷模式：补前景/擦背景。滚轮缩放，长按空白区域拖动画布，Ctrl+滚轮或 [ ] 调笔刷大小。")
        tips.setWordWrap(True)
        layout.addWidget(tips)

        self.editor = ContourEditorView(session=session, image=image, parent=self)
        layout.addWidget(self.editor, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QPushButton("取消")
        apply_button = QPushButton("应用微调")
        buttons.addWidget(cancel_button)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)

        self.contour_mode_button.clicked.connect(lambda: self._set_mode("contour"))
        self.brush_mode_button.clicked.connect(lambda: self._set_mode("brush"))
        self.add_brush_button.clicked.connect(lambda: self._set_brush_mode("add"))
        self.erase_brush_button.clicked.connect(lambda: self._set_brush_mode("erase"))
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        undo_action.triggered.connect(self.editor.undo)
        redo_action.triggered.connect(self.editor.redo)
        reset_action.triggered.connect(self.editor.reset)
        cancel_button.clicked.connect(self.reject)
        apply_button.clicked.connect(self._accept_if_valid)

        self._set_mode("contour")
        self._set_brush_mode("add")

    def _set_mode(self, mode: str) -> None:
        self.editor.set_edit_mode(mode)
        self.contour_mode_button.setChecked(mode == "contour")
        self.brush_mode_button.setChecked(mode == "brush")
        enable_brush = mode == "brush"
        self.add_brush_button.setEnabled(enable_brush)
        self.erase_brush_button.setEnabled(enable_brush)
        self.brush_slider.setEnabled(enable_brush)

    def _set_brush_mode(self, mode: str) -> None:
        self.editor.set_brush_mode(mode)
        self.add_brush_button.setChecked(mode == "add")
        self.erase_brush_button.setChecked(mode == "erase")

    def _on_brush_size_changed(self, value: int) -> None:
        self.editor.set_brush_radius(value)
        self.brush_label.setText(f"{value} px")

    def _accept_if_valid(self) -> None:
        try:
            self.session.build_mask()
        except ValueError as exc:
            QMessageBox.warning(self, "轮廓无效", str(exc))
            return
        self.accept()

    def build_manual_mask(self) -> np.ndarray:
        return self.session.build_mask()

    def build_edit_result(self):
        return self.session.build_edit_result()
