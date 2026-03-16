from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
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
    QToolBar,
    QVBoxLayout,
)

from core.manual_refine import ManualRefineSession


def _rgb_to_qimage(image: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(image)
    height, width = contiguous.shape[:2]
    return QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_RGB888).copy()


def _mask_overlay(mask: np.ndarray) -> QPixmap:
    alpha = np.clip(mask * 130.0, 0, 130).astype(np.uint8)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 1] = 180
    rgba[..., 2] = 255
    rgba[..., 3] = alpha
    qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


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
    def __init__(self, session: ManualRefineSession, image: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.image = image
        self.selected_index: int | None = None
        self._moving_handle_index: int | None = None
        self._panning = False
        self._did_initial_fit = False
        self._user_changed_view = False
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#111827"))
        self.setSceneRect(0, 0, image.shape[1], image.shape[0])

        self._image_item = QGraphicsPixmapItem(QPixmap.fromImage(_rgb_to_qimage(image)))
        self._overlay_item = QGraphicsPixmapItem(_mask_overlay(session.build_mask()))
        self._overlay_item.setOpacity(0.45)
        self._path_item = QGraphicsPathItem()
        self._path_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        self._path_item.setZValue(10)
        self._scene.addItem(self._image_item)
        self._scene.addItem(self._overlay_item)
        self._scene.addItem(self._path_item)
        self._handle_items: list[ControlPointItem] = []
        self._rebuild_items()
        self._fit_scene_to_view()

    def begin_handle_move(self, index: int) -> None:
        self._moving_handle_index = index
        self.session.push_history()

    def preview_handle_move(self, index: int, pos: QPointF) -> None:
        if self._moving_handle_index != index:
            return
        self.session.current_points[index] = np.array([pos.x(), pos.y()], dtype=np.float32)
        self._refresh_path_only()

    def finish_handle_move(self, index: int, pos: QPointF) -> None:
        self._moving_handle_index = None
        self.session.current_points[index] = np.array([pos.x(), pos.y()], dtype=np.float32)
        self._refresh_path_only()

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

    def wheelEvent(self, event) -> None:  # noqa: N802
        self._user_changed_view = True
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected_point()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(item, ControlPointItem):
            self._user_changed_view = True
            self._panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.insert_point_at_scene_pos(self.mapToScene(event.position().toPoint()))
        event.accept()

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
        self._refresh_path_only()

    def _fit_scene_to_view(self) -> None:
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _refresh_path_only(self) -> None:
        points = self.session.current_points
        path = QPainterPath(QPointF(float(points[0][0]), float(points[0][1])))
        for point in points[1:]:
            path.lineTo(float(point[0]), float(point[1]))
        path.closeSubpath()
        self._path_item.setPath(path)
        try:
            self._overlay_item.setPixmap(_mask_overlay(self.session.build_mask()))
            self._path_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        except ValueError:
            self._path_item.setPen(QPen(QColor("#ef4444"), 2.0))


class ContourEditorDialog(QDialog):
    def __init__(self, session: ManualRefineSession, image: np.ndarray | None, parent=None) -> None:
        super().__init__(parent)
        if image is None:
            raise ValueError("手动微调需要原始图像")
        self.session = session
        self.setWindowTitle("手动轮廓微调")
        self.resize(1080, 820)

        layout = QVBoxLayout(self)
        toolbar = QToolBar()
        undo_action = toolbar.addAction("撤销")
        redo_action = toolbar.addAction("重做")
        reset_action = toolbar.addAction("恢复自动轮廓")
        layout.addWidget(toolbar)

        tips = QLabel("滚轮缩放，长按空白区域拖动画布，拖拽控制点调整轮廓，双击线段插入点，Delete 删除选中点。")
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

        undo_action.triggered.connect(self.editor.undo)
        redo_action.triggered.connect(self.editor.redo)
        reset_action.triggered.connect(self.editor.reset)
        cancel_button.clicked.connect(self.reject)
        apply_button.clicked.connect(self._accept_if_valid)

    def _accept_if_valid(self) -> None:
        try:
            self.session.build_mask()
        except ValueError as exc:
            QMessageBox.warning(self, "轮廓无效", str(exc))
            return
        self.accept()

    def build_manual_mask(self) -> np.ndarray:
        return self.session.build_mask()
