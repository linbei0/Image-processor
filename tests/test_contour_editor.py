from __future__ import annotations

import os

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QGraphicsView

from core.manual_refine import ManualRefineSession
from ui.contour_editor import ContourEditorView

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def editor_view(qtbot):
    mask = np.zeros((120, 120), dtype=np.float32)
    mask[20:100, 20:100] = 1.0
    session = ManualRefineSession.from_mask(mask)
    image = np.full((120, 120, 3), 180, dtype=np.uint8)
    view = ContourEditorView(session=session, image=image)
    qtbot.addWidget(view)
    view.resize(500, 500)
    view.show()
    return view


def test_space_key_no_longer_enters_pan_mode(editor_view, qtbot) -> None:
    qtbot.keyPress(editor_view.viewport(), Qt.Key_Space)
    assert editor_view.dragMode() == QGraphicsView.DragMode.NoDrag


def test_press_and_hold_background_enters_pan_mode(editor_view, qtbot) -> None:
    qtbot.mousePress(editor_view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert editor_view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    qtbot.mouseRelease(editor_view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert editor_view.dragMode() == QGraphicsView.DragMode.NoDrag


def test_editor_supports_switching_between_contour_and_brush_modes(editor_view) -> None:
    editor_view.set_edit_mode("brush")
    assert editor_view.edit_mode == "brush"
    assert editor_view.current_brush_mode == "add"
    editor_view.set_brush_mode("erase")
    assert editor_view.current_brush_mode == "erase"
    editor_view.set_edit_mode("contour")
    assert editor_view.edit_mode == "contour"
