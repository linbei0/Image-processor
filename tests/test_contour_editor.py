from __future__ import annotations

import os

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtWidgets import QGraphicsView

from core.manual_refine import ManualRefineSession
import ui.contour_editor as contour_editor_module
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


def test_editor_throttles_repeated_brush_refreshes(
    editor_view,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor_view.set_edit_mode("brush")
    refresh_calls = {"count": 0}

    def fake_refresh() -> None:
        refresh_calls["count"] += 1

    timeline = iter([0.0, 0.005, 0.006, 0.030])
    monkeypatch.setattr(editor_view, "_refresh_visuals", fake_refresh)
    monkeypatch.setattr(contour_editor_module.time, "monotonic", lambda: next(timeline))

    editor_view._maybe_refresh_visuals()
    editor_view._maybe_refresh_visuals()
    editor_view._maybe_refresh_visuals(force=True)

    assert refresh_calls["count"] == 2


def test_editor_uses_partial_overlay_refresh_for_brush_updates(
    editor_view,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor_view.set_edit_mode("brush")
    editor_view.session.begin_brush_stroke()
    calls = {"full": 0, "partial": 0}

    monkeypatch.setattr(editor_view, "_refresh_visuals", lambda: calls.__setitem__("full", calls["full"] + 1))
    monkeypatch.setattr(
        editor_view,
        "_refresh_overlay_region",
        lambda rect: calls.__setitem__("partial", calls["partial"] + 1),
    )

    editor_view._apply_brush_at_scene_pos(QPointF(20.0, 20.0))

    assert calls["full"] == 0
    assert calls["partial"] == 1
