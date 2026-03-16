from __future__ import annotations

import numpy as np
import pytest

from core.manual_refine import (
    ManualEditResult,
    ManualRefineSession,
    compose_active_mask,
    extract_main_contour,
    polygon_to_mask,
    resample_closed_contour,
)


def test_extract_main_contour_returns_largest_external_contour() -> None:
    mask = np.zeros((80, 80), dtype=np.float32)
    mask[10:25, 10:25] = 1.0
    mask[30:70, 20:60] = 1.0

    contour = extract_main_contour(mask)

    assert contour.shape[1] == 2
    assert contour.shape[0] >= 4
    assert contour[:, 1].max() >= 60


def test_resample_closed_contour_returns_evenly_spaced_points() -> None:
    contour = np.array(
        [[10.0, 10.0], [70.0, 10.0], [70.0, 50.0], [10.0, 50.0]],
        dtype=np.float32,
    )

    points = resample_closed_contour(contour, target_spacing=10.0)

    assert points.shape[0] >= 16
    distances = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    assert distances.min() > 0
    assert distances.max() - distances.min() < 5.0


def test_manual_refine_session_supports_undo_and_redo() -> None:
    contour = np.array(
        [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
        dtype=np.float32,
    )
    session = ManualRefineSession(
        image_shape=(80, 80),
        auto_mask=np.zeros((80, 80), dtype=np.float32),
        default_points=contour,
        current_points=contour.copy(),
    )

    session.update_point(0, (12.0, 14.0))
    moved = session.current_points.copy()
    session.undo()
    undone = session.current_points.copy()
    session.redo()

    assert not np.array_equal(moved, contour)
    assert np.array_equal(undone, contour)
    assert np.array_equal(session.current_points, moved)


def test_polygon_to_mask_rejects_self_intersection() -> None:
    bow = np.array(
        [[10.0, 10.0], [50.0, 50.0], [10.0, 50.0], [50.0, 10.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="轮廓无效"):
        polygon_to_mask(bow, (80, 80))


def test_compose_active_mask_applies_contour_then_brush_add_and_erase() -> None:
    auto_mask = np.zeros((20, 20), dtype=np.float32)
    auto_mask[4:16, 4:16] = 1.0
    contour_mask = np.zeros((20, 20), dtype=np.float32)
    contour_mask[6:14, 6:14] = 1.0
    brush_add = np.zeros((20, 20), dtype=np.float32)
    brush_add[2:6, 2:6] = 1.0
    brush_erase = np.zeros((20, 20), dtype=np.float32)
    brush_erase[10:14, 10:14] = 1.0

    active = compose_active_mask(
        auto_mask=auto_mask,
        contour_mask=contour_mask,
        brush_add_mask=brush_add,
        brush_erase_mask=brush_erase,
    )

    assert active[3, 3] == 1.0
    assert active[11, 11] == 0.0
    assert active[7, 7] == 1.0
    assert active[5, 15] == 0.0


def test_manual_refine_session_supports_brush_add_erase_and_undo_redo() -> None:
    auto_mask = np.zeros((40, 40), dtype=np.float32)
    auto_mask[8:32, 8:32] = 1.0
    session = ManualRefineSession.from_mask(auto_mask)

    session.begin_brush_stroke()
    session.apply_brush((4.0, 4.0), radius=3, brush_mode="add")
    session.end_brush_stroke()
    assert session.active_mask[4, 4] == 1.0

    session.begin_brush_stroke()
    session.apply_brush((20.0, 20.0), radius=4, brush_mode="erase")
    session.end_brush_stroke()
    assert session.active_mask[20, 20] == 0.0

    session.undo()
    assert session.active_mask[20, 20] == 1.0
    session.redo()
    assert session.active_mask[20, 20] == 0.0


def test_manual_refine_session_exports_manual_edit_result() -> None:
    auto_mask = np.zeros((40, 40), dtype=np.float32)
    auto_mask[8:32, 8:32] = 1.0
    session = ManualRefineSession.from_mask(auto_mask)
    session.begin_brush_stroke()
    session.apply_brush((4.0, 4.0), radius=2, brush_mode="add")
    session.end_brush_stroke()

    result = session.build_edit_result()

    assert isinstance(result, ManualEditResult)
    assert result.active_mask[4, 4] == 1.0
    assert result.auto_mask.shape == auto_mask.shape


def test_manual_refine_session_caches_contour_and_active_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    auto_mask = np.zeros((40, 40), dtype=np.float32)
    auto_mask[8:32, 8:32] = 1.0
    session = ManualRefineSession.from_mask(auto_mask)
    calls = {"count": 0}
    original = polygon_to_mask

    def counting_polygon(points, image_shape):
        calls["count"] += 1
        return original(points, image_shape)

    monkeypatch.setattr("core.manual_refine.polygon_to_mask", counting_polygon)

    _ = session.active_mask
    _ = session.active_mask
    _ = session.contour_mask

    assert calls["count"] == 1


def test_manual_refine_session_invalidates_cache_after_brush_and_point_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto_mask = np.zeros((40, 40), dtype=np.float32)
    auto_mask[8:32, 8:32] = 1.0
    session = ManualRefineSession.from_mask(auto_mask)
    calls = {"count": 0}
    original = polygon_to_mask

    def counting_polygon(points, image_shape):
        calls["count"] += 1
        return original(points, image_shape)

    monkeypatch.setattr("core.manual_refine.polygon_to_mask", counting_polygon)

    _ = session.active_mask
    session.begin_brush_stroke()
    session.apply_brush((4.0, 4.0), radius=2, brush_mode="add")
    session.end_brush_stroke()
    _ = session.active_mask
    session.update_point(0, (12.0, 14.0))
    _ = session.active_mask

    assert calls["count"] == 2
