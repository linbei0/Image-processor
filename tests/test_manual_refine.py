from __future__ import annotations

import numpy as np
import pytest

from core.manual_refine import (
    ManualRefineSession,
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
