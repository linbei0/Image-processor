from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


MIN_POLYGON_POINTS = 4
DEFAULT_HISTORY_LIMIT = 32


def extract_main_contour(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到有效主轮廓")
    largest = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.005 * cv2.arcLength(largest, True))
    approx = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2).astype(np.float32)
    if len(approx) < MIN_POLYGON_POINTS:
        raise ValueError("未找到有效主轮廓")
    return approx


def resample_closed_contour(contour: np.ndarray, target_spacing: float = 12.0) -> np.ndarray:
    if len(contour) < 2:
        raise ValueError("轮廓点不足")
    closed = np.vstack([contour, contour[0]])
    segments = np.diff(closed, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    perimeter = float(lengths.sum())
    target_count = max(MIN_POLYGON_POINTS, int(round(perimeter / max(target_spacing, 1.0))))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, perimeter, num=target_count, endpoint=False)
    points: list[np.ndarray] = []
    for distance in samples:
        seg_index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        seg_index = min(seg_index, len(segments) - 1)
        seg_start = closed[seg_index]
        seg_length = max(lengths[seg_index], 1e-6)
        local = (distance - cumulative[seg_index]) / seg_length
        points.append(seg_start + segments[seg_index] * local)
    return np.asarray(points, dtype=np.float32)


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    d1 = _cross(a1, a2, b1)
    d2 = _cross(a1, a2, b2)
    d3 = _cross(b1, b2, a1)
    d4 = _cross(b1, b2, a2)
    return ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4))


def is_self_intersecting(points: np.ndarray) -> bool:
    count = len(points)
    if count < MIN_POLYGON_POINTS:
        return True
    for i in range(count):
        a1 = points[i]
        a2 = points[(i + 1) % count]
        for j in range(i + 1, count):
            if abs(i - j) <= 1 or {i, j} == {0, count - 1}:
                continue
            b1 = points[j]
            b2 = points[(j + 1) % count]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def polygon_to_mask(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if len(points) < MIN_POLYGON_POINTS or is_self_intersecting(points):
        raise ValueError("轮廓无效，请调整")
    height, width = image_shape
    polygon = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    polygon[:, 0, 0] = np.clip(polygon[:, 0, 0], 0, width - 1)
    polygon[:, 0, 1] = np.clip(polygon[:, 0, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(mask, [polygon], 1.0)
    return mask


@dataclass(slots=True)
class ManualRefineSession:
    image_shape: tuple[int, int]
    default_points: np.ndarray
    current_points: np.ndarray
    history_limit: int = DEFAULT_HISTORY_LIMIT
    _undo_stack: list[np.ndarray] = field(default_factory=list)
    _redo_stack: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_mask(cls, mask: np.ndarray, target_spacing: float = 12.0) -> "ManualRefineSession":
        contour = extract_main_contour(mask)
        points = resample_closed_contour(contour, target_spacing=target_spacing)
        return cls(image_shape=mask.shape[:2], default_points=points.copy(), current_points=points.copy())

    def snapshot(self) -> np.ndarray:
        return self.current_points.copy()

    def push_history(self) -> None:
        self._undo_stack.append(self.current_points.copy())
        if len(self._undo_stack) > self.history_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def update_point(self, index: int, point: tuple[float, float]) -> None:
        self.push_history()
        x = float(np.clip(point[0], 0, self.image_shape[1] - 1))
        y = float(np.clip(point[1], 0, self.image_shape[0] - 1))
        self.current_points[index] = np.array([x, y], dtype=np.float32)

    def insert_point(self, segment_index: int, point: tuple[float, float]) -> None:
        self.push_history()
        insert_at = segment_index + 1
        new_point = np.array([[point[0], point[1]]], dtype=np.float32)
        self.current_points = np.insert(self.current_points, insert_at, new_point, axis=0)

    def delete_point(self, index: int) -> None:
        if len(self.current_points) <= MIN_POLYGON_POINTS:
            raise ValueError("控制点数量不能再减少")
        self.push_history()
        self.current_points = np.delete(self.current_points, index, axis=0)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.current_points.copy())
        self.current_points = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.current_points.copy())
        self.current_points = self._redo_stack.pop()
        return True

    def reset(self) -> None:
        self.push_history()
        self.current_points = self.default_points.copy()

    def build_mask(self) -> np.ndarray:
        return polygon_to_mask(self.current_points, self.image_shape)
