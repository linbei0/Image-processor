from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


def numpy_rgb_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("预览图必须是 RGB 三通道数组")
    contiguous = np.ascontiguousarray(image)
    height, width = contiguous.shape[:2]
    bytes_per_line = contiguous.strides[0]
    qimage = QImage(contiguous.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return qimage.copy()


class ImagePreview(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("imagePreview")
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.setText(f"{self._title}\n\n等待图片")
        self.setPixmap(QPixmap())

    def set_array(self, image: np.ndarray) -> None:
        self._image = numpy_rgb_to_qimage(image)
        self._refresh_pixmap()

    def image(self) -> QImage | None:
        return self._image

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
