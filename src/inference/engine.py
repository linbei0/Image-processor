from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from core.config import AppConfig
from core.models import HardwareMode


MODEL_FILE_NAME = "modnet_photographic_portrait_matting.onnx"
BUNDLED_MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "models" / MODEL_FILE_NAME


def _normalize(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    original_h, original_w = image.shape[:2]
    target_h, target_w = size
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = ((tensor - 0.5) / 0.5).transpose(2, 0, 1)[None, ...]
    return tensor, (original_h, original_w)


@dataclass(slots=True)
class ProviderInfo:
    active_provider: str
    session: ort.InferenceSession


class ModNetOnnxEngine:
    def __init__(
        self,
        config: AppConfig | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.config = config or AppConfig.default()
        self.config.ensure_directories()
        self.model_path = model_path or self._resolve_default_model_path()
        self._provider_info: ProviderInfo | None = None
        self._last_mode: HardwareMode | None = None

    def get_available_providers(self) -> list[str]:
        return list(ort.get_available_providers())

    def load(self, hardware_mode: HardwareMode) -> str:
        if self._provider_info is not None and self._last_mode == hardware_mode:
            return self._provider_info.active_provider
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到模型文件: {self.model_path}。请先运行 scripts/download_model.py 下载 MODNet ONNX 模型。"
            )
        providers = self._resolve_providers(hardware_mode)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=providers)
        self._provider_info = ProviderInfo(active_provider=session.get_providers()[0], session=session)
        self._last_mode = hardware_mode
        return self._provider_info.active_provider

    def warmup(self) -> None:
        if self._provider_info is None:
            return
        height, width = self.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self._provider_info.session.run(None, {self.input_name: dummy})

    @property
    def input_name(self) -> str:
        if self._provider_info is None:
            raise RuntimeError("模型尚未加载")
        return self._provider_info.session.get_inputs()[0].name

    @property
    def input_size(self) -> tuple[int, int]:
        if self._provider_info is None:
            raise RuntimeError("模型尚未加载")
        shape = self._provider_info.session.get_inputs()[0].shape
        height = int(shape[2])
        width = int(shape[3])
        return height, width

    def infer(self, image: np.ndarray) -> np.ndarray:
        if self._provider_info is None:
            raise RuntimeError("调用 infer 前必须先 load 模型")
        tensor, (original_h, original_w) = _normalize(image, self.input_size)
        outputs = self._provider_info.session.run(None, {self.input_name: tensor})
        alpha = outputs[0][0][0]
        alpha = cv2.resize(alpha, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        return alpha

    def _resolve_default_model_path(self) -> Path:
        if BUNDLED_MODEL_PATH.exists():
            return BUNDLED_MODEL_PATH
        return self.config.model_dir / MODEL_FILE_NAME

    def _resolve_providers(self, hardware_mode: HardwareMode) -> list[str]:
        available = self.get_available_providers()
        if hardware_mode is HardwareMode.GPU:
            if "DmlExecutionProvider" not in available:
                raise RuntimeError("当前环境未检测到 DirectML，无法启用 GPU 模式")
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if hardware_mode is HardwareMode.CPU:
            return ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
