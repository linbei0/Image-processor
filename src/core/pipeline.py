from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import time

import numpy as np

from core.export import ExportService
from core.image_io import create_preview, load_image_rgb, resize_long_side
from core.image_ops import BackgroundComposer, EdgeRefiner
from core.manual_refine import ManualEditResult
from core.models import ProcessingRequest, ProcessingResult
from inference.engine import ModNetOnnxEngine


@dataclass(slots=True)
class CachedAlpha:
    signature: str
    alpha_mask: np.ndarray
    provider: str


class ProcessingPipeline:
    def __init__(
        self,
        engine: ModNetOnnxEngine,
        export_service: ExportService | None = None,
    ) -> None:
        self.engine = engine
        self.export_service = export_service or ExportService()
        self._cache: CachedAlpha | None = None

    def process(self, request: ProcessingRequest) -> ProcessingResult:
        started = time.perf_counter()
        image = load_image_rgb(request.input_path)
        image = resize_long_side(image, 1600)
        signature = self._build_signature(request.input_path, request.edge_refine_level)

        provider = self.engine.load(request.hardware_mode)
        self.engine.warmup()

        if self._cache and self._cache.signature == signature:
            alpha_mask = self._cache.alpha_mask
        else:
            raw_alpha = self.engine.infer(image)
            alpha_mask = EdgeRefiner(request.edge_refine_level).refine(image, raw_alpha)
            self._cache = CachedAlpha(signature=signature, alpha_mask=alpha_mask, provider=provider)

        final_image = BackgroundComposer.compose(image, alpha_mask, request.bg_color.rgb)
        preview_image = create_preview(final_image, max_side=960)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ProcessingResult(
            preview_image=preview_image,
            final_image=final_image,
            alpha_mask=alpha_mask,
            elapsed_ms=elapsed_ms,
            provider=provider,
            source_image=image,
        )

    def apply_manual_mask(
        self,
        request: ProcessingRequest,
        result: ProcessingResult,
        manual_mask: np.ndarray,
    ) -> ProcessingResult:
        image = result.source_image if result.source_image is not None else resize_long_side(load_image_rgb(request.input_path), 1600)
        refined_mask = EdgeRefiner(request.edge_refine_level).refine(image, manual_mask)
        final_image = BackgroundComposer.compose(image, refined_mask, request.bg_color.rgb)
        preview_image = create_preview(final_image, max_side=960)
        return ProcessingResult(
            preview_image=preview_image,
            final_image=final_image,
            alpha_mask=result.alpha_mask,
            elapsed_ms=0.0,
            provider=result.provider,
            source_image=image,
            manual_mask=refined_mask,
        )

    def apply_manual_edit(
        self,
        request: ProcessingRequest,
        result: ProcessingResult,
        edit_result: ManualEditResult,
    ) -> ProcessingResult:
        image = result.source_image if result.source_image is not None else resize_long_side(load_image_rgb(request.input_path), 1600)
        refined_active = EdgeRefiner(request.edge_refine_level).refine(image, edit_result.active_mask)
        final_image = BackgroundComposer.compose(image, refined_active, request.bg_color.rgb)
        preview_image = create_preview(final_image, max_side=960)
        return ProcessingResult(
            preview_image=preview_image,
            final_image=final_image,
            alpha_mask=result.alpha_mask,
            elapsed_ms=0.0,
            provider=result.provider,
            source_image=image,
            manual_mask=refined_active,
            manual_contour_mask=edit_result.contour_mask,
            manual_brush_add_mask=edit_result.brush_add_mask,
            manual_brush_erase_mask=edit_result.brush_erase_mask,
        )

    def export(self, path: Path, result: ProcessingResult, quality: int = 95) -> None:
        if path.suffix.lower() == ".png":
            self.export_service.save_png(path, result.final_image, result.active_mask)
            return
        self.export_service.save_jpg(path, result.final_image, quality=quality)

    @staticmethod
    def _build_signature(path: Path, edge_refine_level: int) -> str:
        stat = path.stat()
        raw = f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}::{edge_refine_level}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
