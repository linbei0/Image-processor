from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from core.config import AppConfig
from core.pipeline import ProcessingPipeline
from core.settings import AppSettingsStore
from inference.engine import ModNetOnnxEngine
from ui.window import IdPhotoWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    config = AppConfig.default()
    pipeline = ProcessingPipeline(engine=ModNetOnnxEngine(config=config))
    window = IdPhotoWindow(
        pipeline=pipeline,
        settings_store=AppSettingsStore(config.settings_file),
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
