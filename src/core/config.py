from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


APP_NAME = "IDPhotoBackgroundTool"


@dataclass(slots=True, frozen=True)
class AppConfig:
    app_dir: Path
    model_dir: Path
    cache_dir: Path
    settings_file: Path
    last_export_file: Path

    @classmethod
    def default(cls) -> "AppConfig":
        root = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
        return cls(
            app_dir=root,
            model_dir=root / "models",
            cache_dir=root / "cache",
            settings_file=root / "settings.json",
            last_export_file=root / "last_export.txt",
        )

    def ensure_directories(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
