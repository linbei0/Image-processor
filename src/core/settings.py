from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class AppSettings:
    last_export_dir: Path | None = None


class AppSettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        raw_dir = data.get("last_export_dir")
        return AppSettings(last_export_dir=Path(raw_dir) if raw_dir else None)

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(settings)
        payload["last_export_dir"] = str(settings.last_export_dir) if settings.last_export_dir else None
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_last_export_dir(self, export_dir: Path) -> None:
        settings = self.load()
        settings.last_export_dir = export_dir
        self.save(settings)
