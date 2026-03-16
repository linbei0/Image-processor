from pathlib import Path

from core.settings import AppSettingsStore


def test_settings_store_persists_last_export_dir(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    store = AppSettingsStore(settings_file)

    assert store.load().last_export_dir is None

    store.save_last_export_dir(tmp_path / "exports")
    reloaded = AppSettingsStore(settings_file).load()

    assert reloaded.last_export_dir == tmp_path / "exports"
