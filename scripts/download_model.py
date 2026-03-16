from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from core.config import AppConfig
from inference.engine import BUNDLED_MODEL_PATH, MODEL_FILE_NAME


MODEL_URL = "https://github.com/jkjung-avt/tensorrt_demos/raw/master/modnet/modnet.onnx"
MIN_MODEL_SIZE = 20_000_000


def _validate_download(destination: Path) -> None:
    if not destination.exists() or destination.stat().st_size < MIN_MODEL_SIZE:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(
            f"模型下载不完整，目标文件小于 {MIN_MODEL_SIZE} 字节。"
            " 请检查网络后重新运行脚本。"
        )


def _download_with_powershell(url: str, destination: Path) -> None:
    command = [
        "powershell.exe",
        "-Command",
        f"Invoke-WebRequest -Uri '{url}' -OutFile '{destination}'",
    ]
    subprocess.run(command, check=True)


def _download_with_urllib(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"开始下载模型到: {destination}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100.0
                print(f"\r已下载 {percent:5.1f}%", end="")
    print()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        _download_with_powershell(url, destination)
    else:
        _download_with_urllib(url, destination)
    _validate_download(destination)
    print("模型下载完成")


def main() -> None:
    config = AppConfig.default()
    target = config.model_dir / MODEL_FILE_NAME
    if target.exists():
        print(f"模型已存在: {target}")
        return
    if BUNDLED_MODEL_PATH.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED_MODEL_PATH, target)
        print(f"已从项目内置模型复制到: {target}")
        return
    download_file(MODEL_URL, target)


if __name__ == "__main__":
    main()
