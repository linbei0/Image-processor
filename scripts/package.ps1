Param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

& $PythonExe -m pip install pyinstaller
& $PythonExe -m PyInstaller `
    --noconfirm `
    --windowed `
    --name IDPhotoBackgroundTool `
    --add-data "assets/models/modnet_photographic_portrait_matting.onnx;assets/models" `
    src/main.py
