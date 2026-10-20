# build_exe.ps1 - Construye dist\audio-to-midi\audio-to-midi.exe
# Requiere: pip install pyinstaller (Python base 3.14)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Join-Path $PSScriptRoot "..")

python -m PyInstaller --noconfirm --clean `
    --name audio-to-midi `
    --onedir `
    --collect-all gradio `
    --collect-all gradio_client `
    --collect-data safehttpx `
    --collect-data groovy `
    --collect-submodules uvicorn `
    --collect-submodules yt_dlp `
    --collect-submodules anyio `
    --add-data "scripts\basicpitch_worker.py;scripts" `
    --add-data "scripts\separator_worker.py;scripts" `
    app.py

# ffmpeg.exe junto al exe (la app lo añade al PATH al arrancar)
$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if ($ffmpeg) {
    Copy-Item -LiteralPath $ffmpeg `
        -Destination "dist\audio-to-midi\ffmpeg.exe" -Force
    Write-Host "ffmpeg copiado: dist\audio-to-midi\ffmpeg.exe"
} else {
    Write-Warning "ffmpeg no está en PATH: el exe usará el del sistema."
}

Write-Host ""
Write-Host "Listo: dist\audio-to-midi\audio-to-midi.exe"
Write-Host "Smoke test: dist\audio-to-midi\audio-to-midi.exe --selftest"
