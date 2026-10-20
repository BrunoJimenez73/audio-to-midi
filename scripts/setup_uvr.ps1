# setup_uvr.ps1 - Dependencias del modo "Mejor" (audio-separator/UVR).
# Ejecutar DESPUES de instalar requirements-full.txt, dentro del venv 3.11:
#   .\.venv311\Scripts\Activate.ps1 ; .\scripts\setup_uvr.ps1
# Orden crítico: audio-separator pide numpy>=2, pero TensorFlow 2.15
# (basic-pitch) necesita numpy<2 → se instala primero y se corrige al final.
$ErrorActionPreference = "Stop"
pip install audio-separator --progress-bar off
# audio-separator tira de onnx-weekly (protobuf>=6.31, incompatible con TF):
pip uninstall -y onnx-weekly
pip install onnx==1.17.0 onnxruntime --progress-bar off
# Restaurar pins del entorno TF:
pip install numpy==1.26.4 ml-dtypes==0.2.0 protobuf==4.25.9 --progress-bar off
python -c "import numpy, audio_separator, onnxruntime, tensorflow as tf; print('OK numpy', numpy.__version__, '| tf', tf.__version__)"
Write-Host "Listo. Los modelos (~2.5GB) se descargan a models\ en el primer uso del modo Mejor."
