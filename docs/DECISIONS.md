# DECISIONS

- 2026-10-12: Demucs `htdemucs_6s` (drums/bass/vocals/guitar/piano/other).
  Motivo: único que da las 5 pistas pedidas.
- 2026-10-12: Piano → `piano-transcription` (Kong); resto melódico → MT3;
  fallback global Basic Pitch si MT3 falla en Windows.
- 2026-10-12: Batería con modelo ADT dedicado, no solo onsets.
- 2026-10-12: Gradio sobre Streamlit (audio nativo + menos código).
- 2026-10-12: Python 3.11 para entorno pesado CUDA; 3.14 base solo esqueleto.
- 2026-10-18: Demucs 6 stems corre en venv `.venv311` (torch cu126, RTX 4060);
  pipeline lo invoca por subprocesso y detecta stems reales por estéreo.
- 2026-10-18: Basic Pitch ( Spotify ) como transcriptor melódico por defecto
  vía worker batch (una carga TF para todos los stems, ~40 s). Motivo: en el
  tema real, guitarra 3606→~1900 notas musicales vs el transcriptor numpy.
- 2026-10-18: Voz con umbrales BP suaves (onset 0.3 / frame 0.2): evita
  trocear sílabas (188 s cantados vs 67 s con umbrales por defecto).
- 2026-10-18: Con stems Demucs reales se omite `assign_notes` (cada stem ya
  es un instrumento); assign solo actúa en el fallback numpy.
- 2026-10-18: Env gates para tests: `AUDIO2MIDI_NO_DEMUCS=1`,
  `AUDIO2MIDI_NO_BP=1` (ambos en `test_pipeline_e2e`).
- 2026-10-20: Exe PyInstaller **onedir** que reutiliza el venv `.venv311`
  existente (Decisión del usuario por encuesta; autocontenido de 6-8GB
  descartado). Demucs/TF corren en subprocessos `.venv311\Scripts\python.exe`
  localizado con `find_up`, no dentro del exe.
- 2026-10-20: El worker BP solo inserta dirname² en `sys.path` si contiene
  `src/`. Motivo: en el bundle dirname² = `_internal`, que lleva `select.pyd`
  de Python 3.14 → `ImportError: Module use of python314.dll conflicts`
  en el 3.11.
- 2026-10-20: Separación "Mejor" jerárquica con `audio-separator` 0.47.1
  (3 pasadas en `scripts/separator_worker.py`, subprocesso venv 3.11):
  ① Roformer vocals (Jensen `vocals_mel_band_roformer.ckpt`, segment 160,
  fp16) → ② `htdemucs_ft` (shifts=1) → ③ `htdemucs_6s` sobre el residual
  → guitar/piano/other. Motivo: medido en GPU, ~89s/tema de 318s
  (voz 26 + ft 43 + 6s 14) vs 15s del modo rápido; SDR voz 12.6. UI con
  selector Mejor/Rápido (por defecto Mejor). Fallback: NO_UVR → hpss,
  mejor→rápido→demucs→hpss.
- 2026-10-20: Descartes de modelos: BS-Roformer (25s/45s clip, muy lento),
  MVSep Mega 53-stems (16GB VRAM > 8GB de la RTX 4060); onnxruntime-gpu
  1.31 trae CUDA 13 → warnings con torch cu126, se usa `onnxruntime` CPU
  (solo MDX lo necesita y no usamos MDX). `onnx==1.17.0` fijado porque
  `onnx-weekly` exige protobuf>=6.31 incompatible con TF 2.15.
- 2026-10-20: Pins TF en venv 3.11 tras instalar audio-separator:
  `numpy==1.26.4 ml-dtypes==0.2.0 protobuf==4.25.9` (audio-separator
  declara numpy>=2 pero funciona con 1.26.4). Secuencia exacta en
  `scripts/setup_uvr.ps1`; `requirements-full.txt` no lo incluye para no
  romper el resolver.
