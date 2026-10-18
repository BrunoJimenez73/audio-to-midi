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
