# PROGRESS

| Fase | Estado | Notas |
|---|---|---|
| Harness | 🟢 hecho | AGENTS.md, TODO, scripts ps1, 18 tests verdes + 2 slow |
| Ingesta | 🟢 funciona | yt-dlp + ffmpeg, YouTube real (tema 5:18, 318s) |
| Separación | 🟢 v10 Mejor/Rápido | UVR jerárquico (Roformer voz + ft + 6s, ~89s/tema) y `htdemucs_6s` (~15s); fallback numpy |
| Melódico | 🟢 Basic Pitch | worker batch (1 TF load p/ todos los stems, ~40s); voz con umbrales suaves |
| Assign | 🟢 bypass con Demucs | stems reales → se confía en cada stem; assign solo en fallback numpy |
| Batería | 🟢 con gate | rejilla BPM: vacía si no hay batería real (balada → 0) |
| Tempo | 🟢 73.5 BPM | octava corregida + override manual en app |
| Merge MIDI | 🟢 tempo real | conductor + cuantización 16ths |
| Web Gradio | 🟢 ampliada | URL/upload, modelo+shifts Demucs, backend transcripción, BPM, cuantización (o sin), batería on/off, stems+MIDI por pista+zip |
| E2E YouTube | 🟢 v8 | `outputs/20261018_183408/midi/full_mix.mid` (82s end-to-end) |
| Empaquetado | 🟢 v9 | `dist\audio-to-midi\audio-to-midi.exe` (onedir ~68MB + `_internal` + ffmpeg), usa `.venv311` |

v10 (2026-10-20, separación mejorada): modo **Mejor** (por defecto) con
`audio-separator` 0.47.1 en `scripts/separator_worker.py` (3 pasadas:
Roformer vocals Jensen → htdemucs_ft → htdemucs_6s sobre residual; VRAM
pico 1.8GB). Medido en tema real 318s: **mejor 89s / rápido 15s**, 6 stems
completos; drums +12.8dB RMS retenido vs rápido. Bench A/B en
`outputs/bench_sep/<ts>/` con README de escucha (`scripts/compare_sep.py`).
Nuevas env gates: `AUDIO2MIDI_NO_UVR` y `AUDIO2MIDI_MODELS_DIR`; modelos en
`models/` (2.5GB, no commitear). UI: radio Rápido/Mejor en acordeón
"Separación" (Mejor por defecto). Deps vía `scripts/setup_uvr.ps1`
(downgrade numpy/protobuf obligatorio tras audio-separator).

v9 (2026-10-20, empaquetado): exe Windows con PyInstaller onedir construido con
`scripts\build_exe.ps1`. Verificado: `audio-to-midi.exe --selftest` → Demucs
6 stems + BP con notas reales (exit 0) y UI Gradio HTTP 200 en 127.0.0.1:7860.
Bug resuelto: el worker insertaba dirname² en `sys.path` sin comprobar que
contuviera `src/` → en el bundle eso metía `_internal` (con `select.pyd` de
Python 3.14) y rompía `import select` en el intérprete 3.11 (se ejecuta
siempre via `.venv311\Scripts\python.exe`); fix con guard en
`scripts/basicpitch_worker.py`.

v8 (balada guitarra+voz, Demucs + Basic Pitch): guitarra 1897 (p50 0.41s),
voz 858 (~189s cantados), piano 573, bajo 270, other 513, batería 0.

Antes (v6, transcriptor numpy sin Demucs): guitarra 3606 notas fragmentadas,
voz 2143 con armónicos duplicados.

Límites conocidos: cuantización 16ths trocea voz (p90≈0.2s); piano puede
heredar bajos del stem (rango 21); `other` puede contener pad/ruido.
