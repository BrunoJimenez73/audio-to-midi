# PLAN — audio-to-midi (7 puntos)

## Punto 1 — Validar E2E con Demucs real
- [x] T1.1 Smoke test `slow` que corre `run_pipeline` sin `AUDIO2MIDI_NO_DEMUCS` sobre audio corto.
- [ ] T1.2 Script `scripts/e2e_demucs.ps1` con un mp3 real (usuario aporta).
- [ ] T1.3 Medir tiempo/ calidad y documentar en PROGRESS.

## Punto 2 — Backend ML para melodías (Basic Pitch)
- [x] T2.1 Worker `scripts/basicpitch_worker.py` (JSON por stdout).
- [x] T2.2 Módulo `src/transcribe_basicpitch.py` (in-process o subprocess).
- [x] T2.3 Integrar en `transcribe_melodic` vía `backend` + env `AUDIO2MIDI_MELODIC_BACKEND`.
- [x] T2.4 Tests `slow` para Basic Pitch.

## Punto 3 — Batería dedicada
- [ ] T3.1 Mejorar clasificación (más features) — pendiente.
- [ ] T3.2 Modelo ADT si se consigue paquete ligero — pendiente.

## Punto 4 — UI
- [x] T4.1 Selector de cuantización (1/8, 1/16, 1/32) → `grid_beats`.
- [x] T4.2 ZIP de descarga (midis + stems).
- [ ] T4.3 Preview MIDI — pendiente (requiere render audio).

## Punto 5 — Tests slow
- [x] T5.1 Demucs slow test.
- [x] T5.2 Basic Pitch slow test.

## Punto 6 — Higiene
- [x] T6.1 Lint limpio.
- [x] T6.2 `_fallback_hpss` crea `dest`.
- [ ] T6.3 Logging en pipeline/separate (prints → logger) — pendiente.
- [ ] T6.4 Dividir `voice.py`/`assign.py` — pendiente.
- [ ] T6.5 No tragar excepciones silenciosamente — parcial (logging añadido).

## Punto 7 — Reproducibilidad
- [x] T7.1 `.python-version`.
- [x] T7.2 CI workflow (lint + pytest not slow).
- [ ] T7.3 Lock de dependencias (pip freeze) — opcional, pesado.
"""