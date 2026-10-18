# AGENTS.md — audio-to-midi

## Proyecto
Web local (Gradio): YouTube/mp3 → Demucs 6 stems → transcripción → full_mix.mid multicanal.
Mapeo: piano ch0/prog0, guitarra ch1/prog27, bajo ch2/prog33, voz ch4/prog53, batería ch9.

## Comandos (Windows PowerShell)
- Setup: `.\scripts\setup_env.ps1`
- App: `.\scripts\run_app.ps1`
- Tests rápidos: `.\scripts\test.ps1` (= `pytest -m "not slow"`)
- Lint: `.\scripts\lint.ps1`

## Convenciones
- Python 3.11 recomendado (3.14 actual solo para esqueleto; Demucs/MT3 CUDA piden 3.10/11).
- Type hints + docstrings estilo Google en todo `src/`.
- Funciones pequeñas; sin lógica de negocio en `app.py` (solo UI).
- Progreso en `docs/PROGRESS.md`; decisiones en `docs/DECISIONS.md`.

## Límites
- No commitear `outputs/`, `.venv/`, `*.wav`, `*.mid`, cookies ni tokens.
- Modelos pesados con mocks en tests (marca `slow` lo que use GPU/red).
- ffmpeg obligatorio; torch CUDA solo en venv 3.11, no en Python 3.14 base.
