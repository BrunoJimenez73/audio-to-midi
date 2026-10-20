# audio-to-midi

Web local: YouTube/mp3 → stems (UVR/Roformer + Demucs) → MIDI multicanal.

## Estado
v10 ✅ (19 tests verdes, E2E YouTube verificado, exe Windows funcional,
separación Mejor/Rápido con bench A/B).

## Uso rápido
```powershell
.\scripts\setup_env.ps1
.\scripts\setup_uvr.ps1              # solo venv 3.11: deps del modo Mejor
.\scripts\test.ps1
.\scripts\run_app.ps1
```

## Separación: Mejor vs Rápido
- **Mejor (por defecto)**: jerarquía `audio-separator` en
  `scripts/separator_worker.py` → ① Roformer vocals (Jensen, SDR 12.6)
  ② `htdemucs_ft` ③ `htdemucs_6s` sobre el residual (guitar/piano/other).
  ~89s en tema de 5:18 (VRAM pico 1.8GB).
- **Rápido**: `htdemucs_6s` directo, ~15s/tema.
- Modelos (~2.5GB) en `models/` (no commitear); se descargan solos en el
  primer uso si faltan. Gates: `AUDIO2MIDI_NO_UVR=1` (fuerza fallback),
  `AUDIO2MIDI_MODELS_DIR` (ruta de modelos).
- Bench A/B (stems escuchables + tabla): `python scripts\compare_sep.py`
  → `outputs/bench_sep/<ts>/`.

## Exe (Windows)
```powershell
.\scripts\build_exe.ps1                      # construye dist\audio-to-midi\
.\dist\audio-to-midi\audio-to-midi.exe       # UI en http://127.0.0.1:7860
.\dist\audio-to-midi\audio-to-midi.exe --selftest   # smoke test sin red
```
- Onedir (~68MB exe + `_internal`); requiere el venv `.venv311` y ffmpeg en
  la misma máquina (el ffmpeg se copia al dist en el build).
- Si mueves la carpeta, mueve también `.venv311` junto al dist (el exe lo
  busca subiendo directorios).

## Notas
- Base: Python 3.14 + torch CPU (esqueleto).
- Pesado (Demucs/Basic Pitch CUDA RTX 4060): venv Python 3.11 + `requirements-full.txt`.
- ffmpeg ya instalado ✅.
