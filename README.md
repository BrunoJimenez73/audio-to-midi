# audio-to-midi

Web local: YouTube/mp3 → stems Demucs → MIDI multicanal.

## Estado
v0 presentable ✅ (8 tests verdes, E2E mp3→MIDI verificado).

## Uso rápido
```powershell
.\scripts\setup_env.ps1
.\scripts\test.ps1
.\scripts\run_app.ps1
```

## Notas
- Base: Python 3.14 + torch CPU (esqueleto).
- Pesado (Demucs/MT3 CUDA RTX 4060): venv Python 3.11 + `requirements-full.txt`.
- ffmpeg ya instalado ✅.
