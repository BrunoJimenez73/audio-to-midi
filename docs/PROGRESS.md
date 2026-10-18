# PROGRESS

| Fase | Estado | Notas |
|---|---|---|
| Harness | 🟢 hecho | AGENTS.md, TODO, scripts ps1, 18 tests verdes + 2 slow |
| Ingesta | 🟢 funciona | yt-dlp + ffmpeg, YouTube real (tema 5:18, 318s) |
| Separación | 🟢 Demucs GPU | `htdemucs_6s` en `.venv311` (torch cu126, RTX 4060, ~17s/tema); fallback numpy |
| Melódico | 🟢 Basic Pitch | worker batch (1 TF load p/ todos los stems, ~40s); voz con umbrales suaves |
| Assign | 🟢 bypass con Demucs | stems reales → se confía en cada stem; assign solo en fallback numpy |
| Batería | 🟢 con gate | rejilla BPM: vacía si no hay batería real (balada → 0) |
| Tempo | 🟢 73.5 BPM | octava corregida + override manual en app |
| Merge MIDI | 🟢 tempo real | conductor + cuantización 16ths |
| Web Gradio | 🟢 presentable | url+upload, BPM manual, stems, full_mix.mid |
| E2E YouTube | 🟢 v8 | `outputs/20261018_183408/midi/full_mix.mid` (82s end-to-end) |

v8 (balada guitarra+voz, Demucs + Basic Pitch): guitarra 1897 (p50 0.41s),
voz 858 (~189s cantados), piano 573, bajo 270, other 513, batería 0.

Antes (v6, transcriptor numpy sin Demucs): guitarra 3606 notas fragmentadas,
voz 2143 con armónicos duplicados.

Límites conocidos: cuantización 16ths trocea voz (p90≈0.2s); piano puede
heredar bajos del stem (rango 21); `other` puede contener pad/ruido.
