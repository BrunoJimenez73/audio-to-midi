"""UI web local: URL YouTube o mp3 -> stems + MIDI multicanal."""
import os
import sys
import zipfile
from pathlib import Path

import gradio as gr

from src.pipeline import run_pipeline

if getattr(sys, "frozen", False):
    # Modo exe: ffmpeg.exe empaquetado junto al binario.
    os.environ["PATH"] = (
        str(Path(sys.executable).resolve().parent) + os.pathsep
        + os.environ.get("PATH", "")
    )

ORDER = ("drums", "bass", "vocals", "guitar", "piano", "other")
GRID_MAP = {"Sin cuantizar": 0.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125}
BACKEND_MAP = {"Auto (recomendado)": "auto",
               "Basic Pitch (calidad ML)": "bp",
               "Ligero (numpy, sin ML)": "numpy"}
MODELS = {"htdemucs_6s (6 pistas: guitarra, piano, voz…)": "htdemucs_6s",
          "htdemucs (4 pistas, más rápido)": "htdemucs"}
SEP_MAP = {"Mejor (UVR/Roformer + Demucs, ~2-4 min)": "mejor",
           "Rápido (Demucs 6s, ~20s)": "rapido"}


def _pick(user_url: str, user_file: str | None) -> str:
    """Pick effective input from URL box or uploaded file.

    Args:
        user_url: YouTube URL or local path text.
        user_file: Uploaded file path from Gradio.

    Returns:
        Effective input string.

    Raises:
        ValueError: If neither provided.
    """
    if user_file:
        return str(user_file)
    text = (user_url or "").strip()
    if not text:
        raise ValueError("Pega una URL o sube un mp3/wav.")
    return text


def process(user_url: str, user_file: str | None, bpm: float,
            quant: str, backend_label: str, sep_label: str,
            model_label: str, shifts: float, do_drums: bool,
            progress: gr.Progress = gr.Progress()) -> tuple:
    """Run pipeline and return status + artifacts.

    Args:
        user_url: URL text.
        user_file: Uploaded audio path.
        bpm: Forced tempo (0 = auto).
        quant: Quantization label.
        backend_label: Transcription backend label.
        sep_label: Separation mode label (mejor/rapido).
        model_label: Demucs model label (modo rapido).
        shifts: Demucs shifts (quality, modo rapido).
        do_drums: Transcribe drums or not.
        progress: Gradio progress tracker.

    Returns:
        Tuple (status, 6 stems audio, full_mix, 6 per-track midis, zip).
    """
    n_out = 1 + 6 + 1 + 6 + 1
    try:
        src = _pick(user_url, user_file)
    except ValueError as exc:
        return (f"Error: {exc}",) + (None,) * (n_out - 1)
    try:
        res = run_pipeline(
            src, progress=progress, bpm=float(bpm or 0),
            grid_beats=GRID_MAP.get(quant, 0.25),
            backend=BACKEND_MAP.get(backend_label, "auto"),
            sep=SEP_MAP.get(sep_label, "mejor"),
            model=MODELS.get(model_label, "htdemucs_6s"),
            shifts=int(shifts), drums=bool(do_drums))
    except Exception as exc:  # noqa: BLE001
        return (f"Error: {exc}",) + (None,) * (n_out - 1)
    stems = res["stems"]
    assert isinstance(stems, dict)
    audios = [str(stems[s]) if s in stems else None for s in ORDER]
    midis = res["midis"]
    assert isinstance(midis, dict)
    per_track = [str(midis[i]) if i in midis else None
                 for i in ("guitar", "vocals", "piano", "bass", "other", "drums")]
    run_dir = Path(str(res["run_dir"]))
    zip_path = run_dir / "midis.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for mid in sorted((run_dir / "midi").glob("*.mid")):
                zf.write(mid, arcname=mid.name)
    except Exception:  # noqa: BLE001
        zip_path = None
    counts = res["notes_count"]
    msg = f"OK ({res['bpm']} BPM) en {run_dir}. Notas: {counts}."
    return (msg, *audios, str(res["full_mix"]), *per_track,
            str(zip_path) if zip_path else None)


with gr.Blocks(title="audio-to-midi") as demo_ui:
    gr.Markdown(
        "# 🎵 audio-to-midi\n"
        "URL de YouTube o archivo de audio → **6 stems** (UVR/Roformer + "
        "Demucs GPU) → **MIDI multicanal** (Basic Pitch). "
        "Abre `full_mix.mid` en tu DAW."
    )
    with gr.Row():
        url = gr.Textbox(label="URL de YouTube (o ruta local)",
                          placeholder="https://www.youtube.com/watch?v=...")
        f = gr.File(label="o sube un audio (mp3/wav/m4a…)",
                    file_types=["audio"])
    with gr.Accordion("Separación", open=False):
        sep = gr.Radio(choices=list(SEP_MAP), value=list(SEP_MAP)[0],
                       label="Modo de separación")
        with gr.Row():
            model = gr.Dropdown(choices=list(MODELS), value=list(MODELS)[0],
                                label="Modelo (solo Rápido)")
            shifts = gr.Slider(1, 5, value=1, step=1,
                               label="Calidad shifts (solo Rápido): "
                                     "más = mejor y más lento")
    with gr.Accordion("Conversión a MIDI", open=False):
        with gr.Row():
            backend = gr.Radio(choices=list(BACKEND_MAP), value=list(BACKEND_MAP)[0],
                               label="Transcriptor")
            do_drums = gr.Checkbox(value=True, label="Transcribir batería")
        with gr.Row():
            bpm = gr.Number(label="BPM (0 = auto-detectar)", value=0,
                            minimum=0, maximum=220)
            quant = gr.Dropdown(choices=list(GRID_MAP), value="1/16",
                                label="Cuantización")
    btn = gr.Button("Transcribir", variant="primary")
    status = gr.Textbox(label="Estado")
    with gr.Tab("Stems (audio)"):
        with gr.Row():
            a_drums = gr.Audio(label="drums", type="filepath")
            a_bass = gr.Audio(label="bass", type="filepath")
            a_voc = gr.Audio(label="vocals", type="filepath")
        with gr.Row():
            a_gui = gr.Audio(label="guitar", type="filepath")
            a_pia = gr.Audio(label="piano", type="filepath")
            a_oth = gr.Audio(label="other", type="filepath")
    with gr.Tab("MIDI"):
        full_out = gr.File(label="⬇ full_mix.mid (multicanal)")
        with gr.Row():
            g_out = gr.File(label="guitar.mid")
            v_out = gr.File(label="vocals.mid")
            p_out = gr.File(label="piano.mid")
        with gr.Row():
            b_out = gr.File(label="bass.mid")
            d_out = gr.File(label="drums.mid")
            o_out = gr.File(label="other.mid")
        zip_out = gr.File(label="⬇ todos (midis.zip)")
    btn.click(process,
              inputs=[url, f, bpm, quant, backend, sep, model, shifts, do_drums],
              outputs=[status, a_drums, a_bass, a_voc, a_gui, a_pia, a_oth,
                       full_out, g_out, v_out, p_out, b_out, d_out, o_out,
                       zip_out])

def _selftest() -> int:
    """Smoke test for frozen exe: tiny wav through the full pipeline.

    Returns:
        0 on success, 1 on failure.
    """
    import tempfile

    import numpy as np
    import soundfile as sf

    print("SELFTEST: generando wav de 6s...")
    py_env = {k: v for k, v in os.environ.items()
              if k.upper().startswith(("PYTHON", "PYI"))}
    print(f"SELFTEST: env PYTHON/PYI={py_env}")
    print(f"SELFTEST: PATH head={os.environ.get('PATH', '').split(os.pathsep)[:3]}")
    sr = 44100
    t = np.arange(sr * 6) / sr
    y = (0.4 * np.sin(2 * np.pi * 440 * t)
         + 0.2 * np.sin(2 * np.pi * 554.37 * t)).astype(np.float32)
    y[:: sr // 4] += 0.6  # transitorios a rejilla (120 BPM)
    tmp = Path(tempfile.mkdtemp(prefix="a2m_selftest_"))
    wav = tmp / "in.wav"
    sf.write(str(wav), np.stack([y, y], axis=1), sr)
    try:
        res = run_pipeline(str(wav), out_root=tmp / "outputs",
                           bpm=120.0, drums=True, sep="mejor")
        counts = res["notes_count"]
        full = Path(str(res["full_mix"]))
        ok = full.exists() and full.stat().st_size > 0
        print(f"SELFTEST: {'OK' if ok else 'FALLÓ'} notas={counts}")
        print(f"SELFTEST: full_mix={full}")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"SELFTEST: ERROR {exc}")
        return 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    demo_ui.launch(inbrowser=bool(getattr(sys, "frozen", False)))
