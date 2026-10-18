"""UI web local: URL YouTube o mp3 -> stems + MIDI multicanal."""
import zipfile
from pathlib import Path

import gradio as gr

from src.pipeline import run_pipeline

ORDER = ("drums", "bass", "vocals", "guitar", "piano", "other")


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


GRID_MAP = {"1/8": 0.5, "1/16": 0.25, "1/32": 0.125}


def process(user_url: str, user_file: str | None, bpm: float,
            quant: str = "1/16",
            progress: gr.Progress = gr.Progress()) -> tuple:
    """Run pipeline and return status + artifacts.

    Args:
        user_url: URL text.
        user_file: Uploaded audio path.
        bpm: Forced tempo (0 = auto).
        progress: Gradio progress tracker.

    Returns:
        Tuple (status, 6 stems audio, full_mix file, per-track files).
    """
    try:
        src = _pick(user_url, user_file)
    except ValueError as exc:
        return (f"Error: {exc}",) + (None,) * 9
    try:
        grid = GRID_MAP.get(quant, 0.25)
        res = run_pipeline(src, progress=progress, bpm=float(bpm or 0),
                           grid_beats=grid)
    except Exception as exc:  # noqa: BLE001
        return (f"Error: {exc}",) + (None,) * 9
    stems = res["stems"]
    assert isinstance(stems, dict)
    audios = [str(stems[s]) if s in stems else None for s in ORDER]
    full = str(res["full_mix"])
    midis = res["midis"]
    assert isinstance(midis, dict)
    piano_mid = str(midis["piano"]) if "piano" in midis else None
    # ZIP con todos los MIDI de la corrida.
    run_dir = Path(str(res["run_dir"]))
    midi_dir = run_dir / "midi"
    zip_path = run_dir / "midis.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for mid in sorted(midi_dir.glob("*.mid")):
                zf.write(mid, arcname=mid.name)
    except Exception:  # noqa: BLE001
        zip_path = None
    counts = res["notes_count"]
    msg = f"OK en {res['run_dir']}. Notas: {counts}."
    return (msg, *audios, full, piano_mid,
            str(zip_path) if zip_path else None)


with gr.Blocks(title="audio-to-midi") as demo_ui:
    gr.Markdown(
        "# audio-to-midi\n"
        "Sube un mp3/wav o pega URL de YouTube. "
        "Genera 6 stems + `full_mix.mid` multicanal.\n\n"
        "*Sin Demucs usa separación ligera HPSS; con Demucs+GPU, calidad máxima.*"
    )
    with gr.Row():
        url = gr.Textbox(label="URL YouTube (o ruta local)")
        f = gr.File(label="o sube mp3/wav", file_types=["audio"])
    with gr.Row():
        bpm = gr.Number(label="BPM (0 = auto)", value=0, minimum=0, maximum=220)
        quant = gr.Dropdown(choices=["1/8", "1/16", "1/32"], value="1/16",
                            label="Cuantización")
    btn = gr.Button("Transcribir", variant="primary")
    status = gr.Textbox(label="Estado")
    with gr.Row():
        a_drums = gr.Audio(label="drums")
        a_bass = gr.Audio(label="bass")
        a_voc = gr.Audio(label="vocals")
    with gr.Row():
        a_gui = gr.Audio(label="guitar")
        a_pia = gr.Audio(label="piano")
        a_oth = gr.Audio(label="other")
    with gr.Row():
        full_out = gr.File(label="full_mix.mid")
        piano_out = gr.File(label="piano.mid")
        zip_out = gr.File(label="midis.zip")
    btn.click(process, inputs=[url, f, bpm, quant],
              outputs=[status, a_drums, a_bass, a_voc, a_gui,
                       a_pia, a_oth, full_out, piano_out, zip_out])

if __name__ == "__main__":
    demo_ui.launch()
