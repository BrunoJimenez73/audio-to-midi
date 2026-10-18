"""Orquestador: input -> mix.wav -> stems -> midis -> full_mix.mid."""
import time
from pathlib import Path

import numpy as np

from .assign import assign_notes
from .ingest import prepare_input
from .midi_merge import conductor_track, merge_to_midi, notes_to_track, quantize_notes
from .separate import separate_demucs
from .tempo import estimate_bpm
from .transcribe_drums import drums_on_grid, transcribe_drums
from .transcribe_melodic import transcribe_melodic
from .utils import MIDI_MAP, ensure_dir

MELODIC = ("piano", "guitar", "bass", "vocals", "other")


def _stems_are_stereodemucs(stems: dict[str, Path]) -> bool:
    """Detect whether stems came from Demucs (stereo) or numpy fallback.

    Args:
        stems: Stem mapping from separate_demucs.

    Returns:
        True if the vocals stem is stereo (Demucs output).
    """
    wav = stems.get("vocals")
    if wav is None or not wav.exists():
        return False
    try:
        import soundfile as sf

        return sf.info(str(wav)).channels == 2
    except Exception:  # noqa: BLE001
        return False


def run_pipeline(
    user_input: str,
    out_root: str | Path = "outputs",
    progress: object = None,
    bpm: float = 0.0,
    grid_beats: float = 0.25,
) -> dict[str, object]:
    """Run full pipeline and return artifact paths.

    Args:
        user_input: YouTube URL, local audio path, or uploaded file path.
        out_root: Base outputs directory.
        progress: Optional gradio Progress-like callable.
        bpm: Forced tempo (0 = auto-estimate).

    Returns:
        Dict with keys: run_dir, mix, stems, midis, full_mix, notes_count, bpm.
    """
    def say(frac: float, msg: str) -> None:
        if progress is not None:
            try:
                progress(frac, desc=msg)  # type: ignore
            except TypeError:
                pass

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(out_root) / run_id)
    say(0.05, "Preparando audio...")
    mix = prepare_input(user_input, run_dir / "work")
    say(0.25, "Separando stems...")
    stems = separate_demucs(mix, run_dir / "stems")
    stems_real = _stems_are_stereodemucs(stems)
    say(0.5, "Transcribiendo melodías...")
    pools: dict[str, list[dict]] = {}
    bp_done = False
    if stems_real:
        try:
            from .transcribe_basicpitch import bp_available, transcribe_basicpitch_batch

            if bp_available():
                wav_map = {i: stems[i] for i in MELODIC
                           if stems.get(i) is not None and stems[i].exists()}
                pools = transcribe_basicpitch_batch(wav_map)
                bp_done = True
                say(0.8, "Basic Pitch OK")
        except Exception:
            bp_done = False
    if not bp_done:
        for i, inst in enumerate(MELODIC):
            say(0.5 + 0.07 * i, f"Transcribiendo {inst}...")
            wav = stems.get(inst)
            if wav is not None and wav.exists():
                try:
                    pools[inst] = transcribe_melodic(wav, inst)
                except Exception:
                    pools[inst] = []
    for inst in MELODIC:
        pools.setdefault(inst, [])
    say(0.85, "Transcribiendo batería...")
    drums_wav = stems.get("drums")
    drum_list: list[dict] = []
    if drums_wav is not None and drums_wav.exists():
        try:
            import soundfile as _sf

            d, _ = _sf.read(str(drums_wav), always_2d=True)
            m, _ = _sf.read(str(mix), always_2d=True)
            perc_r = float(np.sqrt(np.mean(d.astype(np.float64) ** 2)))
            mix_r = float(np.sqrt(np.mean(m.astype(np.float64) ** 2))) + 1e-9
            has_drums = perc_r > 0.12 * mix_r
        except Exception:
            has_drums = True
        if has_drums:
            try:
                drum_notes = transcribe_drums(drums_wav)
                drum_list = [
                    {"pitch": n["pitch"], "start": n["start"],
                     "end": n["end"], "velocity": n.get("velocity", 100)}
                    for n in drum_notes
                ]
            except Exception:
                drum_list = []
    if stems_real:
        # Demucs ya separó por instrumento: confiamos en los stems.
        tracks_notes = {**pools, "drums": drum_list}
    else:
        say(0.90, "Repartiendo instrumentos...")
        try:
            import soundfile as sf

            y, sr = sf.read(str(mix), always_2d=True)
            if y.shape[1] == 1:
                left = right = y[:, 0].astype(np.float32)
            else:
                left = y[:, 0].astype(np.float32)
                right = y[:, 1].astype(np.float32)
            mid_ch = (left + right) / 2.0
            side_ch = (left - right) / 2.0
            d, _ = sf.read(str(stems["drums"]), always_2d=True)
            drums_mono = d.mean(axis=1).astype(np.float32)
            n = min(len(mid_ch), len(drums_mono))
            harm = (mid_ch[:n] - drums_mono[:n]).astype(np.float32)
            v, _ = sf.read(str(stems["vocals"]), always_2d=True)
            vox_mono = v.mean(axis=1).astype(np.float32)
            tracks_notes = assign_notes(
                {k: pools[k] for k in ("piano", "guitar", "vocals", "other")},
                harm, mid_ch.astype(np.float32), side_ch.astype(np.float32),
                int(sr), vox=vox_mono)
            tracks_notes["bass"] = pools["bass"]
            tracks_notes["drums"] = drum_list
        except Exception:
            tracks_notes = {**pools, "drums": drum_list}
    say(0.93, "Generando MIDI...")
    if bpm and bpm > 0:
        tempo = float(bpm)
    else:
        try:
            tempo = estimate_bpm(mix)
        except Exception:
            tempo = 120.0
    if drum_list and not drums_on_grid(drum_list, tempo):
        drum_list = []  # sin batería real: hits de guitarra, no batería
        tracks_notes["drums"] = []
    say(0.94, f"Tempo {tempo} BPM, cuantizando...")
    for inst in tracks_notes:
        tracks_notes[inst] = quantize_notes(tracks_notes[inst], tempo,
                                           grid_beats=grid_beats)
    midi_dir = ensure_dir(run_dir / "midi")
    midis: dict[str, Path] = {}
    import mido as _mido
    for inst, notes in tracks_notes.items():
        ch, prog = MIDI_MAP.get(inst, (5, 0))
        mid = _mido.MidiFile(ticks_per_beat=480)
        mid.tracks.append(conductor_track(tempo))
        mid.tracks.append(notes_to_track(notes, ch, prog, tempo))
        p = midi_dir / f"{inst}.mid"
        mid.save(str(p))
        midis[inst] = p
    full = merge_to_midi(tracks_notes, midi_dir / "full_mix.mid", tempo)
    say(1.0, f"Listo ({tempo} BPM)")
    counts = {k: len(v) for k, v in tracks_notes.items()}
    return {"run_dir": run_dir, "mix": mix, "stems": stems,
            "midis": midis, "full_mix": full, "notes_count": counts,
            "bpm": tempo}
