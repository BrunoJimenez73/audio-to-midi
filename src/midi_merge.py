"""Merge de tracks a full_mix.mid multicanal con tempo real."""
from pathlib import Path

import mido

from .utils import MIDI_MAP

TICKS = 480


def sec_to_ticks(sec: float, bpm: float) -> int:
    """Convert seconds to MIDI ticks at given tempo.

    Args:
        sec: Time in seconds. bpm: Tempo.

    Returns:
        Tick count (480 ticks per beat).
    """
    return int(round(sec * TICKS * bpm / 60.0))


def notes_to_track(notes: list[dict], channel: int, program: int,
                   bpm: float = 120.0) -> mido.MidiTrack:
    """Convert note dicts (seconds) to a MIDI track.

    Args:
        notes: List of {pitch, start, end, velocity} in seconds.
        channel: MIDI channel (9 = drums).
        program: GM program (ignored for drums).
        bpm: Tempo for seconds-to-beats conversion.

    Returns:
        Populated MidiTrack.
    """
    track = mido.MidiTrack()
    if channel != 9:
        track.append(mido.Message("program_change", program=program,
                                  channel=channel, time=0))
    events: list[tuple[int, str, dict]] = []
    for n in notes:
        s = sec_to_ticks(float(n["start"]), bpm)
        e = sec_to_ticks(float(n["end"]), bpm)
        if e <= s:
            e = s + sec_to_ticks(0.05, bpm)
        events.append((s, "on", n))
        events.append((e, "off", n))
    events.sort(key=lambda x: x[0])
    last = 0
    for tick, kind, n in events:
        delta = tick - last
        last = tick
        if kind == "on":
            track.append(mido.Message("note_on", note=int(n["pitch"]),
                                      velocity=int(n.get("velocity", 90)),
                                      channel=channel, time=delta))
        else:
            track.append(mido.Message("note_off", note=int(n["pitch"]),
                                      velocity=0, channel=channel, time=delta))
    return track


def conductor_track(bpm: float) -> mido.MidiTrack:
    """Build tempo/time-signature conductor track.

    Args:
        bpm: Tempo.

    Returns:
        MidiTrack with time_signature + set_tempo.
    """
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=4,
                                  denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo",
                                  tempo=mido.bpm2tempo(bpm), time=0))
    return track


def quantize_notes(notes: list[dict], bpm: float,
                   grid_beats: float = 0.25) -> list[dict]:
    """Snap note starts/ends to a beat grid.

    Args:
        notes: Note list in seconds.
        bpm: Tempo.
        grid_beats: Grid size in beats (0.25 = 16ths).

    Returns:
        New quantized note list.
    """
    grid_sec = grid_beats * 60.0 / bpm
    out: list[dict] = []
    for n in notes:
        s = round(float(n["start"]) / grid_sec) * grid_sec
        e = round(float(n["end"]) / grid_sec) * grid_sec
        if e <= s:
            e = s + grid_sec
        out.append({"pitch": n["pitch"], "start": s, "end": e,
                    "velocity": n.get("velocity", 90)})
    return out


def merge_to_midi(tracks_notes: dict[str, list[dict]],
                  out_path: str | Path, bpm: float = 120.0) -> Path:
    """Merge per-instrument notes into one multichannel MIDI file.

    Args:
        tracks_notes: Mapping instrument -> note list (seconds).
        out_path: Destination .mid path.
        bpm: Tempo written to conductor track.

    Returns:
        Path to written MIDI file.
    """
    mid = mido.MidiFile(ticks_per_beat=TICKS)
    mid.tracks.append(conductor_track(bpm))
    for inst, notes in tracks_notes.items():
        channel, program = MIDI_MAP.get(inst, (5, 0))
        mid.tracks.append(notes_to_track(notes, channel, program, bpm))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out))
    return out
