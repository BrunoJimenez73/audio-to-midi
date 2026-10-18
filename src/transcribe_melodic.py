"""Transcripción melódica polifónica: picos espectrales + tracking."""
from pathlib import Path

import numpy as np

from .dsp import stft

RANGES: dict[str, tuple[int, int]] = {
    "piano": (21, 108),
    "guitar": (40, 88),
    "bass": (28, 64),
    "vocals": (40, 84),
    "other": (21, 108),
}
POLY: dict[str, int] = {
    "piano": 6, "guitar": 4, "bass": 1, "vocals": 1, "other": 4,
}
MIN_LEN: dict[str, float] = {
    "piano": 0.15, "guitar": 0.15, "bass": 0.25, "vocals": 0.2, "other": 0.15,
}
SR = 22050
N_FFT = 4096
HOP = 1024


def _frame_pitches(mag: np.ndarray, freqs: np.ndarray, lo: int,
                   hi: int, top_k: int) -> list[tuple[int, int]]:
    """Pick strongest pitches in one magnitude frame.

    Args:
        mag: Magnitude spectrum.
        freqs: Bin frequencies.
        lo: Min MIDI. hi: Max MIDI. top_k: Max pitches.

    Returns:
        List of (midi, velocity).
    """
    peak = float(mag[1:-1].max(initial=0.0))
    if peak <= 1e-9:
        return []
    thr = peak * 0.08
    idx = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
                   & (mag[1:-1] > thr))[0] + 1
    if len(idx) == 0:
        return []
    # Interpolación parabólica para afinar frecuencia.
    denom = mag[idx - 1] - 2 * mag[idx] + mag[idx + 1]
    shift = np.where(denom != 0, 0.5 * (mag[idx - 1] - mag[idx + 1]) / denom, 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    bin_w = freqs[1] - freqs[0]
    freq = freqs[idx] + shift * bin_w
    midi_f = 69 + 12 * np.log2(np.maximum(freq, 1e-6) / 440.0)
    midi = np.round(midi_f).astype(int)
    order = np.argsort(mag[idx])[::-1]
    seen: dict[int, tuple[float, int]] = {}
    for j in order:
        m = int(midi[j])
        if not (lo <= m <= hi):
            continue
        if any(abs(m - s) < 1 for s in seen):
            continue
        # Supresión de armónicos: si está la octava inferior más fuerte,
        # el pico superior suele ser su 2º armónico, no una nota real.
        if any((m - s) == 12 and seen[s][0] > 2 * float(mag[idx[j]])
               for s in seen):
            continue
        db = 20 * np.log10(mag[idx[j]] / peak + 1e-9)
        vel = int(np.clip(50 + (db + 32) * 60 / 32, 40, 110))
        seen[m] = (float(mag[idx[j]]), vel)
        if len(seen) >= top_k:
            break
    return [(m, v) for m, (_, v) in seen.items()]


def transcribe_melodic(
    stem_wav: str | Path,
    instrument: str = "piano",
    min_note_len: float | None = None,
    backend: str = "auto",
) -> list[dict]:
    """Transcribe polyphonic stem via spectral peaks + tracking.

    Args:
        stem_wav: Path to stem wav.
        instrument: Instrument name (range + polyphony cap).
        min_note_len: Minimum note duration; defaults per instrument.

    Returns:
        List of {pitch, start, end, velocity}.
    """
    import os

    resolved = backend
    if resolved == "auto":
        resolved = os.environ.get("AUDIO2MIDI_MELODIC_BACKEND", "auto")
    if resolved in ("basicpitch", "basic_pitch", "bp", "auto"):
        try:
            from .transcribe_basicpitch import (bp_available,
                                                transcribe_basicpitch)

            if resolved == "auto" and not bp_available():
                raise RuntimeError("bp no disponible")
            bp_notes = transcribe_basicpitch(stem_wav, instrument)
            if bp_notes:
                return bp_notes
        except Exception:  # noqa: BLE001
            pass
    import soundfile as sf

    lo, hi = RANGES.get(instrument, (21, 108))
    top_k = POLY.get(instrument, 4)
    if min_note_len is None:
        min_note_len = MIN_LEN.get(instrument, 0.15)
    y, fs = sf.read(str(stem_wav), always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)
    if fs != SR and len(mono):
        n = int(len(mono) * SR / fs)
        mono = np.interp(np.linspace(0, len(mono), n),
                         np.arange(len(mono)), mono)
    if len(mono) < SR // 4:
        return []
    mag = np.abs(stft(mono, N_FFT, HOP))
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    frame_dur = HOP / SR
    min_frames = max(2, int(min_note_len / frame_dur))
    active: dict[int, list] = {}
    notes: list[dict] = []

    def close(pitch: int, last: int) -> None:
        s, _, v = active.pop(pitch)
        if last - s + 1 >= min_frames:
            notes.append({"pitch": pitch, "start": s * frame_dur,
                          "end": (last + 1) * frame_dur, "velocity": v})

    for i in range(mag.shape[1]):
        cur = dict(_frame_pitches(mag[:, i], freqs, lo, hi, top_k))
        for p in [p for p in active if p not in cur or i - active[p][1] > 2]:
            close(p, active[p][1])
        for p, v in cur.items():
            if p in active:
                active[p][1] = i
                active[p][2] = max(active[p][2], v)
            else:
                active[p] = [i, i, v]
    for p, (_, last, _) in list(active.items()):
        close(p, last)
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    merged: list[dict] = []
    for n in notes:
        if (merged and merged[-1]["pitch"] == n["pitch"]
                and n["start"] - merged[-1]["end"] < 0.1):
            merged[-1]["end"] = n["end"]
        else:
            merged.append(n)
    return merged
