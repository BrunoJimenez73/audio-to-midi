"""Estimación de tempo (BPM) con numpy puro."""
from pathlib import Path

import numpy as np

from .dsp import stft

SR = 22050


def estimate_bpm(audio_path: str | Path, lo: float = 60.0,
                 hi: float = 200.0) -> float:
    """Estimate constant tempo via onset-envelope autocorrelation.

    Args:
        audio_path: Audio file (mix or drums stem works best).
        lo: Min BPM. hi: Max BPM.

    Returns:
        Estimated BPM rounded to 1 decimal.
    """
    import soundfile as sf

    y, fs = sf.read(str(audio_path), always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)
    if fs != SR and len(mono):
        n = int(len(mono) * SR / fs)
        mono = np.interp(np.linspace(0, len(mono), n),
                         np.arange(len(mono)), mono)
    mag = np.abs(stft(mono, 1024, 256))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    flux = flux - flux.mean()
    if np.abs(flux).max() < 1e-9:
        return 120.0
    flux = flux / np.abs(flux).max()
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    fps = SR / 256
    scores: dict[float, float] = {}
    bpm = lo
    while bpm <= hi:
        lag = int(round(fps * 60.0 / bpm))
        if 0 < lag < len(ac):
            # Refino parabólico del pico.
            y0 = ac[max(0, lag - 1)]
            y1 = ac[lag]
            y2 = ac[min(len(ac) - 1, lag + 1)]
            d = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / d if d != 0 else 0.0
            shift = float(np.clip(shift, -1.0, 1.0))
            scores[bpm] = float(y1 - 0.25 * (y0 - y2) * shift)
        bpm += 1.0
    if not scores:
        return 120.0
    best = max(scores, key=lambda b: scores[b])
    peak = scores[best]
    def prior(b: float) -> float:
        return float(np.exp(-0.5 * (np.log2(b / 100.0) / 0.55) ** 2))
    chosen = best
    # Si el medio tiempo explica casi igual de bien el ritmo y suena más
    # central, el feel real suele ser el lento (baladas a 75 vs 150).
    half = round(best / 2)
    if (half >= lo and scores.get(float(half), -1) > 0.75 * peak
            and prior(float(half)) > prior(best)):
        chosen = float(half)
    lag = int(round(fps * 60.0 / chosen))
    y0 = ac[max(0, lag - 1)]
    y1 = ac[lag]
    y2 = ac[min(len(ac) - 1, lag + 1)]
    d = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / d if d != 0 else 0.0
    return round(float(fps * 60.0 / (lag + float(np.clip(shift, -1, 1)))), 1)
