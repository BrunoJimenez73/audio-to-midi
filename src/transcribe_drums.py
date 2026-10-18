"""Batería: flujo espectral + clasificación por bandas (numpy puro)."""
from pathlib import Path

import numpy as np

from .dsp import stft

DRUM_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "hihat_closed": 42,
    "hihat_open": 46,
    "tom_low": 43,
    "tom_mid": 47,
    "tom_high": 50,
    "crash": 49,
    "ride": 51,
}
SR = 22050
N_FFT = 2048
HOP = 256
BANDS = (200.0, 800.0, 3000.0)


def _band_energy(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Energy in low/lomid/mid/high bands for one frame.

    Args:
        mag: Magnitude spectrum. freqs: Bin frequencies.

    Returns:
        Array of 4 band energies.
    """
    edges = (0.0,) + BANDS + (float(freqs[-1]),)
    out = np.zeros(4)
    for b in range(4):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        out[b] = float((mag[sel] ** 2).sum())
    return out


def _classify(bands: np.ndarray, sustain: float) -> int:
    """Map band energies + sustain to a GM drum pitch.

    Args:
        bands: 4 band energies. sustain: decay ratio 0..1.

    Returns:
        GM drum note number.
    """
    total = bands.sum() + 1e-12
    low, lomid, mid, high = bands / total
    if low > 0.15 or (low + lomid > 0.55 and low > 0.08):
        return DRUM_MAP["kick"]
    if lomid > 0.45:
        return DRUM_MAP["tom_low"]
    if high > 0.5:
        return DRUM_MAP["crash"] if sustain > 0.35 else DRUM_MAP["hihat_closed"]
    if high > 0.3:
        return DRUM_MAP["ride"] if sustain > 0.4 else DRUM_MAP["hihat_open"]
    if mid > 0.28 and low > 0.06:
        return DRUM_MAP["snare"]
    if lomid > 0.3:
        return DRUM_MAP["tom_mid"]
    return DRUM_MAP["snare"] if mid >= high else DRUM_MAP["hihat_closed"]


def transcribe_drums(drums_wav: str | Path) -> list[dict]:
    """Transcribe drums via spectral flux onsets + band cues.

    Args:
        drums_wav: Path to drums.wav.

    Returns:
        List of {pitch (GM), start, end, velocity}.
    """
    import soundfile as sf

    y, fs = sf.read(str(drums_wav), always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)
    if fs != SR and len(mono):
        n = int(len(mono) * SR / fs)
        mono = np.interp(np.linspace(0, len(mono), n),
                         np.arange(len(mono)), mono)
    if len(mono) < SR // 2:
        return []
    mag = np.abs(stft(mono, N_FFT, HOP))
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    thr = flux.mean() + 1.5 * flux.std()
    frame_dur = HOP / SR
    gap = max(1, int(0.07 / frame_dur))
    peaks = [i for i in range(1, len(flux) - 1)
             if flux[i] > thr and flux[i] >= flux[i - 1]
             and flux[i] > flux[i + 1]]
    picked = [p for j, p in enumerate(peaks)
              if j == 0 or p - peaks[j - 1] >= gap]
    fmax = float(flux.max()) if len(flux) else 1.0
    notes: list[dict] = []
    for p in picked:
        atk = min(p + 1, mag.shape[1] - 1)
        dec = min(p + 9, mag.shape[1] - 1)
        bands = _band_energy(mag[:, atk], freqs)
        e_atk = float((mag[:, atk] ** 2).sum()) + 1e-12
        e_dec = float((mag[:, dec] ** 2).sum())
        pitch = _classify(bands, e_dec / e_atk)
        vel = int(np.clip(50 + 60 * flux[p] / (fmax + 1e-12), 50, 120))
        t = p * frame_dur
        notes.append({"pitch": pitch, "start": t, "end": t + 0.15,
                      "velocity": vel})
    return notes


def drums_on_grid(notes: list[dict], bpm: float,
                  tol: float = 0.06) -> bool:
    """Check drum hits align to the beat grid (else probably no drums).

    Args:
        notes: Drum note list. bpm: Tempo. tol: Grid tolerance seconds.

    Returns:
        True if >= 45% of hits fall near 8th-note grid lines.
    """
    if len(notes) < 8:
        return False
    beat = 60.0 / bpm
    grid = beat / 2.0
    tol = min(tol, grid * 0.15)
    starts = np.array([n["start"] for n in notes])
    best = 0.0
    for ph in np.linspace(0, grid, 8, endpoint=False):
        frac = float(np.mean(np.abs((starts - ph + grid / 2) % grid
                                    - grid / 2) < tol))
        best = max(best, frac)
    return bool(best >= 0.45)
