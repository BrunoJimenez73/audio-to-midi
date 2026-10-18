"""HPSS tipo Fitzgerald con numpy (mediana por trozos)."""
import numpy as np

from .dsp import istft, stft


def _medfilt(a: np.ndarray, k: int, axis: int, chunk: int = 400) -> np.ndarray:
    """Median filter along axis, chunked to bound memory.

    Args:
        a: 2D array.
        k: Odd kernel size.
        axis: 0 or 1.
        chunk: Frames per chunk.

    Returns:
        Filtered array, same shape.
    """
    if k % 2 == 0:
        k += 1
    pad = k // 2
    if axis == 1:
        b = np.pad(a, ((0, 0), (pad, pad)), mode="reflect")
        out = np.empty_like(a)
        for s in range(0, a.shape[1], chunk):
            e = min(s + chunk, a.shape[1])
            seg = b[:, s:e + 2 * pad]
            w = np.lib.stride_tricks.sliding_window_view(seg, k, axis=1)
            out[:, s:e] = np.median(w, axis=-1)
        return out
    b = np.pad(a, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(a)
    for s in range(0, a.shape[0], chunk):
        e = min(s + chunk, a.shape[0])
        seg = b[s:e + 2 * pad, :]
        w = np.lib.stride_tricks.sliding_window_view(seg, k, axis=0)
        out[s:e, :] = np.median(w, axis=-1)
    return out


def hpss(y: np.ndarray, sr: int = 22050, n_fft: int = 2048,
         hop: int = 1024, power: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Split mono audio into harmonic and percussive parts.

    Args:
        y: Mono samples.
        sr: Sample rate (informativo).
        n_fft: FFT size.
        hop: Hop size.
        power: Wiener exponent.

    Returns:
        Tuple (harmonic, percussive) same length as input.
    """
    spec = stft(y, n_fft, hop)
    mag = np.abs(spec).astype(np.float64)
    harm_enh = _medfilt(mag, 17, axis=1)
    perc_enh = _medfilt(mag, 17, axis=0)
    denom = harm_enh ** power + perc_enh ** power + 1e-10
    mask_h = (harm_enh ** power / denom).astype(np.float32)
    phase = np.exp(1j * np.angle(spec)).astype(np.complex64)
    mag32 = mag.astype(np.float32)
    harm = istft(mag32 * mask_h * phase, n_fft, hop, len(y))
    perc = istft(mag32 * (1.0 - mask_h) * phase, n_fft, hop, len(y))
    return harm, perc
