"""DSP con numpy puro (scipy/librosa rotos en Python 3.14)."""
import numpy as np


def stft(y: np.ndarray, n_fft: int = 2048, hop: int = 1024) -> np.ndarray:
    """Short-time Fourier transform (one-sided).

    Args:
        y: Mono samples.
        n_fft: Window size.
        hop: Hop size.

    Returns:
        Complex spectrogram (bins x frames).
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + (len(y) - n_fft + hop - 1) // hop
    need = (n_frames - 1) * hop + n_fft
    if need > len(y):
        y = np.pad(y, (0, need - len(y)))
        n_frames = 1 + (len(y) - n_fft) // hop
    shape = (n_frames, n_fft)
    strides = (y.strides[0] * hop, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape, strides).copy()
    window = np.hanning(n_fft + 1)[:-1]
    return np.fft.rfft(frames * window, axis=1).T


def istft(spec: np.ndarray, n_fft: int = 2048, hop: int = 1024,
          length: int = 0) -> np.ndarray:
    """Inverse STFT via overlap-add with window-sum normalization.

    Args:
        spec: Complex spectrogram (bins x frames).
        n_fft: Window size used in stft.
        hop: Hop size used in stft.
        length: Trim output to this many samples.

    Returns:
        Reconstructed mono samples.
    """
    window = np.hanning(n_fft + 1)[:-1]
    n_frames = spec.shape[1]
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    for i in range(n_frames):
        frame = np.fft.irfft(spec[:, i], n=n_fft) * window
        out[i * hop:i * hop + n_fft] += frame
        wsum[i * hop:i * hop + n_fft] += window ** 2
    nz = wsum > 1e-8
    out[nz] /= wsum[nz]
    return out[:length].astype(np.float32) if length else out.astype(np.float32)


def band_filter(y: np.ndarray, sr: int, lo_hz: float = 0.0,
                hi_hz: float = 0.0, n_fft: int = 2048,
                hop: int = 1024) -> np.ndarray:
    """Brick-wall bandpass via STFT masking (zero-phase).

    Args:
        y: Mono samples. sr: Sample rate.
        lo_hz: Low cutoff (0 = none). hi_hz: High cutoff (0 = none).
        n_fft: FFT size. hop: Hop size.

    Returns:
        Filtered samples, same length as input.
    """
    spec = stft(y, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = np.ones(spec.shape[0])
    fade = 40.0  # Hz de transición para evitar ringing
    if lo_hz > 0:
        mask = np.clip((freqs - lo_hz + fade) / (2 * fade), 0.0, 1.0)
        mask = mask * mask * (3 - 2 * mask)  # smoothstep
    if hi_hz > 0:
        hi_m = np.clip((hi_hz + fade - freqs) / (2 * fade), 0.0, 1.0)
        mask = mask * hi_m * hi_m * (3 - 2 * hi_m)
    return istft(spec * mask[:, None].astype(spec.dtype),
                 n_fft, hop, len(y))
