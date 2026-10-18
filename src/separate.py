"""Separación: Demucs htdemucs_6s (venv 3.11 + CUDA) o fallback numpy."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .utils import STEMS, ensure_dir


def _demucs_python() -> str | None:
    """Locate a Python with Demucs installed.

    Args:
        None.

    Returns:
        Path to venv python with Demucs, or None.
    """
    here = Path(__file__).resolve().parent.parent
    venv_py = here / ".venv311" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    try:
        import demucs  # noqa: F401
        return sys.executable
    except ImportError:
        return None


def _run_demucs(demux_py: str, audio: Path, dest: Path,
                model: str) -> dict[str, Path]:
    """Run Demucs via subprocess (cuda, fallback cpu).

    Args:
        demux_py: Python executable with Demucs.
        audio: Input wav. dest: Output dir. model: Demucs model.

    Returns:
        Stem mapping, or empty dict on failure.
    """
    tmp = dest / "_demucs_tmp"
    for device in ("cuda", "cpu"):
        cmd = [demux_py, "-m", "demucs.separate", "-n", model,
               "-d", device, "--out", str(tmp), str(audio)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            break
    else:
        return {}
    found = list(tmp.rglob("*.wav"))
    result: dict[str, Path] = {}
    for stem in STEMS:
        match = next((f for f in found if f.stem == stem), None)
        if match is not None:
            target = dest / f"{stem}.wav"
            shutil.copy(match, target)
            result[stem] = target
    shutil.rmtree(tmp, ignore_errors=True)
    return result


def separate_demucs(
    audio_path: str | Path,
    out_dir: str | Path,
    model: str = "htdemucs_6s",
) -> dict[str, Path]:
    """Separate audio into 6 stems.

    Uses Demucs htdemucs_6s on CUDA when the 3.11 venv exists,
    else the numpy fallback (HPSS + voice mask + band stems).

    Args:
        audio_path: Input wav path.
        out_dir: Directory for stems.
        model: Demucs model name.

    Returns:
        Mapping stem name -> wav path.
    """
    dest = ensure_dir(out_dir)
    if os.environ.get("AUDIO2MIDI_NO_DEMUCS"):
        return _fallback_hpss(Path(audio_path), dest)
    demux_py = _demucs_python()
    if demux_py is not None:
        got = _run_demucs(demux_py, Path(audio_path), dest, model)
        if len(got) >= 4:
            print(f"Demucs {model} OK ({len(got)} stems).")
            return got
        print("Demucs falló, usando fallback numpy.")
    return _fallback_hpss(Path(audio_path), dest)


def _transient_part(y: np.ndarray, sr: int) -> np.ndarray:
    """Emphasize attack frames (piano/pluck-like) via flux gating.

    Args:
        y: Mono samples. sr: Sample rate.

    Returns:
        Audio with steady frames attenuated 12 dB.
    """
    from .dsp import istft, stft

    n_fft, hop = 2048, 512
    spec = stft(y, n_fft, hop)
    mag = np.abs(spec).astype(np.float64)
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    flux = np.concatenate([[flux[0]], flux])
    thr = float(np.median(flux) + np.std(flux))
    gate = np.clip(flux / (thr + 1e-9) - 0.5, 0.06, 1.0)
    # Suaviza la puerta para no trocear.
    k = np.ones(5) / 5
    gate = np.convolve(np.pad(gate, 2, mode="edge"), k, mode="valid")
    return istft((spec * gate[None, :].astype(spec.dtype)), n_fft, hop, len(y))


def _fallback_hpss(audio: Path, dest: Path) -> dict[str, Path]:
    """Stems por contenido: voz (máscara) + backing + transitorios.

    Pensado para mezclas centradas (cantautor): la voz se extrae por
    vibrato+contorno, la guitarra es el backing menos la voz, el piano
    sale de los ataques y el bajo de los graves.

    Args:
        audio: Input wav (estéreo si es posible).
        dest: Output dir.

    Returns:
        Stem mapping (6 stems).
    """
    import soundfile as sf

    from .dsp import band_filter
    from .hpss import hpss
    from .voice import split_voice

    dest = ensure_dir(dest)
    y, sr = sf.read(str(audio), always_2d=True)
    if y.shape[1] == 1:
        left = right = y[:, 0].astype(np.float32)
    else:
        left, right = y[:, 0].astype(np.float32), y[:, 1].astype(np.float32)
    mid = ((left + right) / 2.0).astype(np.float32)
    side = ((left - right) / 2.0).astype(np.float32)
    harm, perc = hpss(mid, sr)
    # Voz por máscara solo si separa algo real (ni vacía ni todo).
    voice, backing, vf = split_voice(harm, sr, side)
    hrms = float(np.sqrt(np.mean(harm.astype(np.float64) ** 2))) + 1e-9
    vrms = float(np.sqrt(np.mean(voice.astype(np.float64) ** 2)))
    use_mask = 0.05 <= vf <= 0.995 and 0.08 <= vrms / hrms <= 0.7
    print(f"voice mask: vf={vf:.2f} energy={vrms / hrms:.2f} -> "
          f"{'mask' if use_mask else 'harmonic fallback'}")
    if use_mask:
        melodic = backing
        vocals_src = voice
    else:
        melodic = harm
        vocals_src = harm
    parts: dict[str, np.ndarray] = {
        "drums": perc,
        "bass": band_filter(harm, sr, 0.0, 250.0),
        "guitar": band_filter(melodic, sr, 120.0, 6500.0),
        "piano": band_filter(_transient_part(melodic, sr), sr, 150.0, 6000.0),
        "vocals": band_filter(vocals_src, sr, 150.0, 5000.0),
        "other": band_filter(melodic, sr, 3000.0, 0.0),
    }
    dest = ensure_dir(dest)
    result: dict[str, Path] = {}
    for stem in STEMS:
        target = dest / f"{stem}.wav"
        sf.write(str(target), parts[stem], sr)
        result[stem] = target
    return result


def expected_stems(out_dir: str | Path) -> dict[str, Path]:
    """Return expected stem paths for a directory.

    Args:
        out_dir: Stems directory.

    Returns:
        Mapping stem name -> expected wav path.
    """
    base = Path(out_dir)
    return {s: base / f"{s}.wav" for s in STEMS}
