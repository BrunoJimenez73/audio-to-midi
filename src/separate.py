"""Separacion: rapido (Demucs htdemucs_6s) o mejor (jerarquia UVR+Demucs)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .utils import STEMS, ensure_dir


def _demucs_python() -> str | None:
    """Locate the project venv python (Demucs, audio-separator, torch).

    Args:
        None.

    Returns:
        Path to venv python with heavy deps, or None.
    """
    from .utils import find_up

    found = find_up(Path(".venv311") / "Scripts" / "python.exe")
    if found is not None:
        return str(found)
    try:
        import demucs  # noqa: F401
        return sys.executable
    except ImportError:
        return None


def _models_dir() -> Path:
    """Resolve the audio-separator model cache directory.

    Order: env AUDIO2MIDI_MODELS_DIR, first existing ``models`` dir from the
    exe/cwd upwards, then ``<exe or cwd>/models`` (created on first use).

    Returns:
        Models directory (may not exist yet).
    """
    env = os.environ.get("AUDIO2MIDI_MODELS_DIR")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
    else:
        start = Path.cwd()
    for cand in (start, *start.parents):
        models = cand / "models"
        if models.is_dir():
            return models
    return start / "models"


def _run_separator_mejor(
    mix: Path, dest: Path, models_dir: Path
) -> dict[str, Path]:
    """Run the hierarchical \"mejor\" worker (Roformer + Demucs x2).

    Args:
        mix: Input mixture wav.
        dest: Output stems directory.
        models_dir: Cache for downloaded UVR models.

    Returns:
        Mapping of the 6 stems to wav paths, or empty dict on failure.
    """
    py = _demucs_python()
    from .utils import bundled_path

    worker = bundled_path(Path("scripts") / "separator_worker.py")
    if py is None or worker is None:
        return {}
    models_dir.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(worker), str(mix), str(dest),
           "--models-dir", str(models_dir)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace",
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print("Separacion Mejor timeout (1800s), usando modo rapido.")
        return {}
    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("@@SEPARATOR@@"):
            try:
                payload = json.loads(line[len("@@SEPARATOR@@"):])
            except json.JSONDecodeError:
                payload = None
    if payload is None or payload.get("status") != "ok":
        detail = payload.get("error") if payload else None
        if detail:
            print(f"Separacion Mejor fallo: {detail}")
        else:
            # stderr mezcla barras de tqdm (contienen 'it/s'); las quitamos.
            clean = [ln for ln in (proc.stderr or "").splitlines()
                     if "it/s" not in ln and "%|" not in ln]
            tail = (clean or ["sin salida"]) [-3:]
            print(f"Separacion Mejor fallo (rc={proc.returncode}): "
                  f"{' | '.join(tail)}")
        return {}
    stems: dict[str, Path] = {}
    for name, raw in payload.get("stems", {}).items():
        path = Path(str(raw))
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            stems[name] = path
    print(f"Separacion Mejor OK: {payload.get('elapsed')}s "
          f"(voz={payload.get('vocals_source')}, "
          f"pasadas={payload.get('stages')})")
    return stems


def _run_demucs(demux_py: str, audio: Path, dest: Path,
                model: str, shifts: int = 1) -> dict[str, Path]:
    """Run Demucs via subprocess (cuda, fallback cpu).

    Args:
        demux_py: Python executable with Demucs.
        audio: Input wav. dest: Output dir. model: Demucs model.
        shifts: Random shifts averaged (higher = better, slower).

    Returns:
        Stem mapping, or empty dict on failure.
    """
    tmp = dest / "_demucs_tmp"
    for device in ("cuda", "cpu"):
        cmd = [demux_py, "-m", "demucs.separate", "-n", model,
               "-d", device, "--shifts", str(max(1, int(shifts))),
               "--out", str(tmp), str(audio)]
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
    shifts: int = 1,
    sep: str = "mejor",
) -> dict[str, Path]:
    """Separate audio into 6 stems.

    ``sep=\"mejor\"`` runs the hierarchical UVR pipeline (Roformer vocals +
    htdemucs_ft + htdemucs_6s on the residual) via separator_worker in the
    3.11 venv; ``sep=\"rapido\"`` runs a single Demucs pass. Both degrade to
    the numpy HPSS fallback when the heavy tools are unavailable.

    Args:
        audio_path: Input wav path.
        out_dir: Directory for stems.
        model: Demucs model name (modo rapido).
        shifts: Random shifts averaged (higher = better, slower).
        sep: Separation mode: mejor | rapido.

    Returns:
        Mapping stem name -> wav path.
    """
    dest = ensure_dir(out_dir)
    if os.environ.get("AUDIO2MIDI_NO_DEMUCS"):
        return _fallback_hpss(Path(audio_path), dest)
    if sep == "mejor" and not os.environ.get("AUDIO2MIDI_NO_UVR"):
        got = _run_separator_mejor(Path(audio_path), dest, _models_dir())
        if len(got) == len(STEMS):
            print(f"Separacion Mejor OK ({len(got)} stems).")
            return got
        if got:
            print("Separacion Mejor incompleta, usando modo rapido.")
    demux_py = _demucs_python()
    if demux_py is not None:
        got = _run_demucs(demux_py, Path(audio_path), dest, model, shifts)
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
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
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
