"""Transcripción melódica con Basic Pitch (ML), con fallback numpy."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .transcribe_melodic import RANGES

# Tope de frecuencia por instrumento: Basic Pitch tiende a marcar
# armónicos como notas; acotamos para quedarnos con fundamentales.
CAPS = {"piano": 108, "guitar": 79, "bass": 55, "vocals": 81, "other": 96}


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def bp_python() -> str | None:
    """Locate a Python with basic-pitch installed.

    Args:
        None.

    Returns:
        Path to python, or None.
    """
    from .utils import find_up

    found = find_up(Path(".venv311") / "Scripts" / "python.exe")
    if found is not None:
        try:
            proc = subprocess.run(
                [str(found), "-c", "import basic_pitch"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0:
                return str(found)
        except Exception:  # noqa: BLE001
            pass
    try:
        import basic_pitch  # noqa: F401
        return sys.executable
    except ImportError:
        return None


def bp_available() -> bool:
    """Whether Basic Pitch can run (env gate included).

    Args:
        None.

    Returns:
        True if usable.
    """
    if os.environ.get("AUDIO2MIDI_NO_BP"):
        return False
    return bp_python() is not None


def _filter_range(notes: list[dict], instrument: str) -> list[dict]:
    """Filter notes to the instrument pitch range + post-process.

    Drops very short/quiet notes and dedupes overlapping same-pitch notes.

    Args:
        notes: Raw note list. instrument: Instrument name.

    Returns:
        Filtered note list.
    """
    lo, hi = RANGES.get(instrument, (0, 127))
    hi = min(hi, CAPS.get(instrument, hi))
    out = []
    for n in notes:
        p = int(n["pitch"])
        s = float(n["start"])
        e = float(n["end"])
        v = int(n.get("velocity", 90))
        if not (lo <= p <= hi):
            continue
        if e - s < 0.08:
            continue
        if v < 40:
            continue
        out.append({"pitch": p, "start": s, "end": e, "velocity": v})
    # Dedup overlapping same pitch (keep higher velocity).
    out.sort(key=lambda n: (n["pitch"], n["start"]))
    deduped: list[dict] = []
    for n in out:
        if (deduped and deduped[-1]["pitch"] == n["pitch"]
                and n["start"] < deduped[-1]["end"]):
            if n["velocity"] > deduped[-1]["velocity"]:
                deduped[-1] = n
        else:
            deduped.append(n)
    return deduped


def _run_worker(stem_wavs: dict[str, Path]) -> dict[str, list[dict]]:
    """Run the Basic Pitch worker on several wavs (one TF load).

    Args:
        stem_wavs: Mapping instrument -> wav path.

    Returns:
        Mapping instrument -> raw note list.
    """
    py = bp_python()
    if py is None:
        return {}
    from .utils import bundled_path

    worker = bundled_path(Path("scripts") / "basicpitch_worker.py")
    if worker is None:
        return {}
    args = [f"{inst}={wav}" for inst, wav in stem_wavs.items()]
    proc = subprocess.run(
        [py, str(worker), *args],
        capture_output=True, text=True, timeout=900,
    )
    result: dict[str, list[dict]] = {}
    for line in (proc.stdout or "").splitlines():
        if not line.startswith("@@JSON@@"):
            continue
        key, _, payload = line[len("@@JSON@@"):].partition("@@")
        data = json.loads(payload)
        if isinstance(data, dict) and "error" in data:
            continue
        result[key] = data
    if not result:
        print(f"BP worker vacío: rc={proc.returncode} "
              f"stderr={(proc.stderr or '')[-400:]!r} "
              f"stdout={(proc.stdout or '')[-200:]!r}")
    return result


def transcribe_basicpitch_batch(
    stem_wavs: dict[str, Path | str],
) -> dict[str, list[dict]]:
    """Transcribe several stems with one Basic Pitch process.

    Args:
        stem_wavs: Mapping instrument -> wav path.

    Returns:
        Mapping instrument -> filtered note list.
    """
    paths = {inst: Path(wav) for inst, wav in stem_wavs.items()}
    raw = _run_worker(paths)
    return {inst: _filter_range(raw.get(inst, []), inst)
            for inst in paths}


def transcribe_basicpitch(
    stem_wav: str | Path, instrument: str = "piano",
) -> list[dict]:
    """Transcribe one stem with Basic Pitch.

    Args:
        stem_wav: Path to stem wav.
        instrument: Instrument name.

    Returns:
        List of {pitch, start, end, velocity}; [] on failure.
    """
    return transcribe_basicpitch_batch({instrument: stem_wav}).get(instrument, [])
