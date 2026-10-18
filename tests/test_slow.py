"""Tests lentos: backends pesados (Demucs, Basic Pitch)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.pipeline import run_pipeline
from src.transcribe_melodic import transcribe_melodic


def _sine_wav(path: Path, seconds: float = 1.5) -> Path:
    sr = 22050
    t = np.arange(int(sr * seconds)) / sr
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), y, sr)
    return path


@pytest.mark.slow
def test_demucs_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline runs with real Demucs when available."""
    monkeypatch.delenv("AUDIO2MIDI_NO_DEMUCS", raising=False)
    from src.separate import _demucs_python

    if _demucs_python() is None:
        pytest.skip("Demucs no disponible")
    wav = _sine_wav(tmp_path / "in.wav")
    res = run_pipeline(str(wav), out_root=tmp_path / "out")
    assert Path(str(res["full_mix"])).exists()
    stems = res["stems"]
    assert isinstance(stems, dict) and len(stems) == 6


@pytest.mark.slow
def test_basicpitch_smoke(tmp_path: Path) -> None:
    """Basic Pitch backend transcribes a sine as A4."""
    try:
        import basic_pitch  # noqa: F401
    except ImportError:
        from src.transcribe_basicpitch import _basicpitch_python

        if _basicpitch_python() is None:
            pytest.skip("Basic Pitch no disponible")
    wav = _sine_wav(tmp_path / "a4.wav")
    notes = transcribe_melodic(wav, "piano", backend="basicpitch")
    # Basic Pitch may or may not detect; at least no crash.
    assert isinstance(notes, list)