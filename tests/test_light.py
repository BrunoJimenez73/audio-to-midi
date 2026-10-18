"""Tests ligeros: transcripción + pipeline E2E con audio sintético."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.pipeline import run_pipeline
from src.transcribe_drums import transcribe_drums
from src.transcribe_melodic import transcribe_melodic


def _sine_wav(path: Path, freq: float = 440.0, seconds: float = 2.0) -> Path:
    """Write mono sine wav for tests.

    Args:
        path: Destination wav.
        freq: Sine frequency.
        seconds: Duration.

    Returns:
        Path written.
    """
    sr = 22050
    t = np.arange(int(sr * seconds)) / sr
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), y, sr)
    return path


def test_melodic_finds_a4(tmp_path: Path) -> None:
    """A 440Hz sine transcribes near MIDI 69."""
    wav = _sine_wav(tmp_path / "a4.wav")
    notes = transcribe_melodic(wav, "piano", backend="numpy")
    assert len(notes) >= 1
    assert any(67 <= n["pitch"] <= 71 for n in notes)


def test_drums_on_clicks(tmp_path: Path) -> None:
    """Click track yields drum notes."""
    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    y[:: sr // 2] = 1.0
    p = tmp_path / "click.wav"
    sf.write(str(p), y, sr)
    notes = transcribe_drums(p)
    assert len(notes) >= 1


def test_pipeline_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pipeline on synth audio produces stems + full_mix."""
    monkeypatch.setenv("AUDIO2MIDI_NO_DEMUCS", "1")
    monkeypatch.setenv("AUDIO2MIDI_NO_BP", "1")
    wav = _sine_wav(tmp_path / "in.wav")
    res = run_pipeline(str(wav), out_root=tmp_path / "outputs")
    full = Path(str(res["full_mix"]))
    assert full.exists() and full.stat().st_size > 0
    stems = res["stems"]
    assert isinstance(stems, dict) and len(stems) == 6


def test_separate_fallback(tmp_path: Path) -> None:
    """Numpy fallback produces 6 stems without Demucs."""
    from src.separate import _fallback_hpss

    wav = _sine_wav(tmp_path / "in.wav")
    stems = _fallback_hpss(wav, tmp_path / "stems")
    assert set(stems) == {"drums", "bass", "vocals", "guitar", "piano", "other"}


def test_demucs_python_detect() -> None:
    """Demucs python locator returns usable path or None."""
    from pathlib import Path

    from src.separate import _demucs_python

    found = _demucs_python()
    assert found is None or Path(found).exists()


def test_chord_finds_both_pitches(tmp_path: Path) -> None:
    """C4+E4 chord transcribes both MIDI 60 and 64."""
    from src.transcribe_melodic import transcribe_melodic as tm

    sr = 22050
    t = np.arange(sr * 2) / sr
    y = (0.4 * np.sin(2 * np.pi * 261.63 * t)
         + 0.4 * np.sin(2 * np.pi * 329.63 * t)).astype(np.float32)
    p = tmp_path / "chord.wav"
    sf.write(str(p), y, sr)
    pitches = {n["pitch"] for n in tm(p, "piano", backend="numpy")}
    assert 60 in pitches and 64 in pitches


def test_hpss_splits_sine_clicks(tmp_path: Path) -> None:
    """HPSS keeps sine in harmonic and clicks in percussive."""
    from src.hpss import hpss

    sr = 22050
    t = np.arange(sr * 2) / sr
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mix = sine.copy()
    mix[:: sr // 4] += 0.9
    harm, perc = hpss(mix)
    assert float(np.corrcoef(harm, sine)[0, 1]) > 0.95
    assert float(np.abs(perc).max()) > 0.5


def _click_track(path: Path, bpm: float = 120.0, seconds: float = 8.0) -> Path:
    """Write click track at given BPM.

    Args:
        path: Destination wav. bpm: Tempo. seconds: Duration.

    Returns:
        Path written.
    """
    sr = 22050
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    step = int(sr * 60.0 / bpm)
    y[::step] = 1.0
    sf.write(str(path), y, sr)
    return path


def test_estimate_bpm_clicks(tmp_path: Path) -> None:
    """BPM estimator finds 120 BPM click track within 3 BPM."""
    from src.tempo import estimate_bpm

    wav = _click_track(tmp_path / "click120.wav", bpm=120.0)
    assert abs(estimate_bpm(wav) - 120.0) < 3.0


def test_band_filter(tmp_path: Path) -> None:
    """Band filter keeps in-band sine, kills out-of-band."""
    from src.dsp import band_filter

    sr = 22050
    t = np.arange(sr * 2) / sr
    mix = (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    low = band_filter(mix, sr, 0.0, 250.0)
    assert float(np.abs(low).max()) > 0.3
    assert float(np.corrcoef(low, np.sin(2 * np.pi * 100 * t))[0, 1]) > 0.9


def test_dedupe_and_skyline() -> None:
    """Dedupe merges copies; skyline finds top line."""
    from src.assign import dedupe_pools, skyline_ids

    pools = {
        "piano": [{"pitch": 60, "start": 0.0, "end": 1.0, "velocity": 90}],
        "guitar": [{"pitch": 60, "start": 0.02, "end": 1.0, "velocity": 95},
                   {"pitch": 67, "start": 0.0, "end": 1.0, "velocity": 90}],
        "vocals": [],
    }
    merged = dedupe_pools(pools)
    assert len(merged) == 2  # 60 merged, 67 alone
    assert skyline_ids(merged) == {1}  # top line is 67


def test_assign_disjoint() -> None:
    """Assigned tracks share no (pitch, onset)."""
    from src.assign import assign_notes

    sr = 22050
    t = np.arange(sr) / sr
    harm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mid = harm * 2.0
    side = harm * 0.1
    pools = {
        "piano": [{"pitch": 69, "start": 0.0, "end": 0.8, "velocity": 90}],
        "guitar": [{"pitch": 69, "start": 0.01, "end": 0.8, "velocity": 90}],
        "vocals": [{"pitch": 69, "start": 0.0, "end": 0.8, "velocity": 90}],
        "other": [],
    }
    out = assign_notes(pools, harm, mid, side, sr)
    keys = [(n["pitch"], round(n["start"], 1))
            for inst in ("piano", "guitar", "vocals") for n in out[inst]]
    assert len(keys) == len(set(keys)) == 1  # one home only


def test_merge_bpm_duration(tmp_path: Path) -> None:
    """Merged MIDI with real BPM plays back at true duration."""
    import mido

    from src.midi_merge import merge_to_midi, quantize_notes

    notes = {"piano": [{"pitch": 60, "start": 0.0, "end": 60.0,
                        "velocity": 90}]}
    out = merge_to_midi(notes, tmp_path / "t.mid", bpm=150.0)
    assert abs(mido.MidiFile(str(out)).length - 60.0) < 1.0
    q = quantize_notes([{"pitch": 60, "start": 0.11, "end": 0.5,
                         "velocity": 90}], bpm=120.0)
    assert q[0]["start"] == 0.125  # 16th grid at 120 BPM


def test_voice_split(tmp_path: Path) -> None:
    """Voice mask isolates solo vibrato tone; steady chord stays out."""
    from src.voice import split_voice

    sr = 22050
    t = np.arange(sr * 4) / sr
    vibf = 440 + 14 * np.sin(2 * np.pi * 5 * t)
    vib = sum(0.6 / k * np.sin(2 * np.pi * k * vibf * t)
              for k in [1, 2, 3]).astype(np.float32)
    chord = (0.25 * np.sin(2 * np.pi * 196 * t)
             + 0.25 * np.sin(2 * np.pi * 246.94 * t)
             + 0.2 * np.sin(2 * np.pi * 293.66 * t)).astype(np.float32)
    voice, backing, vf = split_voice(vib, sr)
    assert vf > 0.5
    assert float(np.corrcoef(voice, vib)[0, 1]) > 0.7
    v2, _, vf2 = split_voice(chord, sr)
    assert vf2 < 0.1


def test_drums_grid_gate() -> None:
    """Grid-aligned hits pass; random hits are rejected as no-drums."""
    from src.transcribe_drums import drums_on_grid

    grid = [{"pitch": 36, "start": i * 0.5, "end": i * 0.5 + 0.1,
             "velocity": 100} for i in range(16)]
    assert drums_on_grid(grid, bpm=120.0) is True
    rng = np.random.default_rng(0)
    rand = [{"pitch": 38, "start": float(t), "end": float(t) + 0.1,
             "velocity": 100} for t in np.sort(rng.uniform(0, 8, 30))]
    assert drums_on_grid(rand, bpm=120.0) is False
