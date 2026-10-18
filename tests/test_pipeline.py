"""Tests de ingesta (sin red) + merge MIDI multicanal."""
from pathlib import Path

import mido
import pytest

from src.ingest import resolve_input
from src.midi_merge import merge_to_midi


def test_resolve_youtube() -> None:
    """YouTube URL is classified as youtube."""
    info = resolve_input("https://www.youtube.com/watch?v=abc123")
    assert info["kind"] == "youtube"


def test_resolve_empty_raises() -> None:
    """Empty input raises ValueError."""
    with pytest.raises(ValueError):
        resolve_input("   ")


def test_resolve_missing_file_raises(tmp_path: Path) -> None:
    """Missing local file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        resolve_input(str(tmp_path / "noexiste.mp3"))


def test_merge_channels(tmp_path: Path) -> None:
    """Merged MIDI keeps piano ch0, bass ch2, drums ch9."""
    notes = {
        "piano": [{"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 90}],
        "bass": [{"pitch": 36, "start": 0.0, "end": 0.5, "velocity": 90}],
        "drums": [{"pitch": 36, "start": 0.0, "end": 0.2, "velocity": 100}],
    }
    out = merge_to_midi(notes, tmp_path / "full_mix.mid")
    assert out.exists()
    mid = mido.MidiFile(str(out))
    assert len(mid.tracks) == 4  # conductor + 3 instruments
    channels = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type in ("note_on", "note_off"):
                channels.add(msg.channel)
    assert channels == {0, 2, 9}
