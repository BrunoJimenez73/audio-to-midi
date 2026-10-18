"""Utilidades compartidas: rutas, validación, constantes MIDI."""
from pathlib import Path

STEMS = ("drums", "bass", "vocals", "guitar", "piano", "other")

# stem -> (canal MIDI, programa GM)
MIDI_MAP: dict[str, tuple[int, int]] = {
    "piano": (0, 0),
    "guitar": (1, 27),
    "bass": (2, 33),
    "vocals": (4, 53),
    "drums": (9, 0),
    "other": (5, 0),
}


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return it.

    Args:
        path: Target directory.

    Returns:
        Resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_youtube_url(value: str) -> bool:
    """Check if string looks like a YouTube URL.

    Args:
        value: User input string.

    Returns:
        True if it contains youtube.com or youtu.be.
    """
    v = value.lower()
    return "youtube.com" in v or "youtu.be" in v
