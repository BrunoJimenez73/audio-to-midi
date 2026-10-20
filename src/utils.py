"""Utilidades compartidas: rutas, validación, constantes MIDI."""
import sys
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


def bundled_path(rel: str | Path) -> Path | None:
    """Find a data file bundled by PyInstaller or present in the repo.

    Checks sys._MEIPASS (frozen) first, then walks up from the source
    root and the executable directory.

    Args:
        rel: Relative path (e.g. 'scripts/basicpitch_worker.py').

    Returns:
        Absolute path if found, else None.
    """
    rel_p = Path(rel)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / rel_p
        if cand.exists():
            return cand
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent.parent)
    seen: set[Path] = set()
    for root in roots:
        for base in (root, *list(root.parents)[:6]):
            if base in seen:
                continue
            seen.add(base)
            cand = base / rel_p
            if cand.exists():
                return cand
    return None


def find_up(rel: str | Path) -> Path | None:
    """Find a file/dir by walking up from exe dir and source root.

    Args:
        rel: Relative path to look for (e.g. '.venv311/Scripts/python.exe').

    Returns:
        Absolute path if found, else None.
    """
    return bundled_path(rel)
