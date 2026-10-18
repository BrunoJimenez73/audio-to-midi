"""Ingesta: YouTube URL o fichero local -> wav 44.1kHz."""
import shutil
import subprocess
from pathlib import Path

from .utils import ensure_dir, is_youtube_url

TARGET_SR = 44100


def resolve_input(user_input: str) -> dict[str, str]:
    """Classify user input as youtube URL or local file.

    Args:
        user_input: URL or local path string.

    Returns:
        Dict with kind ('youtube'|'file') and value.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If input is empty.
    """
    text = user_input.strip().strip('"').strip("'")
    if not text:
        raise ValueError("Input vacío: pega una URL o ruta mp3.")
    if is_youtube_url(text):
        return {"kind": "youtube", "value": text}
    p = Path(text)
    if not p.exists():
        raise FileNotFoundError(f"No existe el fichero: {text}")
    return {"kind": "file", "value": str(p)}


def download_youtube(url: str, out_dir: str | Path) -> Path:
    """Download YouTube audio and convert to wav 44.1kHz.

    Args:
        url: YouTube URL.
        out_dir: Destination directory.

    Returns:
        Path to downloaded wav.

    Raises:
        RuntimeError: If yt-dlp or ffmpeg fail.
    """
    from yt_dlp import YoutubeDL

    dest = ensure_dir(out_dir)
    tmpl = str(dest / "yt_%(id)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": tmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav",
             "preferredquality": "0"}
        ],
    }
    with YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    wavs = sorted(dest.glob("*.wav"))
    if not wavs:
        raise RuntimeError("yt-dlp no generó wav (¿falta ffmpeg?).")
    # Preferir el más reciente (el de esta descarga).
    wavs.sort(key=lambda p: p.stat().st_mtime)
    return wavs[-1]


def convert_to_wav(src: str | Path, out_path: str | Path) -> Path:
    """Convert any audio file to wav 44.1kHz stereo via ffmpeg.

    Args:
        src: Input audio path (mp3/wav/m4a/...).
        out_path: Destination wav path.

    Returns:
        Path to converted wav.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", str(TARGET_SR),
           "-ac", "2", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg falló: {proc.stderr[-500:]}")
    return out


def prepare_input(user_input: str, work_dir: str | Path) -> Path:
    """Resolve URL/file and return normalized wav 44.1kHz.

    Args:
        user_input: URL, local audio path, or direct wav path.
        work_dir: Working directory for downloads/conversions.

    Returns:
        Path to mix.wav listo para Demucs.
    """
    work = ensure_dir(work_dir)
    info = resolve_input(user_input)
    if info["kind"] == "youtube":
        raw = download_youtube(info["value"], work)
    else:
        raw = Path(info["value"])
    # Si ya es wav lo copiamos igual vía ffmpeg para normalizar.
    if raw.suffix.lower() == ".wav":
        try:
            return convert_to_wav(raw, work / "mix.wav")
        except RuntimeError:
            dest = work / "mix.wav"
            shutil.copy(raw, dest)
            return dest
    ext = raw.suffix.lower()
    if ext in (".mp3", ".m4a", ".ogg", ".flac", ".opus", ".webm"):
        return convert_to_wav(raw, work / "mix.wav")
    return convert_to_wav(raw, work / "mix.wav")
