"""CLI mínimo: clasifica input y genera un MIDI demo."""
import argparse
from pathlib import Path

from .ingest import resolve_input
from .midi_merge import merge_to_midi


def main() -> None:
    """Entry point for python -m src.cli."""
    parser = argparse.ArgumentParser(description="audio-to-midi (esqueleto)")
    parser.add_argument("input", help="URL YouTube o ruta mp3/wav")
    parser.add_argument("--out", default="outputs/demo/full_mix.mid")
    args = parser.parse_args()
    info = resolve_input(args.input)
    print(f"Input: {info}")
    demo = {"piano": [{"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 90}]}
    out = merge_to_midi(demo, Path(args.out))
    print(f"MIDI demo escrito en: {out}")


if __name__ == "__main__":
    main()
