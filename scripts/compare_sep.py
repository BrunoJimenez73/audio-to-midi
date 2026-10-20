"""Compara separacion rapido vs mejor y deja stems A/B para escuchar.

Uso:
    python scripts\\compare_sep.py [entrada] [--out outputs\\bench_sep]

``entrada`` es una URL de YouTube o ruta local; por defecto, el tema de
prueba. Crea <out>/<timestamp>/stems_rapido y stems_mejor mas un README con
instrucciones de escucha, e imprime una tabla de RMS/duracion por stem.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.ingest import prepare_input
from src.separate import separate_demucs

DEFAULT_URL = "https://www.youtube.com/watch?v=trVCbDpmIH0"


def _stats(wav: Path) -> dict[str, float]:
    """Compute duration/RMS/peak stats for a wav.

    Args:
        wav: Path to wav file.

    Returns:
        Dict with dur_s, rms_db, peak.
    """
    y, sr = sf.read(str(wav), always_2d=True, dtype="float32")
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    return {
        "dur_s": round(y.shape[0] / sr, 2),
        "rms_db": round(20 * np.log10(rms + 1e-9), 1),
        "peak": round(float(np.max(np.abs(y))), 3),
    }


def main() -> int:
    """Run both modes and print the comparison table.

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--out", default="outputs/bench_sep")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp
    print(f"Preparando audio -> {run_dir}")
    mix = prepare_input(args.entrada, run_dir / "work")
    results: dict[str, dict[str, float]] = {}
    for mode in ("rapido", "mejor"):
        dest = run_dir / f"stems_{mode}"
        print(f"--- modo {mode} ---")
        t0 = time.time()
        stems = separate_demucs(mix, dest, sep=mode)
        elapsed = round(time.time() - t0, 1)
        if len(stems) < 6:
            print(f"AVISO: modo {mode} devolvio {len(stems)} stems")
        for name in ("drums", "bass", "vocals", "guitar", "piano", "other"):
            wav = stems.get(name)
            if wav is not None and Path(wav).exists():
                results[f"{mode}/{name}"] = _stats(Path(wav))
        results[f"{mode}/_elapsed_s"] = {"dur_s": elapsed}
        print(f"modo {mode}: {elapsed}s, {len(stems)} stems")

    lines = ["Comparacion de separacion: rapido vs mejor", ""]
    lines.append(f"Mezcla: {mix}")
    lines.append("")
    lines.append(f"{'stem':10s} {'rapido rms':>12s} {'mejor rms':>12s} "
                 f"{'rapido dur':>11s} {'mejor dur':>11s}")
    for name in ("drums", "bass", "vocals", "guitar", "piano", "other"):
        a = results.get(f"rapido/{name}", {})
        b = results.get(f"mejor/{name}", {})
        lines.append(
            f"{name:10s} {a.get('rms_db', float('nan')):>12} "
            f"{b.get('rms_db', float('nan')):>12} "
            f"{a.get('dur_s', float('nan')):>11} {b.get('dur_s', float('nan')):>11}"
        )
    lines.append("")
    lines.append(
        f"Tiempos: rapido={results.get('rapido/_elapsed_s', {}).get('dur_s')}s "
        f"mejor={results.get('mejor/_elapsed_s', {}).get('dur_s')}s"
    )
    lines += [
        "",
        "ESCUCHA A/B (abre los dos stems del mismo instrumento):",
        "  1. vocals: el modo mejor debe tener menos guitarra/piano de fondo.",
        "  2. guitar y piano: el modo mejor debe ensuciarse menos entre si.",
        "  3. bass y drums: el modo mejor debe sonar mas limpio y definido.",
        "  4. overall: compara con el mezcla original en work/mix.wav.",
    ]
    report = "\n".join(lines)
    (run_dir / "README.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nStems A/B en: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
