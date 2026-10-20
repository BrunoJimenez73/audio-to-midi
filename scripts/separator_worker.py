"""Worker: separacion "Mejor" jerarquica (Roformer + Demucs) en un solo proceso.

Uso:
    separator_worker.py <mix.wav> <out_dir> [--models-dir DIR]

Pasadas:
  1. vocals_mel_band_roformer (UVR, fp16, segment 160) -> voz.
  2. htdemucs_ft (4 stems) sobre la mezcla -> drums, bass, other_raw.
  3. htdemucs_6s sobre other_raw -> guitar, piano, other (+ restos).

Escribe <out_dir>/{vocals,drums,bass,guitar,piano,other}.wav y imprime una
linea en stdout: @@SEPARATOR@@{"status": ...}. Codigo de salida: 0 ok, 1 error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"
VOCAL_SEGMENT = 160
FT_MODEL = "htdemucs_ft.yaml"
SIXS_MODEL = "htdemucs_6s.yaml"
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _empty_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _run_stage(
    model: str,
    audio: Path,
    out_dir: Path,
    models_dir: Path,
    *,
    seg: int | None = None,
    fp16: bool = False,
    shifts: int | None = None,
) -> dict[str, Path]:
    """Carga un modelo y separa; devuelve stems por nombre minuscula.

    Args:
        model: Nombre de modelo (audio-separator).
        audio: Wav de entrada.
        out_dir: Directorio de salida de esta pasada.
        models_dir: Cache de modelos descargados.
        seg: segment_size (mdxc) opcional.
        fp16: Usar precision nativa fp16 en CUDA.
        shifts: shifts de Demucs si aplica.

    Returns:
        Mapping nombre_minuscula -> wav, ignorando stems no canonicos.
    """
    from audio_separator.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    kw: dict = {}
    if seg is not None:
        kw["mdxc_params"] = {
            "segment_size": seg,
            "override_model_segment_size": True,
        }
    if fp16:
        kw["use_native_fp16"] = True
    if shifts is not None:
        kw["demucs_params"] = {"shifts": int(shifts), "overlap": 0.25}
    sep = Separator(
        model_file_dir=str(models_dir),
        output_dir=str(out_dir),
        output_format="WAV",
        log_level=40,
        **kw,
    )
    sep.load_model(model_filename=model)
    produced: list[Path] = []
    for raw in sep.separate(str(audio)):
        path = Path(raw)
        if not path.is_absolute():
            path = out_dir / path.name
        produced.append(path)
    del sep
    _empty_cuda()
    out: dict[str, Path] = {}
    for path in produced:
        # Ultimo grupo entre parentesis: los nombres encadenados tipo
        # "mix_(Other)_x_(Guitar)_y.wav" llevan el stem propio al final.
        groups = re.findall(r"\(([A-Za-z]+)\)", path.name)
        if groups:
            key = groups[-1].lower()
            if key in STEMS:
                out[key] = path
    return out


def _load(path: Path) -> tuple[np.ndarray, int]:
    """Carga wav como float32 2D (n, ch).

    Args:
        path: Ruta del wav.

    Returns:
        (array float32 2D, sample rate).
    """
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return y, int(sr)


def _fit(y: np.ndarray, sr: int, target_sr: int, target_len: int) -> np.ndarray:
    """Iguala sr y longitud al de referencia (recorta o rellena de cero).

    Args:
        y: Audio 2D.
        sr: Sample rate de y.
        target_sr: Sample rate de referencia.
        target_len: Numero de muestras de referencia.

    Returns:
        Audio alineado (target_len, ch).
    """
    if sr != target_sr:
        import librosa

        y = librosa.resample(
            y.T, orig_sr=sr, target_sr=target_sr
        ).T.astype(np.float32)
    if y.shape[0] < target_len:
        y = np.pad(y, ((0, target_len - y.shape[0]), (0, 0)))
    return y[:target_len]


def _finite_or(y: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Devuelve y si es finita y tiene energia; si no, el fallback.

    Args:
        y: Audio candidato.
        fallback: Audio de respaldo (misma forma ya ajustada).

    Returns:
        Audio utilizable.
    """
    if y.shape != fallback.shape:
        return fallback
    if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) < 1e-6:
        return fallback
    return y


def main() -> int:
    """Ejecuta las 3 pasadas y escribe los 6 stems finales.

    Returns:
        0 si todo ok, 1 en error.
    """
    parser = argparse.ArgumentParser(description="Separacion Mejor (jerarquica)")
    parser.add_argument("mix")
    parser.add_argument("out_dir")
    parser.add_argument("--models-dir", required=True)
    args = parser.parse_args()

    mix = Path(args.mix)
    dest = Path(args.out_dir)
    models_dir = Path(args.models_dir)
    dest.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stages: dict[str, float] = {}
    vocals_source = "roformer"

    # Pasada 1: voz (Roformer fp16). Fallo -> se usa la voz de htdemucs_ft.
    t0 = time.time()
    try:
        vocals = _run_stage(
            VOCAL_MODEL,
            mix,
            dest / "_stage_vocals",
            models_dir,
            seg=VOCAL_SEGMENT,
            fp16=True,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"stage vocals fallo ({exc!r}); se usara voz de htdemucs_ft")
        vocals = {}
    stages["vocals"] = round(time.time() - t0, 1)

    # Pasada 2: htdemucs_ft -> drums, bass, other_raw.
    t0 = time.time()
    try:
        ft = _run_stage(FT_MODEL, mix, dest / "_stage_ft", models_dir, shifts=1)
    except Exception as exc:  # noqa: BLE001
        _log(f"stage ft fallo: {exc!r}")
        print("@@SEPARATOR@@" + json.dumps({"status": "error", "error": repr(exc)}))
        return 1
    stages["ft"] = round(time.time() - t0, 1)
    need = ("drums", "bass", "other", "vocals")
    missing = [k for k in need if k not in ft]
    if missing:
        print(
            "@@SEPARATOR@@"
            + json.dumps({"status": "error", "error": f"ft sin {missing}"})
        )
        return 1

    # Pasada 3: htdemucs_6s sobre other_raw -> guitar, piano, other.
    # Nombre limpio: evita que "(Other)" del nombre encadene en la salida.
    raw_other = ft["other"]
    clean_other = raw_other.with_name("other_raw.wav")
    try:
        os.replace(raw_other, clean_other)
    except OSError:
        clean_other = raw_other
    t0 = time.time()
    try:
        six = _run_stage(
            SIXS_MODEL, clean_other, dest / "_stage_6s", models_dir, shifts=1
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"stage 6s fallo: {exc!r}")
        print("@@SEPARATOR@@" + json.dumps({"status": "error", "error": repr(exc)}))
        return 1
    stages["six"] = round(time.time() - t0, 1)
    missing = [k for k in ("guitar", "piano", "other") if k not in six]
    if missing:
        print(
            "@@SEPARATOR@@"
            + json.dumps({"status": "error", "error": f"6s sin {missing}"})
        )
        return 1

    # Ensamblado: referencia de forma = drums de ft.
    ref, ref_sr = _load(ft["drums"])
    t_len, ref_ch = ref.shape

    def at(mapping: dict[str, Path], key: str) -> np.ndarray | None:
        if key not in mapping:
            return None
        y, sr = _load(mapping[key])
        if y.shape[1] != ref_ch:
            y = np.repeat(y[:, :1], ref_ch, axis=1) if y.shape[1] == 1 else y[:, :ref_ch]
        return _fit(y, sr, ref_sr, t_len)

    ft_voc = at(ft, "vocals")
    voc_main = at(vocals, "vocals")
    if voc_main is None:
        vocals_source = "ft"
        voc_main = ft_voc
    else:
        voc_main = _finite_or(voc_main, ft_voc)

    parts: dict[str, np.ndarray | None] = {
        "vocals": voc_main,
        "drums": at(ft, "drums"),
        "bass": at(ft, "bass"),
        "guitar": at(six, "guitar"),
        "piano": at(six, "piano"),
        "other": at(six, "other"),
    }
    # Restos del residual -> su stem (mejora la suma global).
    for key in ("vocals", "drums", "bass"):
        res = at(six, key)
        if res is not None and parts[key] is not None:
            parts[key] = parts[key] + res

    written: dict[str, str] = {}
    for key in STEMS:
        y = parts[key]
        if y is None or not np.all(np.isfinite(y)):
            print(
                "@@SEPARATOR@@"
                + json.dumps({"status": "error", "error": f"stem {key} invalido"})
            )
            return 1
        peak = float(np.max(np.abs(y)))
        if peak > 1.0:
            y = y / peak
        target = dest / f"{key}.wav"
        sf.write(str(target), y, ref_sr, subtype="PCM_16")
        written[key] = str(target)

    # Limpieza de intermedios (se dejan por depuracion salvo si AUDIO2MIDI_KEEP_TMP).
    if not os.environ.get("AUDIO2MIDI_KEEP_TMP"):
        import shutil

        for sub in ("_stage_vocals", "_stage_ft", "_stage_6s"):
            shutil.rmtree(dest / sub, ignore_errors=True)

    result = {
        "status": "ok",
        "elapsed": round(time.time() - t_start, 1),
        "stages": stages,
        "vocals_source": vocals_source,
        "vram_mb": _peak_vram_mb(),
        "stems": written,
    }
    print("@@SEPARATOR@@" + json.dumps(result))
    return 0


def _peak_vram_mb() -> int:
    """Pico de VRAM usado en la sesion.

    Returns:
        Megabytes, o 0 si no hay CUDA.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6)
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
