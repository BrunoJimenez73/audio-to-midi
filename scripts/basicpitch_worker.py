"""Worker: transcribe one or more wavs with Basic Pitch, print JSON.

Usage: basicpitch_worker.py <instrument>=<wav> [<instrument>=<wav> ...]

Prints one line per input: @@JSON@@<instrument>@@<notes_json>
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.transcribe_melodic import RANGES as _RANGES
except Exception:  # noqa: BLE001
    _RANGES = {"piano": (21, 108), "guitar": (40, 88), "bass": (28, 64),
               "vocals": (40, 84), "other": (21, 108)}

from basic_pitch.inference import predict  # noqa: E402

CAPS = {"piano": 108, "guitar": 79, "bass": 55, "vocals": 81, "other": 96}

THRESHOLDS = {
    "vocals": {"onset_threshold": 0.3, "frame_threshold": 0.2},
}
DEFAULT_THRESH = {"onset_threshold": 0.5, "frame_threshold": 0.42}


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _wav_notes(wav: str, instrument: str) -> list[dict]:
    """Transcribe a single wav file.

    Args:
        wav: Audio path. instrument: Instrument key for thresholds.

    Returns:
        List of {pitch, start, end, velocity}.
    """
    lo, hi = _RANGES.get(instrument, (0, 127))
    cap = CAPS.get(instrument, hi)
    hi = min(hi, cap)
    kwargs = dict(THRESHOLDS.get(instrument, DEFAULT_THRESH))
    if lo > 0:
        kwargs["minimum_frequency"] = _midi_to_hz(lo)
    if hi < 127:
        kwargs["maximum_frequency"] = _midi_to_hz(hi)
    _, _, note_events = predict(wav, **kwargs)
    notes = []
    for ev in note_events:
        start = float(ev[0])
        end = float(ev[1])
        pitch = int(ev[2])
        amp = float(ev[3]) if len(ev) > 3 else 0.5
        vel = max(1, min(127, int(round(amp * 127.0))))
        if end <= start:
            end = start + 0.05
        notes.append({"pitch": pitch, "start": start, "end": end,
                      "velocity": vel})
    return notes


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: basicpitch_worker.py <instrument>=<wav> [...]",
              file=sys.stderr)
        return 2
    status = 0
    for arg in sys.argv[1:]:
        inst, _, wav = arg.partition("=")
        if not wav:
            wav, inst = arg, os.path.splitext(os.path.basename(arg))[0]
        try:
            notes = _wav_notes(wav, inst)
        except Exception as exc:  # noqa: BLE001
            notes = {"error": str(exc)}
            status = 1
        print("@@JSON@@" + inst + "@@" + json.dumps(notes), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())