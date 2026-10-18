"""Run full pipeline on a URL with best quality and print a summary."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AUDIO2MIDI_MELODIC_BACKEND", "basicpitch")
# Ensure Demucs is allowed.
os.environ.pop("AUDIO2MIDI_NO_DEMUCS", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline

url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=CoNtYC_XDC8"
t0 = time.time()
res = run_pipeline(url, progress=None, bpm=0.0, grid_beats=0.25)
elapsed = time.time() - t0
print(f"RUN_DIR={res['run_dir']}")
print(f"BPM={res['bpm']}")
print(f"ELAPSED={elapsed:.1f}s")
print(f"NOTES={res['notes_count']}")
print(f"FULL_MIX={res['full_mix']}")
for inst, p in res["midis"].items():
    print(f"MIDI_{inst.upper()}={p}")
# Guitar stats
try:
    import pretty_midi
    mid = pretty_midi.PrettyMIDI(str(res["midis"]["guitar"]))
    notes = mid.instruments[0].notes if mid.instruments else []
    print(f"GUITAR_NOTES={len(notes)}")
    if notes:
        pitches = [n.pitch for n in notes]
        print(f"GUITAR_PITCH_MIN={min(pitches)} MAX={max(pitches)}")
        print(f"GUITAR_DUR={mid.get_end_time():.2f}s")
        preview = [(n.pitch, round(n.start, 2), round(n.end, 2), n.velocity) for n in notes[:5]]
        print(f"GUITAR_FIRST5={preview}")
except Exception as e:
    print(f"GUITAR_STATS_ERR={e}")