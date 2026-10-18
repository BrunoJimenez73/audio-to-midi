"""Reparto disjunto: cada nota a un solo instrumento (rol + timbre)."""
import numpy as np

DUP_PITCH_TOL = 0
DUP_TIME_TOL = 0.07

# Tesitura válida por pista destino (lo demás va a 'other').
OUT_RANGES: dict[str, tuple[int, int]] = {
    "piano": (21, 108),
    "guitar": (40, 88),
    "vocals": (36, 84),
    "other": (0, 127),
}


def dedupe_pools(pools: dict[str, list[dict]]) -> list[dict]:
    """Merge same-pitch/near-onset notes detected in several stems.

    Args:
        pools: Mapping stem -> note list ({pitch,start,end,velocity}).

    Returns:
        Merged notes with extra key 'sources' (set of stems).
    """
    flat: list[dict] = []
    for stem, notes in pools.items():
        for n in notes:
            flat.append({"pitch": int(n["pitch"]), "start": float(n["start"]),
                         "end": float(n["end"]),
                         "velocity": int(n.get("velocity", 90)),
                         "sources": {stem}})
    flat.sort(key=lambda n: (n["start"], n["pitch"]))
    merged: list[dict] = []
    last_by_pitch: dict[int, int] = {}
    for n in flat:
        j = last_by_pitch.get(n["pitch"])
        if j is not None and n["start"] - merged[j]["start"] <= DUP_TIME_TOL:
            m = merged[j]
            m["end"] = max(m["end"], n["end"])
            m["velocity"] = max(m["velocity"], n["velocity"])
            m["sources"] |= n["sources"]
        else:
            last_by_pitch[n["pitch"]] = len(merged)
            merged.append(n)
    return merged


def skyline_ids(notes: list[dict]) -> set[int]:
    """Find notes forming the top-line melody (skyline).

    Args:
        notes: Merged note list.

    Returns:
        Indices of notes that are the highest pitch at some moment.
    """
    bounds = sorted({n["start"] for n in notes} | {n["end"] for n in notes})
    melody: set[int] = set()
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 1e-6:
            continue
        t = (a + b) / 2.0
        best, best_i = -1, -1
        for i, n in enumerate(notes):
            if n["start"] <= t < n["end"] and n["pitch"] > best:
                best, best_i = n["pitch"], i
        if best_i >= 0:
            melody.add(best_i)
    return melody


def _rms(seg: np.ndarray) -> float:
    """Root-mean-square of a segment.

    Args:
        seg: Samples.

    Returns:
        RMS value.
    """
    return float(np.sqrt(np.mean(seg ** 2)) + 1e-9)


def _decay_ratio(harm: np.ndarray, sr: int, start: float, end: float) -> float:
    """Attack vs sustain energy ratio (piano-like if high).

    Args:
        harm: Harmonic mono audio. sr: Rate. start/end: Note bounds.

    Returns:
        RMS(first 80ms) / RMS(200-450ms into note).
    """
    a = int(start * sr)
    attack = harm[a:a + int(0.08 * sr)]
    tail = harm[a + int(0.20 * sr):a + int(0.45 * sr)]
    if len(tail) < int(0.05 * sr):
        tail = harm[a + int(0.10 * sr):int(end * sr)]
    if len(attack) == 0 or len(tail) == 0:
        return 1.0
    return _rms(attack) / _rms(tail)


def _center_ratio(mid: np.ndarray, side: np.ndarray, sr: int,
                  start: float, end: float) -> float:
    """Mid vs side energy (vocals/bass usually centered).

    Args:
        mid: Mid channel. side: Side channel. sr: Rate.

    Returns:
        Mid-energy / side-energy over first 0.4s of note.
    """
    a = int(start * sr)
    b = min(int(end * sr), a + int(0.4 * sr))
    return _rms(mid[a:b]) / _rms(side[a:b])


def _has_vibrato(harm: np.ndarray, sr: int, start: float, end: float) -> bool:
    """Detect F0 wobble typical of voice/lead guitar.

    Args:
        harm: Harmonic mono audio. sr: Rate. start/end: Note bounds.

    Returns:
        True if sustain F0 std exceeds 0.25 semitones.
    """
    dur = end - start
    if dur < 0.35:
        return False
    s0 = start + 0.15 * dur
    s1 = start + 0.85 * dur
    hop = int(0.03 * sr)
    win = int(0.06 * sr)
    f0s: list[float] = []
    p = int(s0 * sr)
    while p + win < int(s1 * sr):
        frame = harm[p:p + win].astype(np.float64)
        if np.mean(frame ** 2) < 1e-7:
            p += hop
            continue
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if corr[0] <= 0:
            p += hop
            continue
        corr = corr / corr[0]
        lo = max(1, int(sr / 1200.0))
        hi = min(len(corr) - 1, int(sr / 70.0))
        peak = lo + int(np.argmax(corr[lo:hi]))
        if corr[peak] > 0.5:
            f0s.append(sr / peak)
        p += hop
    if len(f0s) < 4:
        return False
    semis = 12 * np.log2(np.array(f0s) / np.median(f0s) + 1e-9)
    return bool(np.std(semis) > 0.25)


def assign_notes(pools: dict[str, list[dict]], harm: np.ndarray,
                 mid: np.ndarray, side: np.ndarray,
                 sr: int, vox: np.ndarray | None = None) -> dict[str, list[dict]]:
    """Assign each detected note to exactly one instrument.

    Melody (skyline) with vocal timbre -> vocals, else guitar lead.
    Chords with percussive decay -> piano, sustained -> guitar.

    Args:
        pools: Stem pools for piano/guitar/vocals/other.
        harm: Harmonic mono audio. mid/side: Stereo channels. sr: Rate.
        vox: Optional isolated voice audio for vibrato test.

    Returns:
        Mapping piano/guitar/vocals/other -> disjoint note lists.
    """
    test_audio = vox if vox is not None else harm
    merged = dedupe_pools(pools)
    melody = skyline_ids(merged)
    out: dict[str, list[dict]] = {"piano": [], "guitar": [],
                                  "vocals": [], "other": []}

    def _put(inst: str, note: dict) -> None:
        """Append note to instrument if in range, else to other.

        Args:
            inst: Target instrument. note: Note dict.
        """
        lo, hi = OUT_RANGES[inst]
        out[inst if lo <= note["pitch"] <= hi else "other"].append(note)

    # Grupos de onset: >=3 notas a la vez = acorde en bloque (piano),
    # notas sueltas = línea (guitarra/voz).
    starts = sorted(n["start"] for n in merged)
    import bisect as _bisect

    def _group_size(t: float) -> int:
        """Count notes starting within ±20 ms of t.

        Args:
            t: Onset time in seconds.

        Returns:
            Group size.
        """
        a = _bisect.bisect_left(starts, t - 0.02)
        b = _bisect.bisect_right(starts, t + 0.02)
        return b - a
    for i, n in enumerate(merged):
        dur = n["end"] - n["start"]
        decay = _decay_ratio(harm, sr, n["start"], n["end"])
        center = _center_ratio(mid, side, sr, n["start"], n["end"])
        note = {"pitch": n["pitch"], "start": n["start"], "end": n["end"],
                "velocity": n["velocity"]}
        if i in melody:
            sustained_note = decay < 2.5
            vocal_like = (center > 2.0 and sustained_note
                          and (dur > 0.35 or _has_vibrato(test_audio, sr, n["start"],
                                                         n["end"])))
            if vocal_like and 40 <= n["pitch"] <= 84:
                _put("vocals", note)
            else:
                _put("guitar", note)
        elif "other" in n["sources"] and len(n["sources"]) == 1:
            out["other"].append(note)
        elif (40 <= n["pitch"] <= 79 and dur > 0.25 and decay < 2.0):
            _put("vocals", note)  # línea cantada dentro de la textura
        elif decay < 1.3 and dur > 0.8:
            out["other"].append(note)  # pad/teclado ambiente, sin ataque
        elif _group_size(n["start"]) >= 3 and decay > 2.0:
            _put("piano", note)  # acorde en bloque
        elif decay > 3.0 or dur < 0.3:
            _put("guitar", note)  # línea pulsada suelta
        elif decay < 1.8 or dur > 0.5:
            _put("guitar", note)  # línea sostenida
        else:
            _put("guitar", note)
    return out
