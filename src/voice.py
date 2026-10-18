"""Separación voz vs acompañamiento por máscara armónica (numpy)."""
import numpy as np

from .dsp import istft, stft

FMIN = 80.0
FMAX = 1200.0
HOP_F0 = 1024
WIN_F0 = 2048


def _fft_autocorr_f0(frame: np.ndarray, sr: int) -> tuple[float | None, float]:
    """F0 via FFT autocorrelation peak + confidence.

    Args:
        frame: Windowed audio frame.
        sr: Sample rate.

    Returns:
        Tuple (F0 Hz or None, voiced confidence 0..1).
    """
    if float(np.mean(frame ** 2)) < 1e-7:
        return None, 0.0
    f = frame - frame.mean()
    n = 1
    while n < 2 * len(f):
        n *= 2
    spec = np.fft.rfft(f, n)
    corr = np.fft.irfft(np.abs(spec) ** 2, n)[:len(f)]
    if corr[0] <= 0:
        return None, 0.0
    corr = corr / corr[0]
    lo = max(1, int(sr / FMAX))
    hi = min(len(corr) - 1, int(sr / FMIN))
    peak = lo + int(np.argmax(corr[lo:hi]))
    return float(sr / peak), float(corr[peak])


def predominant_f0(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Track predominant pitched F0 via spectral harmonic summation.

    Args:
        y: Mono samples. sr: Sample rate.

    Returns:
        Tuple (smoothed f0 array Hz, voiced mask) per 1024-hop frame.
    """
    from .dsp import stft as _stft

    n_fft, hop = 4096, 1024
    spec = _stft(np.asarray(y, dtype=np.float64), n_fft, hop)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bin_w = freqs[1] - freqs[0]
    cand_midi = np.arange(40.0, 92.5, 0.5)
    cand_f = 440.0 * 2.0 ** ((cand_midi - 69.0) / 12.0)
    n_frames = mag.shape[1]
    f0 = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    for j in range(n_frames):
        frame = mag[:, j]
        total = float(frame[(freqs >= 80) & (freqs <= 6000)].sum()) + 1e-9
        scores = np.zeros(len(cand_f))
        for c, f in enumerate(cand_f):
            s = 0.0
            k = 1
            while k * f <= 6000.0 and k <= 8:
                pos = k * f / bin_w
                i0 = int(pos)
                if i0 + 1 < len(frame):
                    frac = pos - i0
                    s += ((1 - frac) * frame[i0] + frac * frame[i0 + 1]) / np.sqrt(k)
                k += 1
            scores[c] = s
        top = scores.max()
        if top < 0.06 * total:
            continue
        tied = [c for c in range(len(cand_f)) if scores[c] >= 0.85 * top]
        pick = tied[-1]  # ante el empate, la F0 más aguda (suele ser la voz)
        f0[j] = float(cand_f[pick])
        voiced[j] = True
    # Continuidad: mediana + romper saltos mayores de octava.
    sm = f0.copy()
    for j in range(n_frames):
        a, b = max(0, j - 2), min(n_frames, j + 3)
        seg = f0[a:b][voiced[a:b]]
        if len(seg):
            sm[j] = float(np.median(seg))
    keep = np.zeros(n_frames, dtype=bool)
    for j in range(n_frames):
        if voiced[j] and abs(np.log2(sm[j] / max(f0[j], 1e-6))) < 1.0:
            keep[j] = True
            f0[j] = sm[j]
    return f0, keep


def _salience_scores(mag: np.ndarray, cand_bin: np.ndarray,
                     weights: np.ndarray) -> np.ndarray:
    """Harmonic-summation salience for all candidates, one frame.

    Args:
        mag: Magnitude spectrum. cand_bin: (C,H) fractional bin positions
            (-1 = out of range). weights: (H,) harmonic weights.

    Returns:
        Salience array (C,).
    """
    lo = np.floor(cand_bin).astype(int)
    lo = np.clip(lo, 0, len(mag) - 1)
    hi = np.clip(lo + 1, 0, len(mag) - 1)
    fr = (cand_bin - np.floor(cand_bin)) * (cand_bin >= 0)
    vals = mag[lo] * (1 - fr) + mag[hi] * fr
    vals = np.where(cand_bin >= 0, vals, 0.0)
    return (vals * weights[None, :]).sum(axis=1)


def _periodic_vibrato(detr: np.ndarray, frame_dur: float) -> bool:
    """Check wobble is quasi-periodic in 3-9 Hz (singing vibrato).

    Args:
        detr: Detrended semitone series. frame_dur: Seconds per frame.

    Returns:
        True if autocorrelation peaks in the vibrato lag range.
    """
    x = np.asarray(detr, dtype=np.float64)
    x = x - x.mean()
    if len(x) < 8 or float(np.var(x)) < 0.02:
        return False
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return False
    ac = ac / ac[0]
    lo = max(1, int(round(1.0 / (9.0 * frame_dur))))
    hi = min(len(ac) - 1, int(round(1.0 / (3.0 * frame_dur))))
    if hi <= lo:
        return False
    return bool(ac[lo:hi + 1].max() > 0.45)


def _attack_slow(yy: np.ndarray, sr: int, t0: float) -> bool:
    """Check if onset rises slowly (syllable ~30ms+) vs snap (pluck ~10ms).

    Uses rise time from 20% to peak, robust to prior ringing.

    Args:
        yy: Mono audio. sr: Rate. t0: Onset time in seconds.

    Returns:
        True if soft voice-like onset.
    """
    i0 = int(t0 * sr)
    n = len(yy)
    a = max(0, i0 - int(0.10 * sr))
    b = min(n, i0 + int(0.15 * sr))
    seg = np.abs(np.asarray(yy[a:b], dtype=np.float64))
    if len(seg) < 10:
        return False
    k = max(1, int(0.002 * sr))
    env = np.convolve(seg, np.ones(k) / k, mode="same")
    floor = float(np.median(env)) + 1e-9
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i])
    if peak < 3 * floor:
        return False
    below = np.where(env[:peak_i] < 0.2 * peak)[0]
    if len(below) == 0:
        return True  # ya venía subiendo (legato) -> blando
    return bool((peak_i - below[-1]) / sr > 0.025)


def melody_f0(y: np.ndarray, sr: int,
              side: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Track sung melody: salience contours that are vibrato-led or
    soft-attack sustained movers (voice-like). Optionally gated by
    stereo centeredness.

    Args:
        y: Mono samples (mid channel). sr: Sample rate.
        side: Optional side channel for centeredness cue.

    Returns:
        Tuple (f0 per 1024-hop frame, voiced mask).
    """
    """Track sung melody via salience contours + vibrato selection.

    Args:
        y: Mono samples. sr: Sample rate.

    Returns:
        Tuple (f0 per 1024-hop frame, 0 = unvoiced; voiced mask).
    """
    from .dsp import stft as _stft

    n_fft, hop = 4096, 1024
    spec = _stft(np.asarray(y, dtype=np.float64), n_fft, hop)
    mag = np.abs(spec)
    bin_w = sr / n_fft
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    cand_midi = np.arange(36.0, 96.5, 0.5)
    cand_f = 440.0 * 2.0 ** ((cand_midi - 69.0) / 12.0)
    harm_k = np.arange(1, 9)
    pos = cand_f[:, None] * harm_k[None, :] / bin_w
    pos = np.where((cand_f[:, None] * harm_k[None, :] <= 6000.0)
                   & (pos < mag.shape[0] - 1), pos, -1.0)
    weights = np.array([2.0] + [1 / np.sqrt(k) for k in range(2, 9)])
    n_frames = mag.shape[1]
    # Picos de saliencia por frame (se guarda la saliencia para refinar F0).
    peaks: list[list[tuple[int, float]]] = []
    salience: list[np.ndarray] = []
    for j in range(n_frames):
        s = _salience_scores(mag[:, j], pos, weights)
        salience.append(s)
        mx = s.max()
        if mx <= 0:
            peaks.append([])
            continue
        frame = mag[:, j]
        med = float(np.median(frame[(freqs >= 80) & (freqs <= 6000)])) + 1e-9
        idx = []
        for i in range(2, len(s) - 2):
            if not (s[i] > s[i - 1] and s[i] >= s[i + 1]
                    and s[i] > s[i - 2] and s[i] >= s[i + 2]
                    and s[i] > 0.15 * mx):
                continue
            f = float(cand_f[i])
            b = int(f / bin_w)
            if b < len(frame) and frame[b] > 1.25 * med:
                idx.append(i)
        peaks.append([(i, float(s[i])) for i in idx])
    # Une picos en contornos (tolerancia 1 semitono = 2 celdas,
    # hueco máx 2 frames para no fragmentar el vibrato).
    contours: list[dict] = []
    open_c: list[int] = []
    for j, pk in enumerate(peaks):
        used = set()
        order = sorted(range(len(open_c)),
                       key=lambda o: -contours[open_c[o]]["sal"])
        for o in order:
            c = contours[open_c[o]]
            if j - c["last_j"] > 2:
                continue
            best, bi = 3, -1
            for pi, (ci, _) in enumerate(pk):
                if pi in used:
                    continue
                d = abs(ci - c["last_i"])
                if d < best:
                    best, bi = d, pi
            if bi >= 0:
                ci, s = pk[bi]
                used.add(bi)
                c["pts"].append((j, ci))
                c["sal"] += s
                c["last_j"], c["last_i"] = j, ci
        for pi, (ci, s) in enumerate(pk):
            if pi not in used:
                contours.append({"pts": [(j, ci)], "sal": s,
                                 "last_j": j, "last_i": ci})
                open_c.append(len(contours) - 1)
        open_c = [o for o in open_c if j - contours[o]["last_j"] <= 2]
    # Parte contornos en saltos > 2 semitonos (= cambio de fuente).
    split: list[dict] = []
    for c in contours:
        cur: list[tuple[int, int]] = []
        for p in c["pts"]:
            if cur and abs(p[1] - cur[-1][1]) > 4:
                split.append({"pts": cur, "sal": c["sal"]})
                cur = []
            cur.append(p)
        if cur:
            split.append({"pts": cur, "sal": c["sal"]})
    contours = split
    # Puntúa contornos como voz (ver docstring). Con side, el centro manda.
    yy = np.asarray(y, dtype=np.float64)
    center_j: np.ndarray | None = None
    if side is not None and len(side) == len(y):
        smag = np.abs(stft(np.asarray(side, dtype=np.float64), 2048, hop))
        mmag = np.abs(stft(yy, 2048, hop))
        nfr = min(smag.shape[1], mmag.shape[1])
        ce = mmag[:, :nfr].sum(axis=0) / (smag[:, :nfr].sum(axis=0) + 1e-9)
        center_j = ce
    f0 = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)

    def _refined(j: int, ci: int) -> float:
        """Parabolic refinement of candidate frequency (sub-cell).

        Args:
            j: Frame index. ci: Candidate index.

        Returns:
            Refined frequency in Hz.
        """
        s = salience[j]
        if 1 <= ci < len(s) - 1:
            d = s[ci - 1] - 2 * s[ci] + s[ci + 1]
            off = 0.5 * (s[ci - 1] - s[ci + 1]) / d if d != 0 else 0.0
            off = float(np.clip(off, -0.5, 0.5))
        else:
            off = 0.0
        return float(cand_f[ci] * 2.0 ** (off * 0.5 / 12.0))

    # Encadena fragmentos cercanos en pitch (el vibrato/fraseo se ve en
    # tramos largos) y clasifica cadenas, no fragmentos sueltos.
    chains: list[list[dict]] = []
    for c in sorted(contours, key=lambda c: c["pts"][0][0]):
        med_c = float(np.median([cand_f[i] for _, i in c["pts"]]))
        placed = False
        for ch in chains:
            last = ch[-1]
            gap = (c["pts"][0][0] - last["pts"][-1][0]) * hop / sr
            med_l = float(np.median([cand_f[i] for _, i in last["pts"]]))
            if gap < 0.4 and abs(12 * np.log2(med_c / max(med_l, 1e-6))) < 1.5:
                ch.append(c)
                placed = True
                break
        if not placed:
            chains.append([c])
    mean_sals = [c["sal"] / max(1, len(c["pts"])) for c in contours]
    sal_floor = float(np.median(mean_sals)) if mean_sals else 0.0
    for ch in chains:
        pts = sorted([p for c in ch for p in c["pts"]])
        fs = np.array([_refined(j, ci) for j, ci in pts])
        fref = {j: f for (j, _), f in zip(pts, fs)}
        dur = (pts[-1][0] - pts[0][0] + 1) * hop / sr
        if dur < 0.3:
            continue
        semis = 12 * np.log2(fs / np.median(fs) + 1e-9)
        detr = semis - np.linspace(semis[0], semis[-1], len(semis))
        vib = bool(np.std(detr) > 0.28
                   and _periodic_vibrato(detr, hop / sr))
        moves = bool(float(semis.max() - semis.min()) > 1.5)
        j0, j1 = pts[0][0], pts[-1][0]
        s0 = yy[j0 * hop:j0 * hop + max(1, (j1 - j0) * hop // 4)]
        s1 = yy[max(j0 * hop, j1 * hop - max(1, (j1 - j0) * hop // 4)):j1 * hop]
        decay = (float(np.sqrt(np.mean(s0 ** 2)) + 1e-9)
                 / float(np.sqrt(np.mean(s1 ** 2)) + 1e-9)) if len(s1) else 9.0
        sustained = decay < 2.0
        slow = _attack_slow(yy, sr, pts[0][0] * hop / sr)
        msal = sum(c["sal"] for c in ch) / max(1, sum(len(c["pts"]) for c in ch))
        vocal = bool((vib or (slow and sustained and moves))
                     and msal > 0.75 * sal_floor)
        if vocal and center_j is not None:
            js = [j for j, _ in pts if j < len(center_j)]
            centered = bool(np.median(center_j[js]) > 3.0) if js else False
            vocal = bool(centered or vib)
        if vocal:
            for j, _ in pts:
                f0[j] = float(fref[j])
                voiced[j] = True
    return f0, voiced


def _vibrato_regions(f0: np.ndarray, voiced: np.ndarray,
                     hop_sec: float) -> np.ndarray:
    """Mark frames whose F0 wobbles like a singing voice.

    Args:
        f0: Smoothed F0 track. voiced: Voiced mask.
        hop_sec: Frame hop in seconds.

    Returns:
        Boolean mask of voice-like frames.
    """
    n = len(f0)
    win = max(3, int(0.25 / hop_sec))
    score = np.zeros(n, dtype=bool)
    semis = np.zeros(n)
    med = np.median(f0[voiced]) if voiced.any() else 440.0
    semis[voiced] = 12 * np.log2(f0[voiced] / max(med, 1e-6) + 1e-9)
    for i in range(n):
        if not voiced[i]:
            continue
        a, b = max(0, i - win), min(n, i + win + 1)
        seg = semis[a:b][voiced[a:b]]
        if len(seg) >= 5:
            detr = seg - np.linspace(seg[0], seg[-1], len(seg))
            if float(np.std(detr)) > 0.18:
                score[i] = True
    # Cierra huecos < 0.3s (solo donde hay señal) y elimina islas cortas.
    gap = max(1, int(0.3 / hop_sec))
    dil = np.zeros(n, dtype=bool)
    idx = np.where(score)[0]
    for i in idx:
        a, b = max(0, i - gap), min(n, i + gap + 1)
        dil[a:b] = True
    dil &= voiced
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if dil[i]:
            j = i
            while j < n and dil[j]:
                j += 1
            if j - i >= gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def split_voice(y: np.ndarray, sr: int, side: np.ndarray | None = None,
                n_fft: int = 4096,
                hop: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:
    """Split mono audio into voice (melody mask) and backing.

    Args:
        y: Mono samples (mid channel). sr: Sample rate.
        side: Optional side channel for centeredness cue.
        n_fft: Mask STFT size. hop: Mask hop.

    Returns:
        Tuple (voice, backing, voiced_fraction).
    """
    f0, is_voice = melody_f0(y, sr, side=side)
    # F0 y máscara comparten hop 1024: correspondencia directa.
    spec = stft(y, n_fft, hop)
    mag = np.abs(spec).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = np.zeros_like(mag)
    sigma = 45.0
    for j in range(spec.shape[1]):
        i = min(j, len(f0) - 1)
        if not is_voice[i] or f0[i] <= 0:
            continue
        f = f0[i]
        ks = np.arange(1, int(6000 // f) + 1) * f
        dist = np.abs(freqs[:, None] - ks[None, :]).min(axis=1)
        mask[:, j] = np.exp(-0.5 * (dist / sigma) ** 2)
    phase = np.exp(1j * np.angle(spec)).astype(np.complex64)
    mag32 = mag.astype(np.float32)
    voice = istft(mag32 * mask.astype(np.float32) * phase, n_fft, hop, len(y))
    backing = (y - voice).astype(np.float32)
    return voice, backing, float(is_voice.mean())
