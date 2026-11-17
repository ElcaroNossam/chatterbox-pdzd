"""Clean-up of generated speech (chatterbox-pdzd).

The model leaves three kinds of "noise" around words: breaths and hiss in its own pauses,
a quiet breath before a phrase, and sometimes a short mumble after the text has ended.
Speech is analysed in 10 ms frames relative to the phrase's speech level:
  pause   = a run of >= 150 ms where every frame is > PAUSE_DB below speech level
  segment = sound between pauses
Edge segments much quieter than speech (breath, mumble) are dropped, pauses are muted and
over-long pauses shortened. Gains are smoothed so word edges stay natural.
"""
import torch

HOP_S = 0.01
PAUSE_DB = 25.0        # frame this far below speech level counts as "quiet"
MIN_PAUSE_S = 0.15     # quiet run at least this long is a pause (stop consonants are < 100 ms)
JUNK_DB = 14.0         # an edge segment this much quieter than speech is breath/mumble, not words
EDGE_MARGIN_S = 0.04   # keep a little air around words


def _frames_db(x: torch.Tensor, hop: int) -> torch.Tensor:
    n = x.numel() // hop
    return 10 * torch.log10(x[: n * hop].view(n, hop).pow(2).mean(dim=1) + 1e-12)


def _speech_level(db: torch.Tensor) -> float:
    loud = db[db > db.max() - 30]
    return float(10 * torch.log10((10 ** (loud / 10)).mean()))


def _segments(db: torch.Tensor, level: float, min_pause: int):
    """Return [(start, end)] frame ranges of sound separated by pauses."""
    quiet = (db < level - PAUSE_DB).tolist()
    segs, i, n = [], 0, len(quiet)
    start = None
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= min_pause or i == 0 or j == n:
                if start is not None:
                    segs.append((start, i)); start = None
            i = j
        else:
            if start is None:
                start = i
            i += 1
    if start is not None:
        segs.append((start, n))
    return segs


def _seg_level(db: torch.Tensor, a: int, b: int) -> float:
    return float(10 * torch.log10((10 ** (db[a:b] / 10)).mean()))


def _fade(y: torch.Tensor, n: int) -> torch.Tensor:
    n = min(n, y.numel() // 4)
    if n > 0:
        r = torch.linspace(0, 1, n)
        y[:n] *= r; y[-n:] *= r.flip(0)
    return y


def trim_phrase(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Cut silence, the breath before the phrase and any mumble after it."""
    x = wav[0]
    hop = int(HOP_S * sr)
    if x.numel() < hop * 10:
        return wav
    db = _frames_db(x, hop)
    level = _speech_level(db)
    segs = _segments(db, level, int(MIN_PAUSE_S / HOP_S))
    if not segs:
        return wav
    while len(segs) > 1 and _seg_level(db, *segs[0]) < level - JUNK_DB:
        segs.pop(0)
    while len(segs) > 1 and _seg_level(db, *segs[-1]) < level - JUNK_DB:
        segs.pop()
    m = int(EDGE_MARGIN_S / HOP_S)
    a = max(0, segs[0][0] - m) * hop
    b = min(len(db), segs[-1][1] + m) * hop
    return _fade(x[a:b].clone(), int(0.015 * sr)).unsqueeze(0)


def clean_track(wav: torch.Tensor, sr: int, max_pause_s: float = 0.6, keep_pause_s: float = 0.45,
                target_db: float = -18.0, peak_db: float = -1.5) -> torch.Tensor:
    """Mute every pause (breath/hiss), shorten pauses > max_pause_s, then one linear gain
    to the target speech level (dynamic loudnorm would pump the hiss back up)."""
    x = wav[0]
    hop = int(HOP_S * sr)
    n = x.numel() // hop
    if n < 10:
        return wav
    db = _frames_db(x, hop)
    level = _speech_level(db)
    min_pause = int(MIN_PAUSE_S / HOP_S)
    m = int(EDGE_MARGIN_S / HOP_S)
    voiced = torch.zeros(n, dtype=torch.bool)
    for a, b in _segments(db, level, min_pause):
        voiced[max(0, a - m): min(n, b + m)] = True
    # smoothed gain: 1 on sound, -60 dB in pauses; 20 ms ramps
    g = voiced.float() * (1 - 1e-3) + 1e-3
    k = 2
    g = torch.nn.functional.avg_pool1d(g.view(1, 1, -1), 2 * k + 1, stride=1, padding=k,
                                       count_include_pad=False).view(-1)
    gain = torch.nn.functional.interpolate(g.view(1, 1, -1), size=n * hop, mode="linear",
                                           align_corners=False).view(-1)
    y = (x[: n * hop] * gain).view(n, hop)
    # shorten long pauses (cut from the middle, the ramps stay)
    keep = torch.ones(n, dtype=torch.bool)
    i = 0
    max_p, keep_p = int(max_pause_s / HOP_S), int(keep_pause_s / HOP_S)
    while i < n:
        if not voiced[i]:
            j = i
            while j < n and not voiced[j]:
                j += 1
            if j - i > max_p:
                keep[i + keep_p // 2: j - (keep_p - keep_p // 2)] = False
            i = j
        else:
            i += 1
    y = y[keep].reshape(-1)
    g_lin = 10 ** ((target_db - level) / 20)
    peak = float(y.abs().max()) * g_lin
    if peak > 10 ** (peak_db / 20):
        g_lin *= 10 ** (peak_db / 20) / peak
    return (y * g_lin).unsqueeze(0)
