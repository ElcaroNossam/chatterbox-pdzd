"""Speech QA for generated phrases (chatterbox-pdzd), optional: needs faster-whisper.

The model can't tell us reliably where the text ended: after the last word it sometimes keeps
breathing/mumbling for up to ~2 s, and sometimes it swallows or slurs a word. A small local
Whisper listens to each phrase and returns:
  score    - share of the requested words that were actually heard (fuzzy, 0..1)
  end_s    - where the last heard word ends (to cut the tail right after it)
"""
import difflib
import re

_model = None


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model(size: str = "small"):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def _words(text: str, lang: str):
    if lang == "ru":
        from .text_ru import normalize_ru
        text = normalize_ru(text)
    text = text.lower().replace("ё", "е").replace("+", "")
    return re.findall(r"[\w]+", text)


def word_score(target: str, heard: str, lang: str = "ru") -> float:
    want, got = _words(target, lang), _words(heard, lang)
    if not want:
        return 1.0
    hit, j = 0, 0
    for w in want:  # in order, allow skipping a few heard words
        best, best_k = 0.0, None
        for k in range(j, min(len(got), j + 4)):
            r = difflib.SequenceMatcher(None, w, got[k]).ratio()
            if r > best:
                best, best_k = r, k
        if best >= 0.75:
            hit += 1
            j = best_k + 1
    return hit / len(want)


def check(wav, sr: int, text: str, lang: str = "ru"):
    """wav: (1, N) float tensor. Returns (score, end_s, heard_text)."""
    import torch
    import torchaudio
    x = wav[0]
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    segs, _ = _get_model().transcribe(x.numpy().astype("float32"), language=lang, beam_size=5,
                                      vad_filter=False, word_timestamps=True,
                                      condition_on_previous_text=False)
    words = [w for s in segs for w in (s.words or [])]
    heard = " ".join(w.word.strip() for w in words)
    end_s = words[-1].end if words else None
    return word_score(text, heard, lang), end_s, heard
