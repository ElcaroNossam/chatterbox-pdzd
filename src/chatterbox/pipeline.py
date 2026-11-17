"""Text → speech with a cloned voice: the full chatterbox-pdzd pipeline as a library.

Used by say.py (CLI) and serve.py (local job service). Steps:
normalize Russian text → split into short phrases → generate each phrase (soft lead-in,
length cap) → optional Whisper check (retakes, tail cut) → trim edges → join with pauses →
mute pauses / even level → write MP3 or WAV.
"""
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torchaudio

from . import asr_qa
from .postfx import trim_phrase, clean_track

VOWELS = set("аеёиоуыэюяaeiouy")
SPEECH_TOKENS_PER_SEC = 25  # S3 speech tokenizer rate
MAX_CHARS = 100  # short phrases are much more stable


@dataclass
class Settings:
    lang: str = "ru"
    emotion: float = 0.5  # 0.25 flat … 0.8+ lively
    pace: float = 0.5  # cfg_weight: 0.3 freer … 0.7 more measured
    steps: int = 10  # flow-matching decoder steps; 6 is ~30% faster on CPU
    pause: float = 0.35  # seconds between phrases
    retakes: int = 2  # extra attempts when Whisper hears swallowed words
    qa: bool = True  # Whisper check (tail cut + retakes); needs faster-whisper


@dataclass
class Result:
    seconds: float
    phrases: int
    scores: list = field(default_factory=list)


Progress = Callable[[int, int, str], None]


def load_model(threads: Optional[int] = None):
    torch.set_num_threads(threads or max(1, (os.cpu_count() or 4) - 2))
    from .mtl_tts import ChatterboxMultilingualTTS
    return ChatterboxMultilingualTTS.from_pretrained(device="cpu")


def split_phrases(text: str, lang: str = "ru") -> list:
    if lang == "ru":
        from .text_ru import normalize_ru
        text = normalize_ru(text)
    parts = re.split(r"(?<=[.!?…])\s+", re.sub(r"\s+", " ", text).strip())
    chunks = []
    for p in parts:
        while len(p) > MAX_CHARS:
            cut = p.rfind(",", 0, MAX_CHARS)
            cut = cut if cut > 40 else p.rfind(" ", 0, MAX_CHARS)
            if cut <= 0:
                cut = MAX_CHARS
            chunks.append(p[: cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            chunks.append(p)
    return [c for c in chunks if c]


def token_budget(chunk: str) -> int:
    """At most ~1 s per 3.2 syllables (about half normal speed), so the model can't run on."""
    syllables = max(3, sum(ch.lower() in VOWELS for ch in chunk))
    return int(SPEECH_TOKENS_PER_SEC * (1.0 + syllables / 3.2))


def set_voice(model, voice: Optional[str], emotion: float):
    """voice: WAV/MP3/… sample or a saved .pt profile; None keeps the model's default voice."""
    if not voice:
        return
    if voice.endswith(".pt"):
        from .mtl_tts import Conditionals
        model.conds = Conditionals.load(voice)
    else:
        model.prepare_conditionals(voice, exaggeration=emotion)


def synthesize(model, text: str, s: Settings, progress: Optional[Progress] = None) -> tuple:
    """Returns (wav tensor [1, n], sample_rate, Result)."""
    chunks = split_phrases(text, s.lang)
    if not chunks:
        raise ValueError("пустой текст")
    use_qa = s.qa and asr_qa.available()
    pieces, scores = [], []
    gap = torch.zeros(1, int(model.sr * s.pause))
    for i, c in enumerate(chunks, 1):
        if progress:
            progress(i - 1, len(chunks), c)
        best = None
        for _take in range(1 + (s.retakes if use_qa else 0)):
            # a soft lead-in («… ») stops the model from swallowing the first consonant
            wav = model.generate("… " + c, language_id=s.lang, exaggeration=s.emotion, cfg_weight=s.pace,
                                 n_cfm_timesteps=s.steps, max_new_tokens=token_budget(c)).cpu()
            if not use_qa:
                best = (1.0, wav, None)
                break
            score, end_s, _heard = asr_qa.check(wav, model.sr, c, s.lang)
            if best is None or score > best[0]:
                best = (score, wav, end_s)
            if score >= 0.9:
                break
        score, wav, end_s = best
        if end_s is not None and wav.shape[1] / model.sr - end_s > 0.25:
            wav = wav[:, : int((end_s + 0.15) * model.sr)]  # cut the model's tail after the last word
        pieces += [trim_phrase(wav, model.sr), gap]
        scores.append(round(float(score), 2))
    if progress:
        progress(len(chunks), len(chunks), "")
    out = clean_track(torch.cat(pieces[:-1], dim=1), model.sr)
    return out, model.sr, Result(seconds=out.shape[1] / model.sr, phrases=len(chunks), scores=scores)


def save_audio(wav, sr: int, path: str) -> float:
    """Write WAV or MP3 (192k); returns the duration of the written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw = path.rsplit(".", 1)[0] + f".tmp{os.getpid()}.wav"
    torchaudio.save(raw, wav, sr)
    codec = [] if path.endswith(".wav") else ["-codec:a", "libmp3lame", "-b:a", "192k"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, *codec, path], check=True)
    os.remove(raw)
    return probe_seconds(path)


def probe_seconds(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def prepare_reference(src: str, dst: str, max_seconds: float = 15.0) -> float:
    """Turn any recording (webm/m4a/mp3/wav) into a clean 24 kHz mono reference for cloning:
    leading silence removed, low rumble cut, level evened, at most `max_seconds` long.
    Returns the length of usable audio in seconds."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    chain = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,highpass=f=70,loudnorm=I=-20:TP=-2"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", chain, "-t", str(max_seconds),
                    "-ar", "24000", "-ac", "1", dst], check=True)
    return probe_seconds(dst)


def timed(fn, *args, **kwargs):
    t = time.time()
    r = fn(*args, **kwargs)
    return r, time.time() - t
