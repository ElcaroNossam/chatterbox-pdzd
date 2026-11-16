"""Озвучка текста клоном голоса (chatterbox-pdzd, CPU).

  .venv/bin/python say.py --voice voice.wav --text-file text.txt --out out.mp3
  .venv/bin/python say.py --voice voices/pdzd.pt --text "Привет" --out out.mp3
  .venv/bin/python say.py --save-voice voices/pdzd.pt --voice voice.wav   # один раз сохранить профиль голоса

  --voice     образец WAV/m4a или готовый профиль .pt (без --voice — стандартный голос модели)
  --emotion   0.25 ровно/диктор … 0.5 норма … 0.8+ живо
  --pace      cfg_weight: 0.3 свободнее/быстрее … 0.5 норма … 0.7 размереннее и чётче
  --steps     шаги декодера звука: 10 = как в оригинале, 6 = быстрее на CPU
  Ударения вручную: «м+ука», «зам+ок». Числа, сокращения и разметка листов записи обрабатываются сами.
"""
import argparse, os, re, subprocess, time
import torch, torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--voice"); ap.add_argument("--save-voice")
ap.add_argument("--text"); ap.add_argument("--text-file")
ap.add_argument("--out")
ap.add_argument("--lang", default="ru")
ap.add_argument("--emotion", type=float, default=0.5)
ap.add_argument("--pace", type=float, default=0.5)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--pause", type=float, default=0.35)
a = ap.parse_args()

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
t0 = time.time()
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

# voice profile: computed once (pdzd fix), reusable as .pt
if a.voice and a.voice.endswith(".pt"):
    model.conds = Conditionals.load(a.voice)
elif a.voice:
    model.prepare_conditionals(a.voice, exaggeration=a.emotion)
if a.save_voice:
    os.makedirs(os.path.dirname(a.save_voice) or ".", exist_ok=True)
    model.conds.save(a.save_voice)
    print(f"voice profile saved: {a.save_voice}")
    if not (a.text or a.text_file):
        raise SystemExit(0)

text = a.text or open(a.text_file, encoding="utf-8").read()
if a.lang == "ru":
    from chatterbox.text_ru import normalize_ru
    text = normalize_ru(text)
parts = re.split(r"(?<=[.!?…])\s+", re.sub(r"\s+", " ", text).strip())
# short chunks are much more stable: split long sentences at commas (~100 chars max)
MAX_CHARS = 100
chunks = []
for p in parts:
    while len(p) > MAX_CHARS:
        cut = p.rfind(",", 0, MAX_CHARS); cut = cut if cut > 40 else p.rfind(" ", 0, MAX_CHARS)
        chunks.append(p[:cut + 1].strip()); p = p[cut + 1:].strip()
    if p:
        chunks.append(p)

VOWELS = set("аеёиоуыэюяaeiouy")
SPEECH_TOKENS_PER_SEC = 25  # S3 speech tokenizer rate


def token_budget(chunk: str) -> int:
    """Cap generation length: at most ~1 s per 3.2 syllables (about half the normal 5-6 syllables/s),
    so the model can't run on and invent extra speech after the text ends."""
    syllables = max(3, sum(ch.lower() in VOWELS for ch in chunk))
    return int(SPEECH_TOKENS_PER_SEC * (1.0 + syllables / 3.2))
print(f"{len(chunks)} фраз", flush=True)

pieces, gap = [], torch.zeros(1, int(model.sr * a.pause))
for i, c in enumerate(chunks, 1):
    t = time.time()
    wav = model.generate(c, language_id=a.lang, exaggeration=a.emotion, cfg_weight=a.pace,
                         n_cfm_timesteps=a.steps, max_new_tokens=token_budget(c))
    pieces += [wav.cpu(), gap]
    print(f"[{i}/{len(chunks)}] {time.time()-t:.0f}s for {wav.shape[1]/model.sr:.1f}s  {c[:60]}", flush=True)

out = torch.cat(pieces[:-1], dim=1)
tmp = a.out.rsplit(".", 1)[0] + ".wav"
torchaudio.save(tmp, out, model.sr)
if not a.out.endswith(".wav"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", "loudnorm=I=-16:TP=-1.5",
                    "-codec:a", "libmp3lame", "-b:a", "192k", a.out], check=True)
print(f"DONE {a.out}  {out.shape[1]/model.sr:.1f}s audio  total {time.time()-t0:.0f}s")
