"""Озвучка текста клоном голоса (chatterbox-pdzd, CPU). Тонкая обёртка над chatterbox.pipeline.

  .venv/bin/python say.py --voice voice.wav --text-file text.txt --out out.mp3
  .venv/bin/python say.py --voice voices/pdzd.pt --text "Привет" --out out.mp3
  .venv/bin/python say.py --save-voice voices/pdzd.pt --voice voice.wav   # один раз сохранить профиль голоса

  --voice     образец WAV/m4a или готовый профиль .pt (без --voice — стандартный голос модели)
  --emotion   0.25 ровно/диктор … 0.5 норма … 0.8+ живо
  --pace      cfg_weight: 0.3 свободнее/быстрее … 0.5 норма … 0.7 размереннее и чётче
  --steps     шаги декодера звука: 10 = как в оригинале, 6 = быстрее на CPU
  Ударения вручную: «м+ука», «зам+ок». Числа, сокращения и разметка листов записи обрабатываются сами.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--voice"); ap.add_argument("--save-voice")
ap.add_argument("--text"); ap.add_argument("--text-file")
ap.add_argument("--out")
ap.add_argument("--lang", default="ru")
ap.add_argument("--emotion", type=float, default=0.5)
ap.add_argument("--pace", type=float, default=0.5)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--pause", type=float, default=0.35)
ap.add_argument("--retakes", type=int, default=2, help="extra attempts when words are swallowed (needs faster-whisper)")
ap.add_argument("--no-qa", action="store_true", help="skip the Whisper check (faster, no tail cut / retakes)")
a = ap.parse_args()

from chatterbox import pipeline as P

t0 = time.time()
model = P.load_model()
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
P.set_voice(model, a.voice, a.emotion)
if a.save_voice:
    os.makedirs(os.path.dirname(a.save_voice) or ".", exist_ok=True)
    model.conds.save(a.save_voice)
    print(f"voice profile saved: {a.save_voice}")
    if not (a.text or a.text_file):
        raise SystemExit(0)

text = a.text or open(a.text_file, encoding="utf-8").read()
s = P.Settings(lang=a.lang, emotion=a.emotion, pace=a.pace, steps=a.steps, pause=a.pause,
               retakes=a.retakes, qa=not a.no_qa)
wav, sr, res = P.synthesize(model, text, s, progress=lambda i, n, c: print(f"[{i}/{n}] {c[:60]}", flush=True) if c else None)
secs = P.save_audio(wav, sr, a.out)
print(f"DONE {a.out}  {secs:.1f}s audio  {res.phrases} phrases  scores {res.scores}  total {time.time()-t0:.0f}s")
