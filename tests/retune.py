"""Re-run the clean-up on dumped raw phrases: python retune.py DIR out.wav [pause_s]"""
import glob, sys, torch, torchaudio
from chatterbox.postfx import trim_phrase, clean_track
d, out = sys.argv[1], sys.argv[2]
pause = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
pieces = []
for f in sorted(glob.glob(f"{d}/*.wav")):
    w, sr = torchaudio.load(f)
    t = trim_phrase(w, sr)
    print(f"{f.rsplit('/',1)[-1]}: {w.shape[1]/sr:.2f}s -> {t.shape[1]/sr:.2f}s")
    pieces += [t, torch.zeros(1, int(sr * pause))]
y = clean_track(torch.cat(pieces[:-1], dim=1), sr)
torchaudio.save(out, y, sr, encoding="PCM_S", bits_per_sample=16)
print(f"{out}: {y.shape[1]/sr:.2f}s")
