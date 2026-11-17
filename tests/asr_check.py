"""ASR check of generated speech (separate env ~/ai/asr): python asr_check.py file.wav [start end]..."""
import sys
from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8")
for f in sys.argv[1:]:
    segs, _ = m.transcribe(f, language="ru", beam_size=5, vad_filter=False, word_timestamps=True)
    words = [w for s in segs for w in s.words]
    print(f"== {f}: " + " ".join(f"{w.word.strip()}[{w.start:.2f}-{w.end:.2f}]" for w in words), flush=True)
