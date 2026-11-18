"""Local voice job service for SoundRep (chatterbox-pdzd).

Renders jobs one at a time (the CPU cannot do two at once). The model is loaded on the first
job and unloaded after $PDZD_IDLE_SECONDS (default 900) without work, so it holds ~7 GB of RAM
only while someone is actually using it.
Listens on 127.0.0.1 only; every request must carry X-Worker-Token = $PDZD_WORKER_TOKEN, and
every file path must lie inside $PDZD_ALLOWED_ROOT (the SoundRep data directory).

  PDZD_WORKER_TOKEN=... PDZD_ALLOWED_ROOT=/path/to/soundrep/data .venv/bin/python serve.py [--port 7861]

  GET  /health                         → {ready, queue, running}
  POST /prepare {src, dst}             → {seconds}   make a clean reference WAV from a recording
  POST /jobs {id, voice, text, out, settings} → {status, position}
  GET  /jobs/<id>                      → {status, position, done, total, seconds, error, scores}
"""
import argparse
import ctypes
import gc
import hmac
import json
import os
import queue
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chatterbox import pipeline as P

TOKEN = os.environ.get("PDZD_WORKER_TOKEN", "")
ROOT = os.path.realpath(os.environ.get("PDZD_ALLOWED_ROOT", "/nonexistent"))
MAX_TEXT = 1200
IDLE_SECONDS = int(os.environ.get("PDZD_IDLE_SECONDS", "900"))  # unload the model after this much idle time
ID_RE = re.compile(r"^[0-9a-f-]{36}$")

jobs: dict = {}  # id → state (in memory; a restart drops unfinished jobs and SoundRep reports them)
order: list = []
work: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()
state = {"ready": True, "loaded": False, "loading": False, "running": None, "error": None}


def inside_root(path: str) -> str:
    real = os.path.realpath(path)
    if not (real == ROOT or real.startswith(ROOT + os.sep)):
        raise ValueError("path outside the allowed root")
    return real


def position(job_id: str) -> int:
    with lock:
        waiting = [j for j in order if jobs[j]["status"] == "queued"]
    return waiting.index(job_id) + 1 if job_id in waiting else 0


def free_memory():
    """Give the model's RAM back to the OS (Python alone keeps freed blocks in its heap)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def worker():
    """Model is loaded on the first job and unloaded after IDLE_SECONDS without work,
    so the service costs ~0 RAM while nobody is using it."""
    model = None
    while True:
        try:
            job_id = work.get(timeout=IDLE_SECONDS if model is not None else None)
        except queue.Empty:
            # idle: unload the TTS model and Whisper, return the memory
            model = None
            from chatterbox import asr_qa
            asr_qa._model = None
            free_memory()
            state["loaded"] = False
            print(f"[serve] idle {IDLE_SECONDS}s — model unloaded", flush=True)
            continue
        job = jobs.get(job_id)
        if not job:
            continue
        state["running"] = job_id
        job.update(status="running", started=time.time())
        try:
            if model is None:
                job["loading"] = True
                state["loading"] = True
                t = time.time()
                model = P.load_model()
                state.update(loaded=True, loading=False, error=None)
                job["loading"] = False
                print(f"[serve] model loaded in {time.time() - t:.0f}s", flush=True)
            s = P.Settings(**job["settings"])
            P.set_voice(model, job["voice"], s.emotion)

            def progress(done, total, _chunk, job=job):
                job.update(done=done, total=total)

            wav, sr, res = P.synthesize(model, job["text"], s, progress=progress)
            job["seconds"] = P.save_audio(wav, sr, job["out"])
            job.update(status="done", scores=res.scores)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            state["loading"] = False
            job.update(status="error", loading=False, error=str(e)[:300])
        finally:
            job["finished"] = time.time()
            state["running"] = None
            with lock:  # keep the last 500 jobs
                while len(order) > 500:
                    jobs.pop(order.pop(0), None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter log
        print(f"[serve] {self.command} {self.path} → {args[1] if len(args) > 1 else ''}", flush=True)

    def reply(self, code: int, body: dict):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self) -> bool:
        if TOKEN and hmac.compare_digest(self.headers.get("X-Worker-Token", ""), TOKEN):
            return True
        self.reply(401, {"error": "unauthorized"})
        return False

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 64_000:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/health":
            return self.reply(200, {"ready": state["ready"], "loaded": state["loaded"], "loading": state["loading"],
                                    "error": state["error"], "queue": work.qsize(), "running": state["running"]})
        m = re.match(r"^/jobs/([0-9a-f-]{36})$", self.path)
        if m:
            job = jobs.get(m.group(1))
            if not job:
                return self.reply(404, {"error": "unknown job"})
            return self.reply(200, {k: job.get(k) for k in ("status", "done", "total", "seconds", "error", "scores", "loading")}
                              | {"position": position(m.group(1))})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if not self.authorized():
            return
        try:
            b = self.body()
            if self.path == "/prepare":
                src, dst = inside_root(b["src"]), inside_root(b["dst"])
                return self.reply(200, {"seconds": P.prepare_reference(src, dst)})
            if self.path == "/jobs":
                job_id = str(b["id"])
                if not ID_RE.match(job_id) or job_id in jobs:
                    return self.reply(400, {"error": "bad or duplicate id"})
                text = str(b["text"])[:MAX_TEXT]
                allowed = {k: b.get("settings", {})[k] for k in ("emotion", "pace", "steps", "retakes", "qa", "pause")
                           if k in b.get("settings", {})}
                job = {"status": "queued", "done": 0, "total": 0, "voice": inside_root(b["voice"]),
                       "out": inside_root(b["out"]), "text": text, "settings": allowed, "created": time.time()}
                with lock:
                    jobs[job_id] = job
                    order.append(job_id)
                work.put(job_id)
                return self.reply(202, {"status": "queued", "position": position(job_id)})
            self.reply(404, {"error": "not found"})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.reply(400, {"error": str(e)[:200]})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.reply(500, {"error": str(e)[:200]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    if not TOKEN or not os.path.isdir(ROOT):
        raise SystemExit("set PDZD_WORKER_TOKEN and PDZD_ALLOWED_ROOT (an existing directory)")
    threading.Thread(target=worker, daemon=True).start()
    print(f"voice service on 127.0.0.1:{a.port}, root {ROOT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
