"""ATF JSON-lines backend for GUI frontends.

Protocol (one JSON object per line on stdin):
  {"prompt": "...", "max_tokens": 256, "temperature": 0.7,
   "top_p": 0.9, "thinking": "off"}              -> run one generation
  {"stop": true}                                  -> abort current generation

stdout events (one JSON object per line):
  {"type": "ready"}
  {"type": "start", "prompt_tokens": N}
  {"type": "token", "t": " Hel"}                 -> one decoded token
  {"type": "done", "text": "...", "tok_s": 4.1, "stopped": false}
  {"type": "error", "message": "..."}

Usage:
  python -m atf.serve models/Qwen3.5-9B-BF16.atf

The model loads once at startup and stays GPU-resident. Generation runs in a
worker thread so {"stop": true} is honored mid-stream.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from .engine import Engine, GenConfig
from .model import load_atf
from .tokenizer import Tokenizer


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class StopFlag:
    def __init__(self):
        self.event = threading.Event()

    def set(self):
        self.event.set()

    def clear(self):
        self.event.clear()

    def check(self):
        if self.event.is_set():
            raise _Stopped()


class _Stopped(Exception):
    pass


def main() -> None:
    if len(sys.argv) < 2:
        _emit({"type": "error", "message": "usage: python -m atf.serve MODEL.atf"})
        sys.exit(1)

    try:
        model = load_atf(Path(sys.argv[1]))
        tok = Tokenizer.from_hf("Qwen/Qwen3.5-9B")
        engine = Engine(model, tok)
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "error", "message": str(exc)})
        sys.exit(1)
    _emit({"type": "ready"})

    stop = StopFlag()
    lock = threading.Lock()
    state = {"text": "", "n": 0, "error": None}

    def run_generation(req: dict) -> None:
        cfg = GenConfig(
            max_tokens=int(req.get("max_tokens", 256)),
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 0.9)),
            thinking=req.get("thinking", "off"),
            seed=req.get("seed"),
        )
        prompt = req.get("prompt", "")
        n_prompt = len(tok.encode(prompt))
        state.update(text="", n=0, error=None)
        t0 = time.time()

        def on_token(tid: int) -> None:
            stop.check()
            text = tok.decode_token(tid)
            state["text"] += text
            state["n"] += 1
            _emit({"type": "token", "t": text})

        _emit({"type": "start", "prompt_tokens": n_prompt})
        text = engine.generate(prompt, cfg, stream=False, token_cb=on_token)
        gen_t = time.time() - t0
        stopped = stop.event.is_set()
        stop.clear()
        _emit({"type": "done", "text": state["text"] or text,
               "tok_s": round(state["n"] / gen_t, 2) if gen_t > 0 else None,
               "stopped": stopped})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"type": "error", "message": f"bad json: {exc}"})
            continue

        if req.get("stop"):
            stop.set()
            continue
        if "prompt" not in req:
            _emit({"type": "error", "message": "missing 'prompt'"})
            continue

        stop.clear()
        worker = threading.Thread(target=lambda r=req: _guarded(run_generation, r))
        worker.start()
        # Read stdin concurrently: keep consuming lines (e.g. {"stop": true})
        # while generation runs in the background.


def _guarded(fn, arg):
    try:
        fn(arg)
    except _Stopped:
        pass
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "error", "message": str(exc)})


if __name__ == "__main__":
    main()
