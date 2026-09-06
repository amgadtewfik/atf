"""GPU-hang watchdog.

MLX dispatch is async but mx.eval() blocks synchronously; when a custom
Metal kernel wedges the GPU the whole process sits inside that call with
no traceback. This module makes the hang location visible:

- ``watch(label)`` context manager records what GPU work is in flight.
- A daemon thread polls every 0.25 s. If an op stays in flight past
  ``timeout``, it dumps *all* thread stacks via faulthandler (this works
  even though the main thread is blocked inside C code), prints the
  in-flight label and the recent-op history, and then either raises
  SystemExit or just reports (report_only=True).
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager

DEFAULT_TIMEOUT = float(os.environ.get("ATF_WATCHDOG_TIMEOUT", "20"))

_state = threading.local()
_lock = threading.Lock()
_history: deque[str] = deque(maxlen=64)
_inflight: str | None = None
_inflight_since: float | None = None
_started = False
_report_only = False


def start(timeout: float = DEFAULT_TIMEOUT, report_only: bool = False) -> None:
    """Start the watchdog thread once."""
    global _started, _report_only
    if _started:
        return
    _started = True
    _report_only = report_only
    t = threading.Thread(target=_loop, args=(timeout,), daemon=True,
                         name="gpu-watchdog")
    t.start()


def _dump(reason: str) -> None:
    with _lock:
        cur = _inflight
        since = _inflight_since
    hist = list(_history)
    buf = sys.stderr
    print(f"\n{'='*72}\nGPU WATCHDOG: {reason}", file=buf)
    if cur is not None and since is not None:
        print(f"in-flight GPU op ({time.monotonic()-since:.1f}s): {cur}",
              file=buf)
    else:
        print("no labeled op in flight (hang between ops?)", file=buf)
    print("recent ops:", file=buf)
    for h in reversed(hist[-12:]):
        print(f"   {h}", file=buf)
    print("--- all thread stacks ---", file=buf)
    faulthandler.dump_traceback(file=buf)
    print('='*72 + "\n", file=buf)
    buf.flush()


def _loop(timeout: float) -> None:
    while True:
        time.sleep(0.25)
        with _lock:
            cur, since = _inflight, _inflight_since
        if cur is None or since is None:
            continue
        waited = time.monotonic() - since
        if waited >= timeout:
            _dump(f"op exceeded {timeout:.0f}s watchdog budget")
            if not _report_only:
                # Can't interrupt the stuck Metal call; exit so the log
                # isn't lost and the GPU isn't left wedged indefinitely.
                os._exit(70)


@contextmanager
def watch(label: str):
    """Mark a region as the suspected GPU-hang site."""
    global _inflight, _inflight_since
    with _lock:
        prev, prev_since = _inflight, _inflight_since
        _inflight = label
        _inflight_since = time.monotonic()
    try:
        yield
    finally:
        with _lock:
            _history.append(f"[{time.strftime('%H:%M:%S')}] {label}")
            _inflight, _inflight_since = prev, prev_since


class Watched:
    """Mixin: self._gpu_watch(label) context manager bound to an object."""

    def _gpu_watch(self, label: str):
        return watch(label)
