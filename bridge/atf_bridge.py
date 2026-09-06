#!/usr/bin/env python3
"""stdio/socket JSON-line bridge between the Electron UI and the ATF engine.

Protocol (one JSON object per line):

Requests:
  -> {"type":"generate","id":str,"message":str,
      "history":[{"user":str,"assistant":str},...],   # optional; stateless
      "system":str,                                    # optional system prompt
      "max_tokens":int,"temperature":float,"top_p":float,
      "repeat_penalty":float,"thinking":str,"context_tokens":int}
  -> {"type":"stop"}
  -> {"type":"list_models"}
  -> {"type":"load","model":id}
  -> {"type":"unload"}

Events (all carry the request "id" when the request had one):
  <- {"type":"token","text":str}
  <- {"type":"status","text":str}
  <- {"type":"progress","label":str,"pct":float}
  <- {"type":"tier",...}
  <- {"type":"think","open":bool}
  <- {"type":"done","seconds":float,"prompt_tokens":int,"truncated":bool}
  <- {"type":"usage","prompt_tokens":int,"context_tokens":int}
  <- {"type":"error","message":str,"busy":bool}

v15: generation is STATELESS -- the caller supplies the conversation history
and system prompt on every request. The bridge never accumulates state, so
model switches, engine restarts and UI reloads cannot silently erase memory.
History is truncated newest-first to a real tokenizer-measured token budget
instead of growing until the context window overflows.

Debug knobs (set in the env before launching run.sh, all go to stderr):
  ATF_VERBOSE=0  -- default: silent. Existing rich.console status lines only.
  ATF_VERBOSE=1  -- per-request summary: prefill/decode totals, MoE block
                    count, first-token latency, sample-logit NaN count.
  ATF_VERBOSE=2  -- + per-chunk prefill progress, per-MoE-block stats.
  ATF_VERBOSE=3  -- + per-token decode timing/logit-stats, per-block
                    prefill and decode timing.
  ATF_VERBOSE=4  -- + NaN/Inf check on every forward output (residual,
                    router scores, expert weights, decode logits). Very
                    chatty; use this when hunting a numeric bug.
  ATF_NAN_TRAP=1 -- raise on the first NaN/Inf in any of the same spots.
                    Reports the block / stage so the offending op is
                    obvious in the traceback.
  ATF_NAN_WARN=1 -- like ATF_NAN_TRAP but only prints [atf-nan] lines,
                    does not raise. Safer for production-ish runs.
  ATF_OM_SKIP=1  -- bypass the generate()-level OOM preflight (which
                    refuses to start if estimated peak memory exceeds
                    95% of the macOS working set). Set this only if
                    you know the box has the headroom and want to
                    skip the check; otherwise the engine will raise
                    a clear "[ATF_OM]" RuntimeError with a budget
                    breakdown instead of crashing Metal mid-prefill
                    with kIOGPUCommandBufferCallbackErrorOutOfMemory.
"""
import atexit
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent



# ── System-prompt smart router (v9) ───────────────────────────────────────────
# Problem: every request carries the full system prompt, which must be
# prefilled from scratch. On Q2_K_XL the 4-5K-token prefill runs at
# ~20 tok/s, so even a trivial "hi" takes >2 minutes.
#
# Fix: a lightweight router that decides whether each request actually
# needs the system prompt.  Single-turn greetings ("hi", "thanks", "ok")
# are stateless — the model responds fine without the preamble.
# Everything else includes the system, and the engine's existing _pcache
# (LCP-based KV reuse) means subsequent requests with the same system
# skip the 4-5K prefill and only prefill the new user tail.
#
# The router is conservative: it returns True (include system) when
# uncertain.  This preserves the original behavior for any request where
# skipping the system might degrade the answer.


def _sys_hash(sys_text: str, tools=None) -> str:
    """Cache key: SHA-256(system_text + tools)[:24]."""
    h = hashlib.sha256()
    h.update(sys_text.encode("utf-8"))
    for t in (tools or []):
        import json
        h.update(json.dumps(t, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:24]


def needs_system(message: str, history: list | None = None) -> tuple[bool, str]:
    """Decide whether the system prompt is needed for this request.

    Args:
        message: the new user message text
        history: list of prior turns [{"user": str, "assistant": str}, ...]
                 or None

    Returns (needs: bool, reason: str).  Reason is a debug label only.
    """
    history = history or []
    user_n = len(history) + 1          # prior turns + this one
    asst_n = len(history)              # prior assistant responses

    # Rule 0: multi-turn → always include (even "ok" is confirming context)
    if (user_n > 1) or (user_n > 0 and asst_n > 0):
        return True, f"multi_turn:{user_n}u/{asst_n}a"

    # Rule 1: extract the last user message (single-turn only from here)
    msg = (message or "").strip()
    if not msg:
        return False, "empty_message"
    lower = msg.lower()
    words = msg.split()

    # Rule 2: obvious greetings / acknowledgments (single-turn only)
    GREETINGS = {
        "hi", "hi!", "hey", "hey!", "yo", "yo!", "hello", "hello!",
        "hii", "hiii", "hiya", "heya", "howdy",
        "thanks", "thank you", "thx", "thx!", "ty", "tysm",
        "ok", "okay", "ok!", "k", "kk", "mk", "got it", "understood",
        "noted", "acknowledged", "sounds good", "sounds great",
        "sounds fine", "works for me", "makes sense", "i agree",
        "agreed", "right", "correct", "indeed", "exactly",
        "yes", "no", "nope", "yep", "yeah", "ya",
        "bye", "goodbye", "cya", "see ya", "later",
        "good night", "good morning", "good afternoon", "good evening",
        "nice", "cool", "awesome", "great", "wow", "yay", "amazing",
        "lol", "lmao", "haha", "hehe", "hmm", "hmmm",
        "oh", "oh!", "ah", "ah!", "ahh", "aah", "oof", "ouch", "yikes",
        "please", "pls", "sure", "sure!",
        "👍", "👋", "🙂", "😊", "❤️", "💙", "🙏", "✌️", "🤝",
    }
    if lower in GREETINGS or lower.rstrip(".!?,") in GREETINGS:
        return False, f"greeting:{msg!r}"

    # Rule 3: single short word (≤5 alphabetic chars, not a command)
    COMMANDS = {
        "read", "write", "list", "show", "search", "find",
        "delete", "create", "update", "edit", "check",
        "run", "exec", "help", "start", "stop", "kill",
        "what", "where", "which", "who", "when", "how", "why",
        "send", "grep", "use", "act", "make", "give",
    }
    if (len(words) == 1 and len(lower) <= 5 and lower.isalpha()
            and lower not in COMMANDS):
        return False, f"short_word:{msg!r}"

    # Rule 4: tool / action keywords
    TOOL_KW = (
        "read ", "write ", "edit ", "delete ", "create ", "copy ", "move ",
        "rename ", "append ", "prepend ", "replace ", "merge ", "split ",
        "extract ", "archive ", "mkdir ", "rmdir ", "touch ", "chmod ",
        "search ", "find ", "grep ", "locate ", "list ", "show ",
        "display ", "view ", "inspect ", "examine ", "check ", "verify ",
        "validate ", "test ", "run ", "code ", "compile ", "execute ",
        "debug ", "profile ", "benchmark ", "install ", "pip ", "npm ",
        "cargo ", "import ", "export ", "build ", "agent ", "task ",
        "plan ", "reason ", "think ", "analyze ", "compare ",
        "implement ", "refactor ", "review ", "audit ", "optimize ",
        "fix ", "fetch ", "download ", "upload ", "parse ",
        "scrape ", "crawl ", "query ", "database ", "sql ",
        "http", "url", "kill ", "start ", "stop ", "restart ",
        "deploy ", "ssh ", "shell ", "process ", "memory ",
        "cpu ", "disk ", "network ", "port ", "server ",
        "help ", "explain ", "describe ", "document ",
        "summarize ", "translate ",
        "read", "write", "edit", "delete", "create", "copy", "move",
        "rename", "search", "find", "grep", "list", "show", "check",
        "verify", "test", "run", "compile", "execute", "debug", "install",
        "build", "implement", "refactor", "fix", "fetch", "download",
        "upload", "parse", "help", "explain", "describe", "summarize",
        "translate", "send", "kill", "start", "stop", "restart", "deploy",
    )
    for kw in TOOL_KW:
        if kw in lower:
            return True, f"tool_kw:{kw!r}"

    # Rule 5: patterns (paths, URLs, code, SQL)
    PATTERNS = (
        r"\./", r"\.\./", r"/[a-z]", r"[A-Z]:\\", r"[a-z]:[/\\]",
        r"https?://", r"www\.",
        r"import\s+\w+", r"from\s+\w+",
        r"def\s+\w+", r"class\s+\w+",
        r"function\s+\w+", r"const\s+\w+",
        r"SELECT\s+", r"INSERT\s+", r"UPDATE\s+", r"DELETE\s+",
        r"git\s+", r"docker\s+",
        r"\w+\.py\b", r"\w+\.js\b", r"\w+\.ts\b",
        r"\w+\.json\b", r"\w+\.md\b",
    )
    for pat in PATTERNS:
        if re.search(pat, msg, re.IGNORECASE):
            return True, f"pattern:{pat!r}"

    # Rule 6: interrogative question
    INTERROG = {"how", "what", "why", "where", "when", "which", "who", "whose"}
    if words and words[0].lower().rstrip("?,.") in INTERROG:
        return True, f"question:{words[0]!r}"

    # Rule 7: imperative verb
    IMPERATIVE = {
        "do", "make", "let", "get", "use", "call", "tell", "show",
        "give", "find", "take", "put", "set", "go", "try", "keep",
        "send", "come", "look", "open", "close", "save", "quit",
        "explain", "describe", "help", "act", "be", "include",
        "write", "read", "run", "check", "list", "create",
    }
    if words and words[0].lower().rstrip("?.,") in IMPERATIVE:
        return True, f"imperative:{words[0]!r}"

    # Rule 8: short ambiguous input → conservative include
    if len(words) <= 3:
        return True, f"short_ambig:{msg!r}"

    # Rule 9: default include
    return True, f"default:{msg[:30]!r}"





def _find_model() -> Path:
    """Single shared model outside releases.

    Resolution order:
      1. $ATF_MODEL (explicit override)
      2. <root>/models/, <root>/../models/, <root>/../../models/
         (covers project root and release folders alike) — at the top
         level *and* in any versioned subdirectory (v1/, v2/, v3/, v4/).
    """
    env = os.environ.get("ATF_MODEL")
    if env:
        return Path(env).expanduser()
    name = "Qwen3.5-9B-BF16.atf"
    version_dirs = ("v1", "v2", "v3", "v4")
    for base in (ROOT, ROOT.parent, ROOT.parent.parent):
        candidate = base / "models" / name
        if candidate.exists():
            return candidate
        for v in version_dirs:
            legacy = base / "models" / v / name
            if legacy.exists():
                return legacy
    return ROOT / "models" / name


MODEL = _find_model()

sys.path.insert(0, str(ROOT))

# v16 safeguard: wire the GPU watchdog into the bridge. If a Metal op stays
# in flight past the budget, dump all thread stacks and os._exit(70) so the
# GPU is not left wedged indefinitely (the machine-hang failure class).
# Electron's crash banner auto-restarts the bridge; the machine survives.
from atf.gpu_watchdog import start as _gpu_watchdog_start
try:
    _gpu_watchdog_start(
        timeout=float(os.environ.get("ATF_WATCHDOG_TIMEOUT", "20")),
        report_only=os.environ.get("ATF_WATCHDOG_REPORT_ONLY") == "1",
    )
except Exception:
    pass

# Electron tells the bridge where models live: $ATF_MODELS_DIR at spawn
# (covers bridge restarts) and {"type":"set_models_dir"} at runtime (covers
# the user changing the folder in Settings without a restart). Without this
# the registry falls back to the compile-time parents[3]/models guess, which
# in dev mode points at the project tree instead of the user's folder.
_env_models_dir = os.environ.get("ATF_MODELS_DIR")
if _env_models_dir:
    try:
        from atf.model_registry import set_models_root
        set_models_root(_env_models_dir)
        print(f"[bridge] models dir override: {_env_models_dir}", file=sys.stderr)
    except Exception as _exc:
        print(f"[bridge] models dir override failed: {_exc}", file=sys.stderr)


import re

_MARKUP = re.compile(r"\[/?(\w+)\]")


class StreamShim:
    """Replaces rich consoles: prints become JSON lines."""

    def __init__(self, out, mode="load"):
        self._out = out
        self.mode = mode

    def _emit(self, obj):
        self._out.write(json.dumps(obj) + "\n")
        self._out.flush()

    def print(self, text="", end="\n", **kw):
        s = str(text)
        if self.mode == "gen" and end == "" and s and not s.startswith("["):
            self._emit({"type": "token", "text": s})
            return
        clean = _MARKUP.sub("", s).strip()
        if self.mode == "gen" and clean.startswith("Prefill:"):
            self._emit({"type": "status", "text": "decoding…"})
            return
        if clean and not clean.startswith(("Dense tensors", "Expert INT8", "GPU transfer", "Pre-quantizing")):
            self._emit({"type": "status", "text": clean})

    def rule(self, *a, **kw):
        pass

    def input(self, *a, **kw):
        return ""

    def __getattr__(self, name):
        return lambda *a, **k: None


class ProgressShim:
    """Replaces rich.progress.Progress: advances become pct events."""

    def __init__(self, out, *args, **kwargs):
        self._out = out
        self._tasks = {}

    def add_task(self, description, total=None, **kw):
        tid = len(self._tasks)
        self._tasks[tid] = [description, 0, total or 1]
        self._emit_progress(tid)
        return tid

    def advance(self, task_id, n=1):
        t = self._tasks.get(task_id)
        if t:
            t[1] += n
            self._emit_progress(task_id)

    def update(self, task_id, **kw):
        pass

    def _emit_progress(self, task_id):
        desc, done, total = self._tasks[task_id]
        pct = min(100.0, done / total * 100.0) if total else 0.0
        self._out.write(json.dumps({"type": "progress", "label": desc,
                                    "pct": round(pct, 1)}) + "\n")
        self._out.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def load_engine(out, model_path=None):
    import atf.model as model_mod
    from atf.model import load_atf
    from atf.engine import Engine
    from atf.tokenizer import Tokenizer

    shim = StreamShim(out, mode="load")
    model_mod.console = shim
    model_mod.Progress = lambda *a, **k: ProgressShim(out, *a, **k)

    mp = Path(model_path) if model_path else MODEL
    out.write(json.dumps({"type": "status", "text": f"reading {mp.name}…"}) + "\n")
    out.flush()
    cache_gb = float(os.environ.get("ATF_CACHE_GB", "6"))
    model = load_atf(str(mp), lru_cap_bytes=int(cache_gb * (1 << 30)))
    src = "Qwen/Qwen3.5-9B" if (model.vocab_size > 200000 or "3.5" in mp.name) else "Qwen/Qwen3.5-4B"
    out.write(json.dumps({"type": "status", "text": "Loading Tokenizer…"}) + "\n")
    out.flush()
    tok = Tokenizer.from_hf(src)
    return Engine(model, tok), tok


def main():
    import atf.engine as engine_mod

    out = sys.stdout
    sys.stdout = sys.stderr

    def _trace(stage, **kw):
        """Stderr-only diagnostic mirror that the Electron main process
        forwards to the API log so we can pinpoint where the gen worker
        stalls. Cheap, single-line, never raises."""
        try:
            import json as _json
            sys.stderr.write("[bridge-trace] " + stage + " " +
                             _json.dumps(kw, default=str) + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    # v8: NO model is loaded at startup. Scan the registry (headers only)
    # and let the UI pick; the chosen model loads on demand.
    import mlx.core as _mx

    def free_engine():
        eng = state["engine"]
        # drop the prefix-cache KV/GDN state explicitly -- it can pin
        # gigabytes of GPU-resident KV buffers even after the engine itself
        # is dereferenced (GC may not run promptly on refcount alone).
        if eng is not None:
            try:
                eng._pcache = None
            except Exception:
                pass
        state["engine"] = None
        state["tok"] = None
        state["id"] = None
        # v10: clear sys_baseline too (engine is gone, KV is invalid)
        state["sys_baseline"] = None
        import gc as _gc; _gc.collect()
        try:
            _mx.synchronize()
        except Exception:
            pass
        try:
            _mx.clear_cache()
        except Exception:
            pass

    def emit_models():
        try:
            from atf.model_registry import scan_models
            items = [{"id": m.id, "format": f"v{m.format_major}",
                      "size_gb": round(m.size_bytes / 1e9, 2),
                      "path": str(m.path),
                      "loaded": m.id == state["id"]}
                     for m in scan_models()]
        except Exception as exc:
            import traceback as _tb; _tb.print_exc(file=sys.stderr)
            items = []
            out.write(json.dumps({"type": "error",
                                  "message": f"registry scan failed: {exc}"}) + "\n")
        out.write(json.dumps({"type": "models", "items": items}) + "\n")
        out.flush()

    def ensure_model(model_id):
        """Load a model on the CALLING thread (main/gen thread only!)."""
        if state["id"] == model_id and state["engine"] is not None:
            return
        from atf.model_registry import scan_models
        info = next((m for m in scan_models() if m.id == model_id), None)
        if info is None:
            raise ValueError(f"unknown model: {model_id!r}")
        if state["engine"] is not None:
            out.write(json.dumps({"type": "status",
                                  "text": f"unloading {state['id']}…"}) + "\n")
            out.flush()
            free_engine()
        size_gb = round(info.size_bytes / 1e9, 2)
        out.write(json.dumps({"type": "status",
                              "text": f"loading {info.id} ({size_gb} GB)…"}) + "\n")
        out.flush()
        engine, tok = load_engine(out, str(info.path))
        state["engine"], state["tok"], state["id"] = engine, tok, info.id
        out.write(json.dumps({"type": "model-loaded", "id": info.id,
                              "format": f"v{info.format_major}"}) + "\n")
        out.write(json.dumps({"type": "ready"}) + "\n")   # engine usable now
        out.flush()

    state = {"engine": None, "tok": None, "id": None,
          # v10: cached system-prompt KV state, scoped per (model, system_text).
          # Injected into engine._pcache before each generate() so the LCP
          # path can skip the multi-second/minute system prefill.
          "sys_baseline": None,  # (sys_hash, sys_ids, sys_kv, sys_gdn)
          # v20: id of the request _generate() is currently processing, so
          # a "stop" carrying an id only cancels that specific request (see
          # the rtype == "stop" handler above and the gen-worker loop below).
          "current_rid": None}
    emit_models()
    out.write(json.dumps({"type": "ready", "note": "no model loaded yet"}) + "\n")
    out.flush()

    shim = StreamShim(out, mode="gen")
    real_console = engine_mod.console
    engine_mod.console = shim
    import queue
    gen_queue: "queue.Queue" = queue.Queue()
    gen_busy = threading.Event()

    def emit(w, obj, rid=None):
        """Write an event; echo the originating request id when present."""
        if rid is not None and "id" not in obj:
            obj["id"] = rid
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")
        w.flush()

    def handle_line(line, w=None):
        if not line:
            return
        if w is None:
            w = out
        try:
            _handle_line(line, w)
        except Exception as exc:
            import traceback as _tb; _tb.print_exc(file=sys.stderr)
            try:
                w.write(json.dumps({"type": "error",
                                    "message": f"handler: {exc}"}) + "\n")
                w.flush()
            except Exception:
                pass

    stop_event = threading.Event()

    class _Stopped(Exception):
        pass

    def _handle_line(line, w):
        if not line:
            return
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            return
        rtype = req.get("type", "generate" if "message" in req else None)
        if rtype == "stop":
            # v20: scope to the request that's actually running when an id
            # is given (API disconnects pass their own id -- see main.cjs).
            # A bare {"type":"stop"} with no id (the UI's Stop button)
            # keeps the old "cancel whatever's running" behavior.
            target = req.get("id")
            if target is None or target == state.get("current_rid"):
                stop_event.set()
            return
        if rtype == "list_models":
            emit_models()
            return
        if rtype == "set_models_dir":
            from atf.model_registry import set_models_root
            p = req.get("path")
            if p:
                set_models_root(p)
                print(f"[bridge] models dir set via UI: {p}", file=sys.stderr)
            emit_models()
            return
        if rtype == "refresh_registry":
            # The user changed the models directory from the UI; the symlink
            # at <projectRoot>/models has been updated, but MODELS_ROOT is a
            # module-level constant computed at import time. Re-import so the
            # next scan picks up the new path.
            import importlib
            import atf.model_registry
            importlib.reload(atf.model_registry)
            # Also re-bind in atf namespace if anything imported it directly.
            import atf
            atf.model_registry = importlib.import_module("atf.model_registry")
            emit_models()
            return
        if rtype == "load":
            gen_queue.put(({"__op": "load", "model": req.get("model"),
                            "id": req.get("id")}, w))
            return
        if rtype == "unload":
            # aborts any in-flight generation first, then frees the engine
            # on the gen thread. Never kills the process.
            stop_event.set()
            gen_queue.put(({"__op": "unload", "id": req.get("id")}, w))
            return
        if rtype == "generate":
            gen_queue.put((req, w))
            return

    # MLX streams are per-thread AND tied to where the model was loaded:
    # generation must stay on THIS (main) thread. Socket/stdin handlers only
    # enqueue requests; stop arrives via stop_event.

    # ── v15: stateless prompt assembly with real token budgeting ──────
    def build_prompt(tok, req, skip_system: bool = False):
        """ChatML prompt, history truncated newest-first to fit the budget.

        v9: when skip_system is True the system message is omitted from
        the rendered prompt, so the prefill is just the user message +
        history.  The router sets this for stateless single-turn greetings
        ("hi", "thanks") where the system prompt is unused.
        """
        ctx = int(req.get("context_tokens") or 65536)
        # gamma/v4 (2026-09-02): default raised from 512 to 65536
        # to match the UI default (no artificial truncation; the engine
        # stops on EOS naturally or when the user clicks Stop).
        max_new = int(req.get("max_tokens", 65536))
        margin = 64                      # safety for chat-template overhead
        # gamma/v7 fix (2026-09-05): do NOT reserve the full `max_new` here.
        # The UI's own defaults set max_tokens == context_tokens (both
        # 65536, per the gamma/v4 change above), so `ctx - max_new - margin`
        # collapsed to a negative number and the history budget floored at
        # 1024 tokens on every request -- regardless of the real 65536-token
        # context window. Any prior turn bigger than ~1024 tokens (e.g. a
        # multi-thousand-token generated SVG/code reply) was silently
        # dropped as "outside context window", so the very next turn's
        # prompt carried NO history at all and the model appeared to have
        # total amnesia mid-conversation. The REAL output cap is already
        # correctly computed further down (max_new_for_engine, after the
        # actual prompt+history is built), so this pre-reservation only
        # needs to be generous enough to leave room for a reply -- it must
        # not be allowed to consume the whole context and starve history.
        reply_reserve = min(max_new, max(1024, ctx // 4))
        budget = max(1024, ctx - reply_reserve - margin)

        def n_tok(s):
            return len(tok.encode(s))

        system = (req.get("system") or "").strip()
        head = ""
        used = 0
        if system and not skip_system:
            head = f"<|im_start|>system\n{system}<|im_end|>\n"
            used += n_tok(head)

        tail = ("<|im_start|>user\n" + req["message"] + "<|im_end|>\n"
                + "<|im_start|>assistant\n")
        used += n_tok(tail)

        history = req.get("history") or []
        kept = []
        dropped = 0
        for turn in reversed(history):
            u = str(turn.get("user", ""))
            a = str(turn.get("assistant", ""))
            if not u and not a:
                continue
            block = (f"<|im_start|>user\n{u}<|im_end|>\n"
                     f"<|im_start|>assistant\n{a}<|im_end|>\n")
            t = n_tok(block)
            if used + t > budget:
                dropped += 1
                continue          # older turns are larger still; keep counting
            used += t
            kept.append(block)
        kept.reverse()
        prompt = head + "".join(kept) + tail
        return prompt, used, len(history), dropped


    def _generate(req, w):
        rid = req.get("id")
        gen_busy.set()
        stop_event.clear()
        _trace("_generate.entry", id=rid, op=req.get("__op"),
               model=req.get("model"), state_id=state.get("id"))

        if req.get("__op") == "unload":
            try:
                if state["engine"] is not None:
                    emit(w, {"type": "status",
                             "text": f"unloading {state['id']}…"}, rid)
                    free_engine()
                    emit_models()
                emit(w, {"type": "unloaded"}, rid)
            except Exception as exc:
                import traceback as _tb; _tb.print_exc(file=sys.stderr)
                emit(w, {"type": "error", "message": f"unload failed: {exc}"}, rid)
            finally:
                gen_busy.clear()
            return

        if req.get("__op") == "load":
            try:
                ensure_model(req["model"])
                emit_models()
            except Exception as exc:
                import traceback as _tb; _tb.print_exc(file=sys.stderr)
                emit(w, {"type": "error", "message": f"load failed: {exc}"}, rid)
            finally:
                gen_busy.clear()
            return

        # per-request model selection; loads on this (gen) thread
        want = req.get("model") or state["id"]
        if not want:
            emit(w, {"type": "error", "busy": False,
                     "message": "no model loaded — pick one from the "
                                "model dropdown first"}, rid)
            gen_busy.clear()
            return
        try:
            ensure_model(want)
        except Exception as exc:
            import traceback as _tb; _tb.print_exc(file=sys.stderr)
            emit(w, {"type": "error", "message": f"load failed: {exc}"}, rid)
            gen_busy.clear()
            return

        engine, tok = state["engine"], state["tok"]

        # v9: smart router — does this turn actually need the system prompt?
        # Single-turn greetings skip it (zero prefill). Multi-turn, tool
        # requests, and questions include it.
        sys_text = (req.get("system") or "").strip()
        needs_sys, sys_reason = needs_system(req.get("message", ""),
                                            req.get("history") or [])
        skip_sys = not needs_sys
        sys.stdout.write(f"[bridge-sys] id={rid} needs={needs_sys} "
                         f"reason={sys_reason} sys_len={len(sys_text)} "
                         f"msg={req.get('message', '')[:60]!r}\n")
        sys.stdout.flush()

        # v10: sys_baseline injection — if we have a cached system KV for this
        # (model, system_text) pair, inject it into engine._pcache BEFORE the
        # prompt is encoded so the LCP path inside generate() can skip the
        # multi-second system prefill.  Cache is built at the END of each
        # successful generate() call (see below).
        if needs_sys:
            sys_hash = _sys_hash(sys_text, [])
            baseline = state.get("sys_baseline")
            if baseline is not None and baseline[0] == sys_hash:
                try:
                    engine._pcache = (baseline[1], baseline[2], baseline[3])
                    sys.stdout.write(f"[bridge-sys] id={rid} cache=HIT "
                                     f"n_sys={len(baseline[1])} "
                                     f"engine._pcache restored\n")
                    sys.stdout.flush()
                except Exception:
                    pass  # cache corrupted; fall through to full prefill

        try:
            prompt, prompt_tokens, n_hist, dropped = build_prompt(
                tok, req, skip_system=skip_sys)
        except Exception as exc:
            import traceback as _tb; _tb.print_exc(file=sys.stderr)
            emit(w, {"type": "error", "message": f"context build failed: {exc}"}, rid)
            gen_busy.clear()
            return
        ctx = int(req.get("context_tokens") or 65536)

        from atf.router import route, BADGES
        _trace("_generate.before_route", id=rid,
               prompt_tokens=prompt_tokens, n_hist=n_hist)
        tier, diff = route(req["message"], req.get("thinking", "auto"))
        _trace("_generate.before_tier_emit", id=rid, tier=tier.name)
        emit(w, {"type": "tier", "level": tier.level,
                 "name": tier.name, "badge": BADGES[tier.level],
                 "score": round(diff, 2)}, rid)
        note = f"prefill ({n_hist} prior turn(s), {prompt_tokens} tok)"
        if dropped:
            note += f" — oldest {dropped} turn(s) outside context window"
        emit(w, {"type": "status", "text": note}, rid)
        emit(w, {"type": "usage", "prompt_tokens": prompt_tokens,
                 "context_tokens": ctx}, rid)
        cfg_thinking = tier.thinking
        if cfg_thinking != "off":
            emit(w, {"type": "think", "open": True}, rid)
        t0 = time.time()
        gen_ids = []
        prev_text = ""
        # v18 perf fix: decode only the small pending window, not the whole
        # reply-so-far, on every token -- tok.decode(gen_ids) with gen_ids
        # growing every step made streaming O(n^2) in reply length (worst
        # on the 27B, whose replies are already the slowest to produce
        # token-by-token). A clean (no U+FFFD) decode of the pending window
        # is final text: BPE decode only needs enough adjacent tokens to
        # complete a split multi-byte glyph, it never retroactively changes
        # text already resolved by an earlier decode.
        pending_ids = []

        last_progress = {"t": 0.0, "done": -1}

        def on_progress(done, total):
            # Emit on every meaningful change. The engine only fires this
            # callback when a prefill block completes, so it's naturally
            # bounded by the work itself. A soft 60Hz cap (every 16ms)
            # protects against pathological back-to-back events without
            # dropping intermediate progress on fast prefills (e.g. the
            # 27B finishes 64 blocks in ~60ms, which a 0.25s throttle
            # would have collapsed to a single 0% -> 100% jump).
            if stop_event.is_set():
                # Cancel mid-prefill: bail out now instead of burning the
                # rest of the system-prompt prefill on a prompt the user
                # no longer wants. The engine's _pcache is cleared by
                # the _Stopped handler below.
                raise _Stopped()
            now = time.time()
            if done == last_progress["done"] and done < total:
                return  # no change, nothing to send
            if now - last_progress["t"] < 0.016 and done < total:
                return  # 60Hz soft cap, only during in-flight prefill
            last_progress["t"] = now
            last_progress["done"] = done
            emit(w, {"type": "progress", "stage": "prefill",
                     "done": done, "total": total}, rid)

        def on_token(tid):
            nonlocal prev_text
            if stop_event.is_set():
                raise _Stopped()
            if tid in (248068, 248069):   # <think>, </think>
                emit(w, {"type": "think", "open": tid == 248068}, rid)
                return
            # gen_ids still grows every token (its length is used below for
            # the max_tokens truncation check); pending_ids is the small
            # decode window -- see comment where both are declared.
            gen_ids.append(tid)
            pending_ids.append(tid)
            delta = tok.decode(pending_ids)
            if "\ufffd" in delta:
                return   # glyph incomplete; wait for the next token
            pending_ids.clear()
            if delta:
                emit(w, {"type": "token", "text": delta}, rid)
                prev_text += delta

        # gamma/v4 (2026-09-02): clamp max_tokens so prompt + max_tokens
        # never exceeds max_context. The engine raises ValueError if
        # this is violated, so we trim the request's max_tokens to fit.
        # The user wanted "no artificial truncation" -- this still
        # produces up to the full context budget for the answer, just
        # never more than (ctx - actual_prompt_len - safety_margin).
        safety_margin = 16
        requested_max = int(req.get("max_tokens", 65536))
        max_new_for_engine = min(requested_max, max(0, ctx - prompt_tokens - safety_margin))
        cfg = engine_mod.GenConfig(
            # user's explicit max_tokens wins (up to the context budget)
            max_tokens=max_new_for_engine,
            temperature=float(req.get("temperature", 0.0)),
            top_p=float(req.get("top_p", 0.9)),
            repeat_penalty=float(req.get("repeat_penalty", 1.15)),
            exact_ffn=True,
            thinking=cfg_thinking,
            mem_log_interval=0,
            max_context=ctx,
            # gamma/v7: read ATF_PREFILL_CHUNK from env (set by main.cjs).
            # Smaller chunk = shorter Metal command buffer = less GPU watchdog
            # timeout risk on long-context prefill. Default 512 (was 1024).
            prefill_chunk=int(os.environ.get("ATF_PREFILL_CHUNK", "512")),
            # gamma/v7 persistence (ATF_KV_SSD_PERSIST): scopes the on-disk
            # paged-KV cache key to the model actually loaded for this
            # request, so two different models can never collide on the
            # same session key.
            model_id=state.get("id"),
        )

        engine_mod.console = shim
        text = None
        err = None
        _trace("_generate.before_engine", id=rid, prompt_len=len(prompt),
               max_tokens=cfg.max_tokens, thinking=cfg.thinking)
        # Emit an atf-verbose marker so the request boundary is visible
        # even when ATF_VERBOSE=0 (it just shows one line per request).
        import os as _osv
        _verbose_lvl = _osv.environ.get("ATF_VERBOSE", "0")
        _nan_trap = _osv.environ.get("ATF_NAN_TRAP", "0")
        sys.stderr.write(f"[bridge-trace] _generate.engine_call id={rid} "
                         f"ATF_VERBOSE={_verbose_lvl} ATF_NAN_TRAP={_nan_trap} "
                         f"prompt_len={len(prompt)} max_tokens={cfg.max_tokens} "
                         f"thinking={cfg.thinking}\n")
        sys.stderr.flush()
        try:
            text = engine.generate(prompt, cfg, stream=False, token_cb=on_token,
                                   progress_cb=on_progress)
        except _Stopped:
            _trace("_generate.engine_stopped", id=rid)
            raise
        except Exception as exc:
            import traceback as _tb; _tb.print_exc(file=sys.stderr)
            _trace("_generate.engine_exception", id=rid, err=str(exc))
            err = str(exc)
        else:
            _trace("_generate.engine_returned", id=rid,
                   text_len=len(text) if text else 0)
            # v10: snapshot the system portion of engine._pcache for reuse
            # on the NEXT request.  We cap each KV cache to n_sys tokens
            # and store the system prefix; the LCP path in generate() will
            # pick it up via the injection above.
            sys.stdout.write(f"[bridge-sys] id={rid} cache=TRY "
                             f"needs_sys={needs_sys} text_len={len(text) if text else 0}\n")
            sys.stdout.flush()
            if needs_sys and text is not None:
                try:
                    # Re-encode the system block the same way build_prompt did.
                    # NOTE: must include the trailing "\n" that build_prompt's
                    # `head` has (`f"<|im_start|>system\n{...}<|im_end|>\n"`).
                    # Without it, the engine re-tokenizing the full prompt may
                    # produce a different token sequence than our sys_ids, and
                    # the LCP loop in generate() can overshoot -- if our sys_ids
                    # is encoded as N tokens but the engine's re-tokenized system
                    # block is only M<N tokens, lcp hits the shorter length and
                    # (in extreme cases) covers the entire new prompt, leaving
                    # nothing to prefill and crashing the LM head.
                    sys_block = f"<|im_start|>system\n{sys_text}<|im_end|>\n"
                    sys_ids = tok.encode(sys_block)
                    n_sys = len(sys_ids)
                    pcache = getattr(engine, "_pcache", None)
                    if pcache is not None and n_sys > 0:
                        old_ids, old_kv, old_gdn = pcache
                        if n_sys <= len(old_ids):
                            # Take a VIEW of the system portion of the KV caches.
                            # Slicing MLX arrays is a view, not a copy -- this is
                            # zero-cost in memory; the data stays in old_kv.
                            sys_kv = {}
                            for b_, c in old_kv.items():
                                if c.n >= n_sys:
                                    new_c = type(c)()
                                    new_c.n_kv = c.n_kv
                                    new_c.head_dim = c.head_dim
                                    new_c.n = n_sys
                                    # v10 fix: use .copy() instead of slicing to avoid
                                    # view-aliasing that can corrupt the original cache
                                    # when the engine reuses it on the next request.
                                    new_c.k_buf = c.k_buf[:n_sys].copy()
                                    new_c.v_buf = c.v_buf[:n_sys].copy()
                                    sys_kv[b_] = new_c
                            if sys_kv:
                                state["sys_baseline"] = (
                                    sys_hash, sys_ids, sys_kv,
                                    dict(old_gdn) if old_gdn else {})
                                sys.stdout.write(
                                    f"[bridge-sys] id={rid} cache=SAVE "
                                    f"n_sys={n_sys} blocks={len(sys_kv)} "
                                    f"model={state['id']}\n")
                                sys.stdout.flush()
                except Exception as exc:
                    sys.stdout.write(f"[bridge-sys] id={rid} cache=SAVE_ERR "
                                     f"{exc!r}\n")
                    sys.stdout.flush()
        finally:
            engine_mod.console = real_console

        seconds = time.time() - t0
        if text is not None:
            # budget exhausted (not stopped, not EOS) => answer was cut short
            truncated = len(gen_ids) >= cfg.max_tokens
            done = {"type": "done", "seconds": round(seconds, 2),
                    "prompt_tokens": prompt_tokens,
                    "gen_tokens": len(gen_ids)}
            if truncated:
                done["truncated"] = True
                done["note"] = (f"response stopped at the {cfg.max_tokens}-token "
                                f"budget ({tier.name} tier)")
            emit(w, done, rid)
        elif err is not None:
            emit(w, {"type": "error", "message": err}, rid)
        gen_busy.clear()
        # report GPU memory after every request so leaks/wedge states are
        # visible immediately in the UI instead of discovered as a freeze.
        try:
            import mlx.core as _mx2
            from atf.model import gpu_mem_line as _gml
            emit(w, {"type": "mem", "text": _gml()}, rid)
        except Exception:
            pass

    sock_path = os.environ.get("ATF_BRIDGE_SOCK")
    if sock_path:
        import socketserver

        parent = Path(sock_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                class W:
                    def __init__(self, f):
                        self.f = f
                    def write(self, s):
                        self.f.write(s.encode("utf-8"))
                    def flush(self):
                        self.f.flush()
                out.write(json.dumps({"type": "status", "text": "client connected"}) + "\n")
                out.flush()
                for raw in self.rfile:
                    handle_line(raw.decode("utf-8", "replace").strip(),
                                w=W(self.wfile))

        # Threading: the UI holds ONE persistent connection for the whole
        # app lifetime, and each API request opens its own dedicated
        # connection. A single-threaded server would block forever inside
        # the first handler, starving every later connection (API requests
        # would connect but never be read -> zero events). Concurrent
        # generation is already serialized by gen_queue + gen_busy.
        class Server(socketserver.ThreadingUnixStreamServer):
            allow_reuse_address = True
            daemon_threads = True

        srv = Server(sock_path, Handler)
        out.write(json.dumps({"type": "socket-listening", "path": sock_path}) + "\n")
        out.flush()
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    else:
        threading.Thread(target=lambda: [handle_line(l) for l in iter(sys.stdin.readline, "")],
                         daemon=True).start()

    # Engine loop -- MAIN thread (same thread that loaded the model).
    while True:
        item = gen_queue.get()
        if item is None:
            return
        req, w = item
        _trace("gen_worker.got", id=req.get("id"), op=req.get("__op"),
               type=req.get("type"), busy=gen_busy.is_set(),
               state_id=state.get("id"))
        if gen_busy.is_set():
            # v15: never silently drop a request -- tell the caller.
            emit(w, {"type": "error", "busy": True,
                     "message": "engine busy — wait for the current "
                                "operation to finish"}, req.get("id"))
            continue
        t0 = time.time()
        # Helper: drop any prefix-cache state the engine accumulated for
        # the cancelled/failed request. If we leave the cache intact, the
        # next request computes LCP against a prompt the user abandoned
        # and may incorrectly reuse partial KV from it. The syscache
        # baseline (system-prompt-only KV) is also dropped because it was
        # built from this model's most recent prefill, which may have
        # been partially overwritten by the cancelled run.
        def _drop_pcache():
            eng = state.get("engine")
            if eng is None:
                return
            try:
                pcache = getattr(eng, "_pcache", None)
                if pcache is not None:
                    eng._pcache = None
            except Exception:
                pass
            # Also clear the in-memory baseline so we don't inject stale
            # system KV on the next request.
            state["sys_baseline"] = None

        state["current_rid"] = req.get("id")
        try:
            _generate(req, w)
        except _Stopped:
            _drop_pcache()
            stop_event.clear()
            gen_busy.clear()
            emit(w, {"type": "done", "stopped": True,
                     "seconds": round(time.time() - t0, 2)}, req.get("id"))
        except Exception as exc:
            import traceback as _tb; _tb.print_exc(file=sys.stderr)
            _drop_pcache()
            gen_busy.clear()
            stop_event.clear()
            emit(w, {"type": "error", "message": str(exc)}, req.get("id"))
        finally:
            # v20: clear so a late/stray stop for this id can't affect the
            # NEXT request once this one is done (success path clears it
            # too, via this finally, not just the exception paths above).
            state["current_rid"] = None


if __name__ == "__main__":
    def _shutdown(*_a):
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        try:
            if os.environ.get("ATF_BRIDGE_SOCK"):
                os.unlink(os.environ["ATF_BRIDGE_SOCK"])
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    atexit.register(_shutdown)
    main()
