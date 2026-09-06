"""OpenAI-compatible HTTP server for ATF (v7+v9).

Endpoints:
  GET  /v1/models                     -> model list
  POST /v1/chat/completions           -> chat, SSE streaming or JSON
  GET  /health                        -> liveness

Any OpenAI client (ChatBox, LibreChat, code editors, `openai` python pkg)
works by pointing base_url at http://localhost:8000/v1.

Usage:
  python -m atf.server_openai models/Qwen3.5-9B-BF16.atf [--port 8000]

Notes:
- Generation is serialized with a lock: the MLX engine is single-stream.
- When "thinking" is omitted, the adaptive router picks tier + thinking
  level from the prompt difficulty (same policy as the Electron bridge).
- <think>/</think> tokens are passed through to content as plain text.

v9: system-prompt KV cache + smart router.
  Single-turn greetings ("hi", "thanks") are sent WITHOUT the system
  prompt, so the 4838-token prefill is skipped entirely. For everything
  else, the system prompt is included but its KV state is cached and
  reused across requests with the same system hash, so the 4838-token
  prefill is skipped on every request after the first one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import threading
import time
import uuid
import mlx.core as mx
from .engine import Engine, GenConfig
from .model import load_atf
from .tokenizer import Tokenizer
# v10: disk persistence for system-prompt KV state
from . import syscache as _syscache
from .toolcall import (
    ToolCallStreamParser,
    format_openai_tool_calls,
    parse_output,
    render_chatml_messages,
)

MODEL_NAME = "atf-qwen3.5-9b"

log = logging.getLogger("atf.server")

# ── System-prompt router + KV cache helpers ───────────────────────────────────

def _sys_hash(sys_text: str, tools) -> str:
    """Cache key for a (system_text, tools) pair.

    Tools affect the rendered system message (append_tools_to_system), so
    they're part of the cache key.
    """
    h = hashlib.sha256()
    h.update(sys_text.encode("utf-8"))
    h.update(b"\x00tools:")
    if tools:
        h.update(json.dumps(tools, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:24]


def needs_system(messages: list, tools=None) -> tuple[bool, str]:
    """Decide whether the system prompt is needed for this request.

    Returns (needs: bool, reason: str).  Conservative: returns True when
    skipping the system might degrade the answer.  Reasons are debug
    labels only; they never affect the output.

    Heuristics (checked in order, first match wins):
      0. tools=non-empty → include (tool defs live in system)
      1. multi-turn (>=2 user msgs OR >=1 assistant msg) → include
      2. obvious single-turn greeting / ack → skip
      3. tool/action keyword or code/path pattern → include
      4. interrogative question → include
      5. imperative verb → include
      6. ≤3-word input that is not a greeting/ack → include
      7. default → include
    """
    if tools:
        return True, "tools_provided"

    user_n = sum(1 for m in messages if m.get("role") == "user")
    asst_n = sum(1 for m in messages if m.get("role") == "assistant")
    if (user_n > 1) or (user_n > 0 and asst_n > 0):
        return True, f"multi_turn:{user_n}u/{asst_n}a"

    # Last user message (only for single-turn classification)
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if isinstance(content, str):
                last_user = content.strip()
            break
    if not last_user:
        return False, "no_user_message"

    lower = last_user.lower().strip()
    words = last_user.split()

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
        return False, f"greeting:{last_user!r}"

    # Single short alphabetic word (excluding obvious commands)
    COMMANDS = {
        "read", "write", "list", "show", "search", "find",
        "delete", "create", "update", "edit", "check",
        "run", "exec", "help", "start", "stop", "kill",
        "what", "where", "which", "who", "when", "how", "why",
        "send", "grep", "use", "act", "make", "give",
    }
    if (len(words) == 1 and len(lower) <= 5 and lower.isalpha()
            and lower not in COMMANDS):
        return False, f"short_word:{last_user!r}"

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
        if re.search(pat, last_user, re.IGNORECASE):
            return True, f"pattern:{pat!r}"

    INTERROG = {"how", "what", "why", "where", "when", "which", "who", "whose"}
    if words and words[0].lower().rstrip("?,.") in INTERROG:
        return True, f"question:{words[0]!r}"

    IMPERATIVE = {
        "do", "make", "let", "get", "use", "call", "tell", "show",
        "give", "find", "take", "put", "set", "go", "try", "keep",
        "send", "come", "look", "open", "close", "save", "quit",
        "explain", "describe", "help", "act", "be", "include",
        "write", "read", "run", "check", "list", "create",
    }
    if words and words[0].lower().rstrip("?.,") in IMPERATIVE:
        return True, f"imperative:{words[0]!r}"

    if len(words) <= 3:
        return True, f"short_ambig:{last_user!r}"

    return True, f"default:{last_user[:30]!r}"


def render_chatml_no_system(messages: list[dict]) -> str:
    """ChatML rendering with role=system messages dropped.

    The tool block is also omitted (no instructions to guide the model).
    Used when needs_system() returns False for a single-turn greeting/ack.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        content = content or ""
        if role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        elif role == "tool":
            parts.append(f"<|im_start|>user\n<tool_response>\n"
                         f"{content}\n</tool_response><|im_end|>\n")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def render_chatml(messages: list[dict], tools=None, tool_choice=None) -> str:
    """Map OpenAI messages[] to the ChatML prompt the engine expects.

    When tools are given they are rendered Qwen3-Hermes style into the first
    system message; assistant history renders <tool_call> blocks and role=='tool'
    messages render <tool_response> inside a user turn.
    """
    return render_chatml_messages(messages, tools=tools,
                                  tool_choice=tool_choice)


class _Stopped(Exception):
    pass


class ModelManager:
    """Multi-model, lazy-loading manager (v8+v9).

    Starts with ZERO models resident. `ensure(model_id)` loads on first use
    on the dedicated engine thread and fully frees the previous model
    (16 GB Macs cannot hold two 9B-class engines). `list_models()` reports
    the registry from headers only.

    v9 system-prompt KV cache:
      After a request that includes a system message, the engine's
      _pcache (fed_ids, kv_caches, gdn_states) holds the KV state for
      EVERYTHING prefilled that turn, including the system portion.
      We save the system portion under a hash key so the NEXT request
      with the same system hash can re-inject it: the engine's LCP check
      then matches the cached system tokens and skips the 4-5K prefill.

      The cache is cleared on model switch (different model = different
      KV layout) and on _free_engine.
    """
    def __init__(self, models_dirs: list | None = None):
        import queue
        from .model_registry import scan_models
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._dirs = models_dirs
        self._registry = scan_models(models_dirs)
        self._engine: Engine | None = None
        self._loaded_id: str | None = None
        self._load_error: Exception | None = None
        # v9: system-prompt KV cache.  Keyed by hash(system_text + tools).
        # Value = (sys_ids, sys_kv_caches, sys_gdn_states, n_sys_tokens).
        # The cache is set after each request that includes a system
        # message and injected before each request that needs the system.
        # It is cleared on model switch (different model = different KV layout).
        self._sys_cache: dict[str, tuple] = {}
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="atf-engine")
        self._thread.start()

    def sys_cache_stats(self) -> dict:
        """Return cache stats for /health or /v1/models debug."""
        total_tok = sum(entry[3] for entry in self._sys_cache.values())
        return {
            "entries": len(self._sys_cache),
            "total_sys_tokens": total_tok,
        }

    def _try_inject_sys_cache(self, sys_text: str, tools) -> bool:
        """Pre-warm engine._pcache with the cached system KV, if any.

        Returns True if a cache hit was applied.  On a hit the engine's
        existing LCP-based _pcache logic will skip prefilling the system
        portion of the next generate() call.

        v10: checks disk cache if in-memory cache misses.
        """
        if not sys_text:
            return False
        if self._engine is None:
            return False
        key = _sys_hash(sys_text, tools)
        entry = self._sys_cache.get(key)
        # v10: disk fallback
        if entry is None and self._loaded_id:
            disk = _syscache.load_cache(self._loaded_id, key)
            if disk is not None:
                sys_ids, sys_kv, sys_gdn = disk
                # Convert syscache's KVCache objects to engine KVCache objects
                from .engine import KVCache
                engine_kv = {}
                for b_, c in sys_kv.items():
                    if isinstance(c, KVCache):
                        engine_kv[b_] = c
                    else:
                        # syscache returned a dict-like object
                        kc = KVCache()
                        import mlx.core as mx
                        kc.n_kv = c.get("n_kv", 0)
                        kc.head_dim = c.get("head_dim", 0)
                        kc.n = c.get("n", 0)
                        k_buf = c.get("k_buf")
                        v_buf = c.get("v_buf")
                        if hasattr(k_buf, "shape"):
                            kc.k_buf = k_buf
                            kc.v_buf = v_buf
                        else:
                            kc.k_buf = mx.array(k_buf)
                            kc.v_buf = mx.array(v_buf)
                        engine_kv[b_] = kc
                entry = (sys_ids, engine_kv, sys_gdn, len(sys_ids))
                self._sys_cache[key] = entry
                log.info("sys cache DISK HIT  key=%s  n_sys=%d  engine=%s",
                         key, len(sys_ids), self._loaded_id)
        if entry is None:
            return False
        sys_ids, sys_kv, sys_gdn, n_sys = entry
        try:
            self._engine._pcache = (sys_ids, sys_kv, sys_gdn)
            log.info("sys cache HIT  key=%s  n_sys=%d  engine=%s",
                     key, n_sys, self._loaded_id)
            return True
        except Exception as exc:
            log.warning("sys cache inject failed: %s", exc)
            try:
                self._engine._pcache = None
            except Exception:
                pass
            return False

    def _save_sys_cache(self, sys_text: str, tools) -> bool:
        """Snapshot the engine's _pcache for the system portion, if any.

        Called AFTER engine.generate() succeeds.  The engine's _pcache
        holds the KV state for the entire prefilled prompt, including
        the system portion.  We extract the system portion by tokenizing
        the rendered system block and storing the prefix of the engine's
        _pcache that corresponds to those tokens.

        v10: also persists to disk so the cache survives server restarts.
        """
        if not sys_text:
            return False
        if self._engine is None:
            return False
        pcache = getattr(self._engine, "_pcache", None)
        if pcache is None:
            return False
        old_ids, old_kv, old_gdn = pcache
        try:
            tok = self._engine.tok
        except Exception:
            return False
        # Render just the system block in the same shape the engine saw.
        if tools:
            from .toolcall import append_tools_to_system
            body = append_tools_to_system(sys_text, tools)
        else:
            body = sys_text
        sys_block = f"<|im_start|>system\n{body}<|im_end|>"
        try:
            sys_ids = tok.encode(sys_block)
        except Exception as exc:
            log.warning("sys cache encode failed: %s", exc)
            return False
        n_sys = len(sys_ids)
        if n_sys < 1 or n_sys > len(old_ids):
            return False
        # Shallow-view: cap each KVCache's .n to n_sys.  The underlying
        # buffer is shared (zero copy) so memory cost is zero.  When the
        # engine next runs and the LCP check matches these n_sys tokens,
        # it will use the same buffer and skip prefilling those tokens.
        sys_kv = {}
        for block_idx, cache in old_kv.items():
            try:
                cache.n = min(cache.n, n_sys)
                sys_kv[block_idx] = cache
            except Exception:
                pass
        sys_gdn = dict(old_gdn) if old_gdn else {}
        key = _sys_hash(sys_text, tools)
        self._sys_cache[key] = (sys_ids, sys_kv, sys_gdn, n_sys)
        log.info("sys cache SAVE key=%s  n_sys=%d  engine=%s  total=%d",
                 key, n_sys, self._loaded_id, len(self._sys_cache))
        # v10: also persist to disk
        if self._loaded_id:
            _syscache.save_cache(self._loaded_id, key, sys_ids, sys_kv, sys_gdn)
        return True


    # -- registry (no weights touched) ------------------------------------
    @property
    def engine(self):
        return self._engine

    def list_models(self) -> list[dict]:
        return [{"id": m.id, "object": "model", "owned_by": "atf",
                 "format": f"v{m.format_major}",
                 "size_bytes": m.size_bytes,
                 "loaded": m.id == self._loaded_id}
                for m in self._registry]

    def resolve(self, ref: str | None):
        from .model_registry import resolve_model, default_models_dirs
        if not ref:
            return self._registry[0] if self._registry else None
        for m in self._registry:
            if m.id == ref:
                return m
        path = resolve_model(ref, self._dirs or default_models_dirs())
        if path is None:
            return None
        for m in self._registry:
            if m.path == path:
                return m
        from .model_registry import peek_header, ModelInfo
        major, minor = peek_header(path) or (1, 0)
        return ModelInfo(path.stem, path, major, minor, path.stat().st_size)

    # -- lazy load / unload ------------------------------------------------
    def ensure(self, info) -> Engine:
        """Load the model if not resident; frees any other model first."""
        import queue
        if self._engine is not None and self._loaded_id == info.id:
            return self._engine
        done = threading.Event()
        err: list = []
        self._queue.put({"op": "load", "info": info,
                         "done": done, "err": err})
        done.wait()
        if err:
            raise err[0]
        return self._engine

    def run(self, info, prompt: str, cfg: GenConfig, cb,
            sys_text: str | None = None, tools=None):
        """ensure() + generate, serialized on the engine thread.

        v9: if sys_text is provided, the engine thread will try to inject
        a cached system-prompt KV state BEFORE generate() and save the
        resulting state AFTER.  This lets the engine's LCP check skip
        prefilling the system portion of subsequent requests.
        """
        import queue
        job = {"op": "run", "info": info, "prompt": prompt, "cfg": cfg,
               "cb": cb, "done": threading.Event(), "result": None,
               "error": None,
               "sys_text": sys_text, "tools": tools}
        self._queue.put(job)
        job["done"].wait()
        if job["error"] is not None:
            raise job["error"]
        return job["result"]

    def _free_engine(self):
        eng = self._engine
        if eng is not None:
            try:
                eng._pcache = None     # drop pinned KV buffers (can be GBs)
            except Exception:
                pass
        self._engine = None
        self._loaded_id = None
        # v9: also clear the system-prompt cache since a different model
        # would have a different KV layout and the cached state is invalid.
        self._sys_cache.clear()
        import gc
        gc.collect()
        try:
            mx.synchronize()
        except Exception:
            pass
        # v9.2: release the wired-weight reservation, otherwise macOS keeps
        # multi-GB memory wired after unload (observed: 7.1 GB stuck wired).
        try:
            mx.set_wired_limit(0)
        except Exception:
            pass
        try:
            mx.clear_cache()
        except Exception:
            pass

    def _loop(self):
        import queue
        while True:
            job = self._queue.get()
            try:
                op = job.get("op")
                if op == "load":
                    info = job["info"]
                    if self._engine is not None and self._loaded_id == info.id:
                        pass                      # already resident
                    else:
                        if self._engine is not None:
                            self._free_engine()
                        model = load_atf(info.path)
                        tok = Tokenizer.from_hf("Qwen/Qwen3.5-9B")
                        self._engine = Engine(model, tok)
                        self._loaded_id = info.id
                elif op == "unload":
                    self._free_engine()
                elif op == "run":
                    info = job["info"]
                    if self._engine is None or self._loaded_id != info.id:
                        if self._engine is not None:
                            self._free_engine()
                        model = load_atf(info.path)
                        tok = Tokenizer.from_hf("Qwen/Qwen3.5-9B")
                        self._engine = Engine(model, tok)
                        self._loaded_id = info.id
                    # v9: try to inject cached system KV before generate().
                    # If cache misses, the engine will full-prefill (normal).
                    sys_text = job.get("sys_text")
                    if sys_text is not None:
                        hit = self._try_inject_sys_cache(sys_text, job.get("tools"))
                        log.info("sys_prefill  sys_text=%r  cache_hit=%s",
                                 (sys_text[:50] + "...") if len(sys_text) > 50 else sys_text,
                                 hit)
                    job["result"] = self._engine.generate(
                        job["prompt"], job["cfg"], stream=False,
                        token_cb=job["cb"])
                    # v9: save the system KV state after generate().  The
                    # engine's _pcache now holds the full prefilled prompt.
                    if sys_text is not None:
                        self._save_sys_cache(sys_text, job.get("tools"))
            except BaseException as exc:      # noqa: BLE001
                if job.get("op") == "run":
                    job["error"] = exc
                else:
                    job.setdefault("err", []).append(exc)
            finally:
                job["done"].set()

def create_app(models_dirs=None, default_model: str | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="ATF OpenAI-compatible server")
    gen_lock = threading.Lock()          # serialize concurrent requests
    # v8: lazy multi-model manager -- nothing is loaded until first use.
    worker = ModelManager(models_dirs)
    _default = default_model

    def _build_config(req: dict, messages: list[dict]) -> tuple[GenConfig, str]:
        """GenConfig from request; router fills adaptive defaults.

        Routes on the LAST USER MESSAGE only -- scoring the full rendered
        ChatML prompt (system text + template tokens) inflates the
        difficulty score and needlessly picks slow thinking tiers.
        """
        user_text = next((m.get("content", "") for m in reversed(messages)
                          if m.get("role") == "user"), "")
        if isinstance(user_text, list):
            user_text = "".join(p.get("text", "") for p in user_text
                                if isinstance(p, dict))
        thinking = req.get("thinking")
        tier_note = ""
        if thinking is None:
            from .router import route, BADGES
            tier, _diff = route(user_text)
            thinking = tier.thinking
            max_tok = tier.max_tokens
            temp = tier.temperature
            tier_note = f"router: {BADGES[tier.level]} ({tier.name})"
        else:
            max_tok = 4096
            temp = 0.0   # gamma/v4 (2026-09-02): greedy default
        cfg = GenConfig(
            # gamma/v4 (2026-09-02): cap raised from 8192 to 65536 to match
            # the bridge + UI defaults. The user explicitly asked for
            # no artificial truncation; the engine stops on EOS or
            # when the user clicks Stop.
            max_tokens=min(int(req.get("max_tokens") or max_tok), 65536),
            temperature=float(req.get("temperature", temp)),
            top_p=float(req.get("top_p", 0.9)),
            repeat_penalty=1.2,
            exact_ffn=True,
            thinking=thinking,
            mem_log_interval=0,
            # gamma/v7 persistence (ATF_KV_SSD_PERSIST): best-effort guess at
            # request-build time (before the model is necessarily resolved/
            # loaded for THIS request) -- falls back to whatever is already
            # resident. Matches the `req.get("model") or _default or
            # worker._loaded_id` pattern used elsewhere in this file.
            model_id=req.get("model") or _default or worker._loaded_id,
        )
        return cfg, tier_note

    def _generate_ids(req: dict, cfg: GenConfig, prompt: str,
                      stop: threading.Event,
                      on_delta=None,
                      sys_text: str | None = None,
                      tools=None) -> tuple[str, str, bool]:
        """Run generation; returns (content, reasoning, stopped).

        Reasoning (<think>...</think>) is kept separate from the answer so
        OpenAI clients get clean 'content' plus an optional
        'reasoning_content' field (DeepSeek-style convention).
        on_delta(kind, text) streams kind='reasoning' | 'content'.
        """
        think_open_id, think_close_id = 248068, 248069
        # engine force-opens <think> during prefill when thinking != off;
        # that token never passes through the callback.
        in_reasoning = cfg.thinking != "off"
        reasoning_parts: list[str] = []
        # v18 perf fix: decode only the small pending window, not the whole
        # reply-so-far, on every token. The old code called
        # tok.decode(gen_ids) with gen_ids growing every step, which made
        # streaming cost O(n^2) in reply length -- worst on the 27B, whose
        # replies are already the slowest to produce token-by-token. A
        # clean (no U+FFFD) decode of the pending window is final text: BPE
        # decode only needs enough adjacent tokens to complete a split
        # multi-byte glyph, it never retroactively changes text already
        # resolved by an earlier decode.
        pending_ids: list[int] = []

        def _emit(kind: str, delta: str):
            if on_delta is not None and delta:
                on_delta(kind, delta)

        def cb(tid: int):
            nonlocal in_reasoning
            if stop.is_set():
                raise _Stopped()
            if tid == think_close_id:
                in_reasoning = False
                return
            if tid == think_open_id:
                in_reasoning = True
                return
            pending_ids.append(tid)
            delta = worker.engine.tok.decode(pending_ids)  # engine resident while run() in flight
            if "\ufffd" in delta:
                return   # glyph incomplete; wait for the next token
            pending_ids.clear()
            if not delta:
                return
            if in_reasoning:
                reasoning_parts.append(delta)
                _emit("reasoning", delta)
            else:
                _emit("content", delta)

        minfo = worker.resolve(req.get("model") or _default)
        if minfo is None:
            raise HTTPException(404, f"unknown model: {req.get('model')!r}; "
                                     f"available: {[m['id'] for m in worker.list_models()]}")
        text = worker.run(minfo, prompt, cfg, cb,
                             sys_text=sys_text, tools=tools)
        # engine's returned text still embeds "...reasoning</think>\n\n";
        # keep only the post-</think> answer as content.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].lstrip("\n")
        for s in req.get("stop") or []:
            if isinstance(s, str) and s and s in text:
                text = text.split(s, 1)[0]
        return text, "".join(reasoning_parts), False

    @app.post("/v1/chat/completions")
    def chat(req: dict):
        messages = req.get("messages")
        if not messages:
            raise HTTPException(400, "messages[] is required")
        # Normalize tools: prefer "tools"; accept legacy "functions"
        _tools = req.get("tools")
        if not _tools and isinstance(req.get("functions"), list):
            _tools = [{"type": "function", "function": fn}
                      for fn in req["functions"]]
        # v9: extract the system text for cache-key purposes
        sys_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict)
                    )
                if isinstance(content, str):
                    sys_text += content
        # v9: smart router — does this turn actually need the system prompt?
        # Single-turn greetings / acks skip it (zero prefill). Everything
        # else includes it, with the KV state cached so the 4-5K prefill
        # is skipped on every request after the first in a session.
        needs, sys_reason = needs_system(messages, _tools)
        if needs:
            prompt = render_chatml(messages, _tools, req.get("tool_choice"))
        else:
            prompt = render_chatml_no_system(messages)
        log.info("sys_router  needs=%s  reason=%s  tools=%s  "
                 "sys_text_len=%d  model=%s",
                 needs, sys_reason, bool(_tools), len(sys_text),
                 req.get("model") or _default or "")

        try:
            cfg, tier_note = _build_config(req, messages)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"bad request: {exc}") from exc

        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        stream = bool(req.get("stream"))
        # Only pass sys_text to the engine when needs_system is True.
        # For greetings, the engine prefill is just the user text — no
        # system tokens at all, so the cache is irrelevant.
        _sys_for_engine = sys_text if needs else None

        if not stream:
            with gen_lock:
                try:
                    text, reasoning, _stopped = _generate_ids(
                        req, cfg, prompt, threading.Event(),
                        sys_text=_sys_for_engine, tools=_tools)
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(500, str(exc)) from exc
            # Strip any <tool_call> markup the model emitted; surface it as
            # OpenAI tool_calls instead of leaking raw JSON into content.
            content, calls, invalid = parse_output(text, disabled=not _tools)
            if invalid:
                log.warning("dropped %d unparsable tool call block(s)",
                            len(invalid))
            msg = {"role": "assistant",
                   "content": (content + "\n[invalid tool call]")
                              if (invalid and content) else
                              ("[invalid tool call]" if invalid else content)}
            if calls:
                msg["tool_calls"] = format_openai_tool_calls(calls)
            if reasoning:
                msg["reasoning_content"] = reasoning
            finish = "tool_calls" if calls else "stop"
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": req.get("model") or _default or (worker._loaded_id or ""),
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": len(worker.engine.tok.encode(prompt)),
                    "completion_tokens": len(text) // 4,  # approx
                    "total_tokens": len(worker.engine.tok.encode(prompt)) + len(text) // 4,
                },
            }

        # ---- SSE streaming ----
        def sse():
            import queue as _q
            stop = threading.Event()
            q: _q.Queue = _q.Queue()
            _SENTINEL = object()
            # Swallows <tool_call> markup from the stream and collects calls;
            # holds back partial-tag prefixes split across chunk boundaries.
            tools_parser = ToolCallStreamParser(disabled=not _tools)
            final_finish = ["stop"]

            def chunk(delta: dict, finish=None) -> str:
                payload = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.get("model") or _default or (worker._loaded_id or ""),
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}],
                }
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            def on_delta(kind: str, delta: str):
                if not stop.is_set():
                    if kind != "content":
                        q.put(chunk({"reasoning_content": delta}))
                        return
                    safe = tools_parser.feed(delta)
                    if safe:
                        q.put(chunk({"content": safe}))

            try:
                yield chunk({"role": "assistant"})
                def run():
                    try:
                        with gen_lock:
                            _generate_ids(req, cfg, prompt, stop, on_delta,
                                          sys_text=_sys_for_engine,
                                          tools=_tools)
                    except _Stopped:
                        pass
                    except Exception as exc:  # noqa: BLE001
                        q.put(chunk({"content": f"\n[error: {exc}]"}))
                    finally:
                        trailing, _ = tools_parser.finish()
                        if trailing:
                            q.put(chunk({"content": trailing}))
                        if tools_parser.invalid:
                            log.warning("dropped %d unparsable tool call "
                                        "block(s)", len(tools_parser.invalid))
                        if tools_parser.has_tool_calls:
                            q.put(chunk({"tool_calls":
                                         format_openai_tool_calls(
                                             tools_parser.calls)}))
                            final_finish[0] = "tool_calls"
                gen_thread = threading.Thread(target=run, daemon=True)
                gen_thread.start()
                while True:
                    item = q.get()
                    if item is _SENTINEL:
                        break
                    yield item
                yield chunk({}, finish=final_finish[0])
                yield "data: [DONE]\n\n"
            finally:
                stop.set()  # unblock the engine thread if client disconnects

        return StreamingResponse(sse(), media_type="text/event-stream")


    return app




def main() -> None:
    ap = argparse.ArgumentParser(description="ATF OpenAI-compatible server")
    ap.add_argument("model", nargs="?", default=None,
                    help="default model id/path (loads lazily on first use)")
    ap.add_argument("--models-dir", action="append", default=None,
                    help="directory to scan for .atf models (repeatable)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn
    uvicorn.run(create_app(args.models_dir, args.model),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
