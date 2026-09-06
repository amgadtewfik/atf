"""ATF inference engine (MLX backend).

Implements the Qwen3.5 hybrid forward pass, matching llama.cpp's
qwen35 graph (src/models/qwen35.cpp + src/models/delta-net-base.cpp):

Full-attention blocks:
  - attn_q.weight outputs PER-HEAD INTERLEAVED [q_h0, gate_h0, q_h1, gate_h1, ...]
    (each head contributes head_dim q values followed by head_dim gate values)
  - QK-norm (per-head RMSNorm) -> RoPE (first rope_dim dims) -> GQA attention
  - output gate applied BEFORE the wo projection: wo(attn * sigmoid(gate))

GatedDeltaNet blocks:
  - fused attn_qkv -> causal conv (SiLU) -> l2norm(q,k) -> q scaled by 1/sqrt(dk)
  - decay g = ssm_a * softplus(ssm_alpha @ x + ssm_dt.bias)
  - beta = sigmoid(ssm_beta @ x)
  - delta-rule state recurrence, gated RMSNorm output multiplied by silu(z)

FFN: dense SwiGLU stored as 8 chunk-experts; exact mode sums all chunks
(which reproduces the original dense FFN), sparse mode routes top-k.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from rich.console import Console

from .model import AtfModel, gpu_mem_line, host_mem_line
from .tokenizer import Tokenizer

console = Console()

# ── Verbose + NaN-trap diagnostics ────────────────────────────────────────
# Controlled by env vars:
#   ATF_VERBOSE=<0..4>  -- 0 = silent (default; only the existing rich
#                         console.status lines from earlier code paths).
#                         1 = per-stage summary (prefill / decode totals,
#                         per-request summary, MoE aggregate stats).
#                         2 = + per-chunk prefill progress, per-block MoE
#                         stats (one line per block per stage).
#                         3 = + per-token decode timing / logits stats /
#                         expert selection per token.
#                         4 = + NaN-check every forward output, dump a few
#                         dequantized expert slices, and trace per-RoPE
#                         per-block kernel timings. Very chatty.
#                         All [atf-verbose] lines go to STDERR so the
#                         bridge's StreamShim doesn't swallow them into
#                         JSON status events.
#   ATF_NAN_TRAP=1      -- raise on the first NaN or Inf seen in logits,
#                         expert weights after dequant, router scores, or
#                         attention output during forward. Reports the
#                         block / stage so the offending op is obvious in
#                         the traceback.
#   ATF_NAN_WARN=1      -- like ATF_NAN_TRAP but just prints [atf-nan] lines
#                         (does not raise). Safer for production-ish runs.
# All three flags are independent; you can run any combination.
import sys as _sys
import os as _os
import time as _time


def _parse_verbose() -> int:
    """Parse ATF_VERBOSE into an int 0..4 (clamped). Defaults to 0."""
    raw = _os.environ.get("ATF_VERBOSE", "0").strip()
    if not raw:
        return 0
    # Accept either an integer ("3") or a true-style value ("true"/"on"/"yes")
    lower = raw.lower()
    if lower in ("true", "on", "yes", "y"):
        return 4   # be generous: a bare "ATF_VERBOSE=1" gets full debug
    try:
        return max(0, min(4, int(raw)))
    except ValueError:
        return 0


_VERBOSE = _parse_verbose()
_NAN_TRAP = _os.environ.get("ATF_NAN_TRAP") == "1"
_NAN_WARN = _os.environ.get("ATF_NAN_TRAP") == "1" or _os.environ.get("ATF_NAN_WARN") == "1"


def _vlog(level: int, msg: str) -> None:
    """Emit a verbose log line at the given level (1..4). No-op if level
    is above the current _VERBOSE threshold. Output goes to STDERR with an
    [atf-verbose] prefix so it is never mixed into rich.Console stream
    output (which is captured by the bridge StreamShim and turned into
    status events)."""
    if level > _VERBOSE:
        return
    try:
        _sys.stderr.write(f"[atf-verbose L{level}] {_time.strftime('%H:%M:%S')} {msg}\n")
        _sys.stderr.flush()
    except Exception:
        pass


# ── Per-process accumulators for the verbose summary at end of generate().
# Reset on each new generate() call. All values are intentionally simple so
# they survive a `dict()` copy without leaking MLX arrays or closures.
_VERBOSE_STATE = {
    "prefill_chunks": 0,        # number of prefill chunks processed
    "prefill_tokens": 0,        # tokens processed by prefill (excludes LCP skip)
    "prefill_t": 0.0,           # total seconds spent inside _forward_prefill
    "prefill_block_ms_total": 0.0,  # sum of per-block times during prefill
    "prefill_block_calls": 0,   # number of (block, prefill-step) calls timed
    "decode_steps": 0,          # number of decode tokens produced
    "decode_t": 0.0,            # total seconds in the decode loop (per-token)
    "decode_block_ms_total": 0.0,
    "decode_block_calls": 0,
    "sample_t": 0.0,            # total seconds spent in _sample
    "sample_logits_nan": 0,     # count of sample calls that hit NaN/Inf logits
    "sample_logits_inf": 0,
    "moe_blocks_seen": 0,       # how many MoE blocks ran
    "moe_unique_experts_sum": 0,    # sum of unique-experts across MoE blocks
    "moe_unique_experts_max": 0,    # max unique-experts in any single block
    "moe_cache_misses_sum": 0,      # sum of expert-cache misses across MoE blocks
    "moe_t": 0.0,               # total seconds in _ffn_moe
    "first_token_t": None,      # wall-clock seconds from generate() start to first decode token
    "generate_t0": None,        # wall-clock seconds when generate() started
    "decode_step_ms_avg": None, # average ms per decode step (decoded count)
    "decode_step_ms_max": None, # max ms of any single decode step
}


def _vstate_reset() -> None:
    for k, v in _VERBOSE_STATE.items():
        if isinstance(v, list):
            _VERBOSE_STATE[k] = []
        elif isinstance(v, bool):
            _VERBOSE_STATE[k] = False
        elif isinstance(v, int):
            _VERBOSE_STATE[k] = 0
        elif isinstance(v, float):
            _VERBOSE_STATE[k] = 0.0
        else:
            _VERBOSE_STATE[k] = None


def _vstate_set(key: str, value) -> None:
    if key in _VERBOSE_STATE:
        _VERBOSE_STATE[key] = value


def _vstate_add(key: str, delta) -> None:
    if key in _VERBOSE_STATE and isinstance(_VERBOSE_STATE[key], (int, float)):
        _VERBOSE_STATE[key] += delta


def _vstate_summary() -> dict:
    """Snapshot the per-run accumulators as a plain dict for trace output."""
    s = dict(_VERBOSE_STATE)
    if s["prefill_block_calls"]:
        s["prefill_block_avg_ms"] = s["prefill_block_ms_total"] / s["prefill_block_calls"]
    if s["decode_block_calls"]:
        s["decode_block_avg_ms"] = s["decode_block_ms_total"] / s["decode_block_calls"]
    if s["decode_steps"]:
        total_ms = s["decode_t"] * 1000.0
        s["decode_step_ms_avg"] = total_ms / s["decode_steps"]
    return s


def _nan_check(arr, where: str, *, raise_on_nan=None):
    """Check an MLX array (or scalar) for NaN/Inf. Returns a stats dict.

    `where` is a short human label included in any log line / exception.
    `raise_on_nan` defaults to _NAN_TRAP; pass False to demote to warn-only.
    Never raises on the array-evaluation step -- if we cannot bring the
    array to host we report `eval_error` and move on.
    """
    if raise_on_nan is None:
        raise_on_nan = _NAN_TRAP
    out = {"where": where, "shape": None, "dtype": None,
           "nan": 0, "inf": 0, "min": None, "max": None, "mean": None,
           "eval_error": None}
    if arr is None:
        return out
    try:
        import numpy as _np
        try:
            shape = tuple(arr.shape) if hasattr(arr, "shape") else ()
        except Exception:
            shape = ()
        out["shape"] = shape
        out["dtype"] = str(getattr(arr, "dtype", ""))
        try:
            mx.eval(arr)
        except Exception as exc:
            out["eval_error"] = f"eval: {type(exc).__name__}: {exc}"
            return out
        try:
            host = _np.asarray(arr)
        except Exception as exc:
            out["eval_error"] = f"asarray: {type(exc).__name__}: {exc}"
            return out
        if host.size == 0:
            return out
        host_f = host.astype(_np.float32, copy=False) if host.dtype.kind == "f" else host
        nan_mask = _np.isnan(host_f)
        inf_mask = _np.isinf(host_f)
        n_nan = int(nan_mask.sum())
        n_inf = int(inf_mask.sum())
        out["nan"] = n_nan
        out["inf"] = n_inf
        finite_mask = ~nan_mask & ~inf_mask
        finite = host_f[finite_mask] if (n_nan or n_inf) else host_f.ravel()
        if finite.size:
            out["min"] = float(finite.min())
            out["max"] = float(finite.max())
            out["mean"] = float(finite.mean())
        if (n_nan or n_inf) and _NAN_WARN:
            _sys.stderr.write(
                f"[atf-nan] {where}: shape={out['shape']} "
                f"nan={n_nan} inf={n_inf} "
                f"min={out['min']} max={out['max']} mean={out['mean']}\n")
            _sys.stderr.flush()
        if (n_nan or n_inf) and raise_on_nan:
            raise RuntimeError(
                f"[ATF_NAN_TRAP] NaN/Inf detected at {where}: "
                f"shape={out['shape']} nan={n_nan} inf={n_inf} "
                f"min={out['min']} max={out['max']} mean={out['mean']}")
    except RuntimeError:
        raise
    except Exception as exc:
        out["eval_error"] = f"{type(exc).__name__}: {exc}"
    return out


# v9.2 safety rail: hard ceiling on total KV-cache GPU memory. Whatever the
# context slider says, KV can never exceed this -- protects 16 GB machines
# from the eager-allocation freeze class of failures forever.
#
# v9.2: the budget is no longer a blind constant. The v9 GPU hang was exactly
# this: 9.5 GB packed weights + a 3 GiB KV allowance + activations exceeded
# what fits, and Metal's command buffer waited on an IOSurface signal that
# never came (silent infinite hang). Wiring the weights makes the same OOM a
# clean, immediate kIOGPUCommandBufferCallbackErrorOutOfMemory instead.
DEFAULT_MAX_KV_GPU_BYTES = int(3 * (1 << 30))
MIN_KV_GPU_BYTES = int(256 * (1 << 20))
# Headroom for activations + Metal internal buffers on top of wired weights.
_ACT_HEADROOM_BYTES = int(768 * (1 << 20))


def kv_budget_bytes(packed_bytes: int = 0) -> int:
    """KV-cache GPU budget derived from the device's actual working set.

    budget = max_recommended_working_set - (wired weights + headroom),
    clamped to [256 MiB, 3 GiB]. On the 16 GB / 27B machine this yields
    ~2.2 GiB (vs the old flat 3 GiB that caused the hang).
    """
    try:
        info = mx.device_info()
        ws = int(info.get("max_recommended_working_set_size", 0))
    except Exception:
        ws = 0
    if not ws:
        return DEFAULT_MAX_KV_GPU_BYTES
    budget = ws - int(packed_bytes) - _ACT_HEADROOM_BYTES
    return max(MIN_KV_GPU_BYTES,
               min(DEFAULT_MAX_KV_GPU_BYTES, budget))


def apply_wired_limit(packed_bytes: int) -> int:
    """Wire the resident packed weights so macOS cannot page them out.

    Unwired multi-GB weight allocations are what turned an ordinary OOM into
    an unkillable-looking GPU hang (command buffer stuck in
    IOSurfaceSharedEvent waitUntilSignaledValue). Wired memory is exempt
    from paging; when memory genuinely runs out Metal now fails fast with
    kIOGPUCommandBufferCallbackErrorOutOfMemory instead of hanging.

    Returns the wired byte count actually applied (may be capped by the
    device's max recommended working set).
    """
    try:
        info = mx.device_info()
        ws = int(info.get("max_recommended_working_set_size", 0))
        # Small margin so the wiring request itself is always accepted.
        want = min(int(packed_bytes * 1.02) + (64 << 20), max(0, ws - (256 << 20)))
        mx.set_wired_limit(want)
        return want
    except Exception as e:      # older MLX or non-Metal fallback: don't block
        console.print(f"[yellow]  (wired limit unavailable: {e})[/yellow]")
        return 0


# v9.1 belt-and-braces: MLX-level active-memory limit. If anything ever
# leaks again, MLX GC kicks in past this and raises instead of swapping the
# machine to death.
MAX_MLX_ACTIVE_BYTES = int(12.5 * (1 << 30))   # ~12.5 GiB of 16 GB


@dataclass
class GenConfig:
    max_tokens: int = 128
    # gamma/v4 (2026-09-02): default flipped to greedy (temperature=0.0)
    # for the fastest, most deterministic, and most reproducible decode.
    # Sampling (temperature>0) remains available per-request; the bridge,
    # server, and router all forward the user's explicit temperature when
    # provided, and use 0.0 as the fallback.
    temperature: float = 0.0
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    seed: int | None = None
    mem_log_interval: int = 20
    exact_ffn: bool = True
    diag: bool = False
    # thinking: "off" | "low" | "medium" | "high" | "xhigh"
    #   off    = empty <think> block in the prompt, model answers directly
    #   low    = force-close reasoning after 256 tokens
    #   medium = 1024
    #   high   = 4096
    #   xhigh  = unlimited (capped by max_tokens)
    thinking: str = "medium"
    # v7: max context (prompt + generated) and prefill chunk size.
    # KV memory = n_full_attn * 2 * n_kv * head_dim * max_context * 4 bytes
    # (8 * 2 * 4 * 256 * 65536 * 4 = ~268 MB here -- cheap).
    max_context: int = 65536
    prefill_chunk: int = 1024
    # gamma/v3.1 (SPEC_DECODING_PROPOSAL.md Phase 2): shallow self-
    # speculative decoding. spec_draft="shallow" enables Engine._spec_step
    # in the decode loop (Medusa/EAGLE-style early-exit draft through the
    # first N blocks, verified by one batched forward through all blocks).
    # None (default) = no change to existing decode behavior. spec_k is
    # the number of tokens drafted per cycle.
    # gamma/v4 (gamma/v4 bench 2026-09-02): shallow self-spec is a
    # NET LOSS on the 9B and 27B (.50-.85x of plain) because the
    # first-quarter-block drafter is too weak to predict useful
    # tokens. Two safer defaults:
    #   1. spec_min_acceptance=0.95: the EMA-tracked accept rate
    #      will fall below this on the first bad cycle and trigger
    #      auto-disable (revert to plain decode). 0.30 was way too
    #      low and let bad spec cycles keep running.
    #   2. spec_draft=None: off by default; users explicitly opt in
    #      for experimental spec decoding.
    # A future cross-engine 9B-on-27B design (SPEC_DECODING_PROPOSAL.md
    # option 1) would replace this, but that is blocked on 16 GB
    # residency and a real Qwen4 MTP-aware converter.
    spec_draft: str | None = None
    spec_k: int = 4
    spec_min_acceptance: float = 0.95
    # gamma/v7: Paged SSD KV cache.  Off-loads cold KV pages to an mmap-
    # backed file on SSD, keeping only the N hottest pages on GPU.  Unlocks
    # long context on memory-constrained Macs (e.g. 16 GB unified, 27B model).
    # ATF_KV_SSD=1 to enable, ATF_KV_SSD_PATH overrides the cache directory,
    # ATF_KV_SSD_HOT_PAGES controls how many pages stay resident on GPU.
    kv_ssd_enabled: bool = os.environ.get("ATF_KV_SSD", "0") == "1"
    kv_ssd_path: str | None = os.environ.get("ATF_KV_SSD_PATH", None)
    kv_ssd_hot_pages: int = int(os.environ.get("ATF_KV_SSD_HOT_PAGES", "256"))
    # gamma/v7 persistence: keep the paged .kvmm files (+ a small GDN-state
    # sidecar) on disk across process restarts and reuse them when the same
    # conversation resumes, instead of deleting on close(). OFF by default
    # -- opt-in until validated on real MLX/Metal, same convention this
    # codebase already uses for ATF_GDN_SCAN / ATF_COMPILE_STEP, because a
    # reuse bug here manifests as garbled output, not a crash. See
    # /Users/amgad/Desktop/Ai/Claude/atf-gamma-v7-ssd-cache-ui/notes.md.
    kv_ssd_persist: bool = os.environ.get("ATF_KV_SSD_PERSIST", "0") == "1"
    # Caller-supplied identifier for the currently-loaded model (e.g. the
    # registry id the bridge/server already track). Only used to scope
    # kv_ssd_persist's on-disk cache keys so two different models never
    # collide on the same session key; harmless if left None (persistence
    # just won't find/save anything).
    model_id: str | None = None


THINK_BUDGETS = {"off": 0, "low": 256, "medium": 1024, "high": 4096, "xhigh": None}
THINK_OPEN = 248068   # <think>
THINK_CLOSE = 248069  # </think>


@dataclass
class KVCache:
    """Preallocated contiguous KV buffer (v7).

    k_buf/v_buf: [capacity, n_kv_heads, head_dim], written in place.
    Replaces the old list-of-per-token-arrays, whose mx.stack() over the
    whole history on every decode step made long contexts unusable.
    Buffers are allocated lazily on first write so non-attention blocks
    (which never receive a cache) cost nothing.

    gamma/v7 SSD mode: when ``ssd_config`` is set (via GenConfig.kv_ssd_*),
    all storage is delegated to a ``PagedKV`` instance from ``kvstore.py``
    that keeps the hottest pages on GPU and evicts cold pages to an
    mmap-backed file on SSD.
    """
    capacity: int = 0              # hard cap (GenConfig.max_context)
    alloc: int = 0                 # tokens currently allocated (grows on demand)
    n_kv: int = 0
    head_dim: int = 0
    n: int = 0                      # tokens currently cached
    k_buf: mx.array | None = None
    v_buf: mx.array | None = None
    # SSD-paged backend (None = use flat buffers above)
    _paged: object = None  # PagedKV | None (set by enable_ssd)

    # v9.1: buffers start small and double as needed. The old code eagerly
    # allocated the FULL max_context capacity (e.g. 16 attn blocks x 0.25 GiB
    # = 4 GiB on the 27B model) on the very first token, which pushed
    # weights + KV past physical RAM and froze the machine before a single
    # token was emitted.
    GROW_TOKENS = 2048

    def enable_ssd(self, *, capacity: int, n_kv: int, head_dim: int,
                   hot_pages: int = 256, mmap_path: str | None = None,
                   session_id: str | None = None, resume_n: int = 0,
                   persist: bool = False):
        """Switch this cache to SSD-backed paged mode.

        gamma/v7 persistence (ATF_KV_SSD_PERSIST): `persist=True` keeps the
        .kvmm file on disk after close() instead of deleting it; `resume_n`
        tells a freshly-created PagedKV that the file already holds valid
        data for the first `resume_n` tokens (from a prior process), so it
        should read those pages back rather than zero-initializing them.
        """
        from .kvstore import PagedKV, PAGE
        capacity_pages = max(1, (capacity + PAGE - 1) // PAGE)
        self._paged = PagedKV(
            n_kv=n_kv, head_dim=head_dim,
            capacity_pages=capacity_pages, hot_pages=hot_pages,
            mmap_path=mmap_path, session_id=session_id,
            resume_n=resume_n, persist=persist,
        )
        self.capacity = capacity
        self.n_kv = n_kv
        self.head_dim = head_dim
        if resume_n > 0:
            self.n = resume_n

    @property
    def is_ssd(self) -> bool:
        return self._paged is not None

    def _ensure(self, capacity: int, n_kv: int, head_dim: int):
        if self._paged is not None:
            # SSD mode: PagedKV manages its own buffers; just sync n_kv/hd
            # for callers that read them (e.g. _attention_full).
            self.n_kv = n_kv
            self.head_dim = head_dim
            return
        if self.k_buf is None:
            self.capacity = capacity
            self.alloc = min(capacity, self.GROW_TOKENS)
            self.n_kv = n_kv
            self.head_dim = head_dim
            # v8: float16 halves KV bandwidth/footprint (decode is
            # memory-bound); values are cast back to fp32 on read for
            # numerically stable softmax.
            self.k_buf = mx.zeros((self.alloc, n_kv, head_dim), dtype=mx.float16)
            self.v_buf = mx.zeros((self.alloc, n_kv, head_dim), dtype=mx.float16)

    def _grow(self, needed: int):
        if self._paged is not None:
            return  # PagedKV handles growth internally
        if needed <= self.alloc:
            return
        new_alloc = min(self.capacity, max(needed, self.alloc * 2))
        if new_alloc <= self.alloc:
            return
        nk, nk_, hd = self.n_kv, self.n_kv, self.head_dim
        k_new = mx.zeros((new_alloc, nk, hd), dtype=mx.float16)
        v_new = mx.zeros((new_alloc, nk_, hd), dtype=mx.float16)
        if self.n:
            k_new[:self.n] = self.k_buf[:self.n]
            v_new[:self.n] = self.v_buf[:self.n]
        self.k_buf = k_new
        self.v_buf = v_new
        self.alloc = new_alloc

    def append(self, k: mx.array, v: mx.array):
        """k, v: [T, n_kv, head_dim] (post-RoPE)."""
        if self._paged is not None:
            self._paged.append(k, v)
            self.n = self._paged.n
            return
        T = k.shape[0]
        k = k.astype(mx.float16)
        v = v.astype(mx.float16)
        if self.n + T > self.capacity:
            raise RuntimeError(
                f"KV cache overflow: {self.n}+{T} > {self.capacity} tokens "
                f"(raise GenConfig.max_context)")
        self._grow(self.n + T)
        self.k_buf[self.n:self.n + T] = k
        self.v_buf[self.n:self.n + T] = v
        self.n += T

    def keys(self) -> mx.array:
        if self._paged is not None:
            return self._paged.keys()
        return self.k_buf[:self.n].astype(mx.float32)

    def values(self) -> mx.array:
        if self._paged is not None:
            return self._paged.values()
        return self.v_buf[:self.n].astype(mx.float32)

    def close(self):
        """Release SSD resources (mmap, file). No-op for flat-buffer mode."""
        if self._paged is not None:
            self._paged.close()
            self._paged = None


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def _rmsnorm(x: mx.array, weight: mx.array) -> mx.array:
    rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    return (x / rms) * weight


def _softmax(x: mx.array, axis: int = -1) -> mx.array:
    x_max = mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(x - x_max)
    return e / mx.sum(e, axis=axis, keepdims=True)


def _apply_rope(x: mx.array, positions: mx.array, freqs: mx.array) -> mx.array:
    """x: [T, n_heads, head_dim]; rotates first rope_dim dims."""
    angles = positions[:, None].astype(mx.float32) * freqs[None, :]
    cos_a = mx.cos(angles)[:, None, :]
    sin_a = mx.sin(angles)[:, None, :]
    rd = freqs.shape[0] * 2
    half = rd // 2
    x1 = x[:, :, :half]
    x2 = x[:, :, half:rd]
    r1 = x1 * cos_a - x2 * sin_a
    r2 = x2 * cos_a + x1 * sin_a
    if rd < x.shape[-1]:
        return mx.concatenate([r1, r2, x[:, :, rd:]], axis=-1)
    return mx.concatenate([r1, r2], axis=-1)


class Engine:
    # v8: GDN sequence chunk size for the block-parallel scan. 64 keeps the
    # CxC solve tiny while cutting Python-dispatched steps by ~64x vs the
    # old per-timestep loop.
    GDN_CHUNK = 64

    def __init__(self, model: AtfModel, tokenizer: Tokenizer | None = None):
        self.model = model
        self.tok = tokenizer
        h = model.header
        self.hidden = h.hidden_dim
        self.num_blocks = h.num_blocks
        self.num_heads = h.num_heads
        self.num_kv = h.num_kv_heads
        self.head_dim = self.hidden // self.num_heads
        self.num_experts = h.num_experts_per_block
        self.top_k = h.top_k

        self.ssm_blocks: set[int] = set()
        self.full_attn_blocks: set[int] = set()
        # gamma/v3: QSA (Qwen Sparse Attention) blocks. A block is QSA
        # iff it carries an `attn_indexer.q_proj.weight` tensor (the
        # indexer is the signature -- main-attn q/k/v also exist for
        # QSA blocks, same shape as full-attn, but the indexer is what
        # makes the path different). The autodetect probe ORDER matters:
        # QSA must be checked BEFORE the `attn_q.weight` probe below,
        # otherwise QSA blocks get misclassified as full_attn.
        # See docs/qwen38_flash_next_analysis.md §3.3 + §3.4.
        self.qsa_blocks: set[int] = set()
        # v19.1 true-MoE discovery, same tensor-name-probing pattern as the
        # hybrid attention split above: a block is a routed-MoE block iff it
        # carries a router (`ffn_gate_inp`) tensor.
        self.moe_blocks: set[int] = set()
        for b in range(self.num_blocks):
            prefix = f"blk.{b}."
            if model.has(f"{prefix}ssm_a"):
                self.ssm_blocks.add(b)
            if model.has(f"{prefix}attn_indexer.q_proj.weight"):
                self.qsa_blocks.add(b)
            elif model.has(f"{prefix}attn_q.weight"):
                self.full_attn_blocks.add(b)
            if model.has(f"{prefix}ffn_gate_inp.weight"):
                self.moe_blocks.add(b)

        # gamma/v3: QSA hyperparameters (read from header). All four
        # default to 0; the QSA path is a no-op when any is 0. The
        # constants pinned here are from the Qwen3.8-Flash-Next model
        # card (analysis doc §1.1 + §3.3) -- not measured on
        # ATF. See docs/gamma_v3_progress.md for the "assumed, not
        # measured" note.
        self.qsa_indexer_heads = int(getattr(h, "qsa_indexer_heads", 0))
        self.qsa_kv_heads = int(getattr(h, "qsa_kv_heads", 0))
        self.qsa_budget_blocks = int(getattr(h, "qsa_budget_blocks", 0))
        self.qsa_block_size = int(getattr(h, "qsa_block_size", 0))
        if self.qsa_blocks:
            console.print(f"    [cyan]QSA blocks: {len(self.qsa_blocks)} "
                          f"indexer_heads={self.qsa_indexer_heads} "
                          f"kv_heads={self.qsa_kv_heads} "
                          f"budget_blocks={self.qsa_budget_blocks} "
                          f"block_size={self.qsa_block_size}[/cyan]")

        # gamma/v1: MTP (Multi-Token Prediction) drafter head autodetect.
        # Qwen3.5 / 3.8 / 4 GGUF and HF safetensors ship one or more MTP
        # blocks (typically 1) whose tensors are renamed to the mtp.*
        # namespace by atf/convert.py and atf/convert_mlx.py. The MTP
        # drafter is OPTIONAL -- the engine must work fine without it
        # (vanilla 9B/27B checkpoints that don't ship MTP). We detect it
        # by the presence of mtp.0.eh_proj.weight (the signature tensor
        # -- enorm/hnorm alone don't guarantee a drafter is wired).
        # See docs/qwen38_flash_next_analysis.md §3.2 and
        # SPEC_DECODING_PROPOSAL.md §0.
        self.mtp_slots: list[int] = []   # 0-based slot indices present
        self.mtp_has_shared_head: bool = False
        if model.has("mtp.0.eh_proj.weight"):
            # Walk mtp.<K>.eh_proj.weight to find contiguous slot range.
            k = 0
            while model.has(f"mtp.{k}.eh_proj.weight"):
                self.mtp_slots.append(k)
                k += 1
            self.mtp_has_shared_head = model.has("mtp.shared_head_head.weight")
            console.print(f"    [cyan]MTP drafter: {len(self.mtp_slots)} slot(s) "
                          f"detected (shared_head={'yes' if self.mtp_has_shared_head else 'no, using output.weight'})[/cyan]")
        # MTP speculative decoding default = OFF in gamma/v1. v1.1 will
        # add the rejection-sampling loop in step_with_speculation(); v1
        # only verifies the drafter weights load and that the engine does
        # not regress on MTP-less checkpoints. ATF_MTP_DRAFT=<k> env var
        # will enable the speculative loop in v1.1; for now it's parsed
        # and logged but unused.
        _mtp_draft_env = os.environ.get("ATF_MTP_DRAFT", "").strip()
        self.mtp_draft_k: int = 0
        if _mtp_draft_env:
            try:
                self.mtp_draft_k = max(0, min(int(_mtp_draft_env), 8))
            except ValueError:
                self.mtp_draft_k = 0
        if self.mtp_draft_k and not self.mtp_slots:
            console.print(f"    [yellow]ATF_MTP_DRAFT={self.mtp_draft_k} requested "
                          f"but no MTP drafter tensors found in checkpoint; "
                          f"ignoring.[/yellow]")
            self.mtp_draft_k = 0
        if self.mtp_draft_k:
            console.print(f"    [cyan]MTP speculative loop: k={self.mtp_draft_k} "
                          f"(v1.1+ -- currently a no-op, weights verified only)[/cyan]")

        # gamma/v3.1 (SPEC_DECODING_PROPOSAL.md Phase 2): shallow self-
        # speculative decoding via early-exit through the first N blocks
        # (Medusa/EAGLE-style option 3 -- see the proposal's §2a: a
        # two-engine 9B-drafts-27B design (option 1) doesn't fit the 16GB
        # host, so drafting reuses THIS model's own first-quarter blocks
        # instead of a second model). Off unless GenConfig.spec_draft==
        # "shallow" is set at generate() time; this just picks the default
        # shallow-layer count once the block count is known.
        self._spec_shallow_layers = max(1, self.num_blocks // 4)

        # Qwen3.8 uses 256-dimensional attention heads even though
        # hidden_dim / head_count is not integral (5120 / 24 = 213.33).
        # Derive the full-attention width from Q, whose output is
        # [query, gate] for every head.
        if self.full_attn_blocks:
            b = min(self.full_attn_blocks)
            key = f"blk.{b}.attn_q.weight"
            q_shape = (model.shapes.get(key)
                       or model._raw_shapes.get(key)
                       or model.w(key).shape)
            q_width = int(q_shape[-1])
            if q_width % (2 * self.num_heads) == 0:
                self.head_dim = q_width // (2 * self.num_heads)

        if self.ssm_blocks:
            p = f"blk.{next(iter(self.ssm_blocks))}."
            self.gdn_n_v_heads = model.w(f"{p}ssm_a").shape[0]
            self.gdn_head_v_dim = model.w(f"{p}ssm_norm.weight").shape[0]
            qkv_shape = model.shapes.get(
                f"{p}attn_qkv.weight",
                tuple(model.w(f"{p}attn_qkv.weight").shape))
            value_dim = self.gdn_n_v_heads * self.gdn_head_v_dim
            key_total = qkv_shape[1] - value_dim
            self.gdn_head_k_dim = self.gdn_head_v_dim
            self.gdn_n_k_heads = key_total // (2 * self.gdn_head_k_dim)
        else:
            self.gdn_n_v_heads = self.gdn_n_k_heads = 0
            self.gdn_head_k_dim = self.gdn_head_v_dim = 0

        self.rope_dim = getattr(h, "rope_dim", 0) or self.head_dim
        if self.rope_dim > self.head_dim:
            self.rope_dim = self.head_dim
        t = np.arange(0, self.rope_dim, 2, dtype=np.float32)
        self.rope_freqs = mx.array(1.0 / (1e7 ** (t / self.rope_dim)))

        # gamma/v2: n-gram embedding detection. The HEADER carries the
        # metadata; the table payload lives in the dense section (proj +
        # norm) and the expert section (main table). When ngram_vocab == 0
        # the ngram path is a no-op and ATF behaves exactly as before.
        # See docs/qwen38_flash_next_analysis.md §6.
        # Use getattr() with a default of 0 so test-side SimpleNamespace
        # fakes (which don't set ngram_vocab) still work -- this keeps
        # v1's test_inference.py contract intact.
        self.ngram = None
        _ngram_vocab = getattr(h, "ngram_vocab", 0)
        _ngram_dim = getattr(h, "ngram_dim", 0)
        if _ngram_vocab > 0 and _ngram_dim > 0:
            self.ngram = {
                "vocab": int(h.ngram_vocab),
                "dim": int(h.ngram_dim),
                "insert_layer": int(h.ngram_insert_layer),
                "context": int(h.ngram_context),
            }
            # The proj and norm are tiny dense tensors (ngram_dim wide);
            # the big table stays in LOD on the expert section and is
            # paged in on demand by model._raw_base. We resolve the
            # dense pieces here, lazily, so a non-ngram checkpoint
            # never touches the path.
            self.ngram["has_proj"] = model.has("ngram_emb_proj.weight")
            self.ngram["has_norm"] = model.has("ngram_emb_norm.weight")
            self.ngram["has_table"] = model.has("ngram_emb.weight")
            console.print(f"    [cyan]N-gram embedding: vocab={self.ngram['vocab']} "
                          f"dim={self.ngram['dim']} insert_layer={self.ngram['insert_layer']} "
                          f"context={self.ngram['context']} "
                          f"(table={self.ngram['has_table']}, "
                          f"proj={self.ngram['has_proj']}, "
                          f"norm={self.ngram['has_norm']})[/cyan]")

        # v9.1: per-token KV cost across all attention blocks (fp16 k+v),
        # used to cap effective context to MAX_KV_GPU_BYTES.
        self.kv_bytes_per_token = (len(self.full_attn_blocks)
                                   * self.num_kv * self.head_dim * 2 * 2)

        # v9.2 GPU-hang fix: wire the packed weights (paging them out is what
        # hung the GPU) and derive the KV budget from real device limits.
        packed = 0
        if getattr(self.model, "_raw_len", None):
            packed = sum(self.model._raw_len.values())
        self.wired_bytes = apply_wired_limit(packed)
        if self.wired_bytes:
            console.print(f"  Wired {self.wired_bytes / (1 << 30):.2f} GiB of "
                          "packed weights (GPU-paging-proof)")
        self.kv_budget_bytes = kv_budget_bytes(packed)
        # Always-on memory budget summary. Single block of numbers so a
        # user pasting a v6/v7 log can read off whether the model even
        # fits the box. Printed both via rich.Console (visible in normal
        # output) and as an [atf-verbose] line (visible in stderr trace).
        try:
            _di = mx.device_info()
            _ws = int(_di.get("max_recommended_working_set_size", 0))
            _mt = int(_di.get("memory_size", 0))
            _mxbuf = int(_di.get("max_buffer_length", 0))
        except Exception:
            _ws = _mt = _mxbuf = 0
        # 8-byte KV per token x max_context across all attention blocks.
        # x2 below = full-attn block has both k and v buffers (1.5x for
        # the GDN conv state is an additional fixed cost we ignore here
        # because the first GDN chunk is small).
        _max_ctx = int(getattr(GenConfig, "max_context", 65536))
        _peak_kv_if_maxctx = self.kv_bytes_per_token * _max_ctx
        _peak_total_if_maxctx = packed + _peak_kv_if_maxctx + _ACT_HEADROOM_BYTES
        console.print(f"  [cyan]Memory budget:[/cyan]")
        console.print(f"    device memory_size            = "
                      f"{_mt/(1<<30):.2f} GiB (total RAM)")
        console.print(f"    max_recommended_working_set   = "
                      f"{_ws/(1<<30):.2f} GiB (macOS paging cap)")
        console.print(f"    max_buffer_length            = "
                      f"{_mxbuf/(1<<30):.2f} GiB (single-allocation cap)")
        console.print(f"    packed weights on disk       = "
                      f"{packed/(1<<30):.2f} GiB")
        console.print(f"    wired_limit applied          = "
                      f"{(self.wired_bytes or 0)/(1<<30):.2f} GiB")
        console.print(f"    KV budget                    = "
                      f"{self.kv_budget_bytes/(1<<30):.2f} GiB")
        console.print(f"    KV @ max_context({_max_ctx})  = "
                      f"{_peak_kv_if_maxctx/(1<<30):.2f} GiB")
        console.print(f"    peak @ max_context           = "
                      f"{_peak_total_if_maxctx/(1<<30):.2f} GiB")
        if _ws and _peak_total_if_maxctx > _ws:
            console.print(
                f"  [yellow]WARNING: peak ({_peak_total_if_maxctx/(1<<30):.2f} GiB) "
                f"exceeds working set ({_ws/(1<<30):.2f} GiB) -- long-context "
                f"requests may OOM Metal. Reduce max_context, close other apps, "
                f"or pick a smaller model.[/yellow]")
        # Stderr mirror so the [atf-verbose] stream captures the same numbers.
        _vlog(1, f"engine.budget device_mem={_mt} ws={_ws} max_buf={_mxbuf} "
                f"packed={packed} wired={self.wired_bytes or 0} "
                f"kv_budget={self.kv_budget_bytes} "
                f"kv_per_tok={self.kv_bytes_per_token} "
                f"peak_maxctx={_peak_total_if_maxctx} "
                f"max_context={_max_ctx}")
        # MoE-specific dequant pressure: with T_chunk tokens routed top_k,
        # worst-case is top_k unique experts per MoE block. Each expert on
        # the raw-GGUF path is dequantized to fp32 on first miss and held
        # in the LRU cache. Print the per-block expert-dequant peak so the
        # "where did 8 GB go" question is answerable.
        try:
            _moe_n = len(getattr(self, "moe_blocks", set()) or [])
            if _moe_n and getattr(self, "top_k", 0) > 0:
                # crude estimate: assume expert intermediate = 4 * hidden
                _expert_packed_bytes = (self.hidden * 4 * self.hidden * 1) // 2
                _expert_fp32_bytes = (self.hidden * 4 * self.hidden) * 4
                _moe_peak = (_moe_n * self.top_k
                             * (3 * _expert_fp32_bytes))   # gate+up+down
                console.print(f"    MoE blocks={_moe_n} top_k={self.top_k} "
                              f"peak expert dequant (all blocks, worst case)= "
                              f"{_moe_peak/(1<<30):.2f} GiB")
                _vlog(1, f"engine.budget moe_blocks={_moe_n} top_k={self.top_k} "
                        f"expert_packed={_expert_packed_bytes} "
                        f"expert_fp32={_expert_fp32_bytes} "
                        f"moe_peak_worst={_moe_peak}")
        except Exception:
            pass
        # NOTE: do NOT set mx.set_memory_limit here. With ~9 GB of weights
        # resident, the limit-check path in MLX's allocator thrashes and
        # stalls every allocation (observed: 100% CPU inside eval_impl /
        # allocator on a tiny prefill). The KV budget cap above is the real
        # protection. Opt-in only via ATF_MEM_LIMIT (GB) for debugging.
        if os.environ.get("ATF_MEM_LIMIT"):
            try:
                mx.set_memory_limit(int(float(os.environ["ATF_MEM_LIMIT"]) * (1 << 30)))
            except Exception:
                pass
        # Stash for later use in generate()-level preflight.
        self._device_ws = _ws
        self._device_max_buf = _mxbuf
        self._packed_bytes = packed

        # v19 Win A (PERFORMANCE_LOD_PLAN.md "Win A"). The mx.compile
        # call fuses ~6 separate elementwise/reduce Metal launches per
        # GDN block per token into one command buffer. gamma/v4 bench
        # 2026-09-02: on the 27B Q2_K_XL this is +4% tok/s (4.63 ->
        # 4.80 on a "hi" prompt) at byte-identical output. Set
        # ATF_COMPILE_STEP=0 to opt out (revert to the eager per-launch
        # Metal path) if a regression is found in a future model.
        self._compile_gdn_step = os.environ.get("ATF_COMPILE_STEP", "1") != "0"
        self._gdn_step_fn = None
        if self._compile_gdn_step:
            console.print("  [cyan]ATF_COMPILE_STEP=1 (default): compiling GDN decode "
                          "step (Win A; +4% on 27B Q2_K_XL, byte-identical output)[/cyan]")

        # Win B: Pin per-block GDN small weights to eliminate LRU overhead & string lookups
        self._gdn_pinned: dict[int, tuple] = {}
        if self.ssm_blocks:
            for b in self.ssm_blocks:
                p = f"blk.{b}."
                try:
                    self._gdn_pinned[b] = (
                        model.w(f"{p}ssm_conv1d.weight"),
                        model.w(f"{p}ssm_alpha.weight"),
                        model.w(f"{p}ssm_beta.weight"),
                        model.w(f"{p}ssm_dt.bias"),
                        model.w(f"{p}ssm_a"),
                        model.w(f"{p}ssm_norm.weight"),
                    )
                except Exception as _e:
                    _vlog(2, f"Failed to pin weights for GDN block {b}: {_e}")

        # Win A Expanded: Full end-to-end GDN step compilation for T=1
        self._compile_full_gdn = os.environ.get("ATF_COMPILE_FULL_GDN", "1") != "0"
        self._gdn_full_step_fn = None
        if self._compile_full_gdn and self._gdn_pinned:
            console.print(f"  [cyan]ATF_COMPILE_FULL_GDN=1: full single-token GDN step "
                          f"fused across {len(self._gdn_pinned)} blocks (Win A+B)[/cyan]")

        console.print(f"  Engine ready: {self.num_blocks} blocks, "
                      f"{len(self.ssm_blocks)} GatedDeltaNet, "
                      f"{len(self.full_attn_blocks)} full-attn, "
                      f"{len(self.moe_blocks)} routed-MoE")

    def max_context_default(self) -> int:
        """Default KV capacity (GenConfig.max_context default)."""
        return GenConfig.max_context

    # ─── public API ──────────────────────────────────────────────────────

    def generate(self, prompt: str, config: GenConfig | None = None,
                 stream: bool = False, token_cb=None,
                 progress_cb=None) -> str:
        if config is None:
            config = GenConfig()
        if self.tok is None:
            raise ValueError("No tokenizer loaded")
        if config.seed is not None:
            mx.random.seed(config.seed)

        _vstate_reset()
        _vstate_set("generate_t0", time.time())
        _vlog(1, f"generate.start prompt_chars={len(prompt)} "
                f"max_tokens={config.max_tokens} max_context={config.max_context} "
                f"thinking={config.thinking} exact_ffn={config.exact_ffn} "
                f"prefill_chunk={config.prefill_chunk} "
                f"moe_blocks={len(self.moe_blocks) if hasattr(self, 'moe_blocks') else 0} "
                f"full_attn_blocks={len(self.full_attn_blocks)} "
                f"top_k={self.top_k} mem={gpu_mem_line()}")

        ids = self.tok.encode(prompt)
        _vlog(1, f"generate.encoded prompt_tokens={len(ids)} mem={gpu_mem_line()}")

        budget = THINK_BUDGETS.get(config.thinking, THINK_BUDGETS["medium"])
        think_closed = True
        if prompt.endswith("assistant\n"):
            if budget == 0:
                ids = ids + self.tok.encode("<think>\n\n</think>\n\n")
            else:
                ids = ids + self.tok.encode("<think>\n")
                think_closed = False

        # gamma/v7 persistence (ATF_KV_SSD_PERSIST): look for a resumable
        # on-disk session BEFORE creating kv_caches, since enable_ssd() needs
        # resume_n/persist at creation time, not after. Only attempted when
        # there is NO in-memory _pcache (i.e. this is a fresh process -- if
        # _pcache is already populated the existing, faster, in-memory reuse
        # path below takes precedence and this is skipped entirely).
        _kv_ssd_disk_reuse = None
        _kv_ssd_cache_dir = None
        if config.kv_ssd_enabled and config.kv_ssd_persist:
            _kv_ssd_cache_dir = Path(config.kv_ssd_path) if config.kv_ssd_path \
                else (Path.home() / ".cache" / "atf" / "kvpages")
            if getattr(self, "_pcache", None) is None and config.model_id:
                from .kvstore import find_reusable, session_key
                try:
                    _kv_ssd_disk_reuse = find_reusable(
                        _kv_ssd_cache_dir, config.model_id, ids)
                except Exception as _exc:
                    _vlog(1, f"generate.kv_ssd_persist.find_reusable failed "
                            f"(falling back to full prefill): {_exc}")
                    _kv_ssd_disk_reuse = None
                if _kv_ssd_disk_reuse is not None:
                    _vlog(1, f"generate.kv_ssd_persist DISK HIT key="
                            f"{_kv_ssd_disk_reuse['key']} "
                            f"lcp={_kv_ssd_disk_reuse['lcp']}/{len(ids)}")
                else:
                    _vlog(1, "generate.kv_ssd_persist no reusable disk "
                            "session found; full prefill (will still "
                            "persist for next time)")

        kv_caches = {b: KVCache() for b in self.full_attn_blocks}
        # gamma/v7: Paged SSD KV cache — enable before prefix cache reuse
        # so that new caches are already in SSD mode. Prefix-cache reuse
        # (below) will overwrite individual entries with the old caches that
        # are already populated (SSD or flat).
        if config.kv_ssd_enabled:
            if _kv_ssd_disk_reuse is not None:
                _ssd_sid = _kv_ssd_disk_reuse["key"]
                _ssd_resume_n = _kv_ssd_disk_reuse["lcp"]
            elif config.kv_ssd_persist and config.model_id:
                from .kvstore import session_key
                _ssd_sid = session_key(config.model_id, ids)
                _ssd_resume_n = 0
            else:
                _ssd_sid = f"sess_{int(time.time()*1000)}_{os.getpid()}"
                _ssd_resume_n = 0
            for _b, _c in kv_caches.items():
                _c.enable_ssd(
                    capacity=config.max_context,
                    n_kv=self.num_kv, head_dim=self.head_dim,
                    hot_pages=config.kv_ssd_hot_pages,
                    mmap_path=(Path(config.kv_ssd_path) / f"{_ssd_sid}_b{_b}.kvmm"
                               if config.kv_ssd_path else
                               ((Path.home() / ".cache" / "atf" / "kvpages" / f"{_ssd_sid}_b{_b}.kvmm")
                                if config.kv_ssd_persist else None)),
                    session_id=f"{_ssd_sid}_b{_b}",
                    resume_n=_ssd_resume_n,
                    persist=config.kv_ssd_persist,
                )
            console.print(f"  [cyan][step] SSD KV cache enabled: "
                          f"{len(kv_caches)} blocks, "
                          f"hot_pages={config.kv_ssd_hot_pages}, "
                          f"capacity={config.max_context}, "
                          f"persist={config.kv_ssd_persist}, "
                          f"resumed={_ssd_resume_n} tokens[/cyan]")
        gdn_states: dict = {}

        # ── prompt prefix cache (v7) ─────────────────────────────────
        # Chat clients resend the entire conversation every turn. If the
        # new prompt shares a prefix with the previous request, reuse the
        # KV/GDN state computed then and prefill only the suffix.
        start = 0
        pcache = getattr(self, "_pcache", None)
        if pcache is not None:
            old_ids, old_kv, old_gdn = pcache
            lcp = 0
            for a, b_ in zip(old_ids, ids):
                if a != b_:
                    break
                lcp += 1
            # lcp < len(ids): LCP must leave at least one token to prefill
            # so the LM head still runs.  Otherwise we hit the crash on the
            # first _sample (logits stays None, _sample raises
            # "LM head returned invalid logits shape ()").  The cause was the
            # v10 sys_baseline injection: sys_ids is encoded as the system
            # block ALONE (no trailing \n) while the engine re-tokenizes the
            # full built prompt, and zip() stops at min(len(old_ids), len(ids))
            # — if the re-tokenized prompt is shorter than sys_ids *and* the
            # shared prefix matches all the way, lcp == len(ids) and the
            # chunked prefill loop never runs.
            if lcp >= 64 and lcp < len(ids):
                for b_, c in old_kv.items():
                    c.n = min(c.n, lcp)
                    kv_caches[b_] = c
                gdn_states = old_gdn
                start = lcp
                _vlog(1, f"generate.lcp HIT lcp={lcp}/{len(ids)} "
                        f"reusing {lcp} tokens of KV/GDN, "
                        f"prefill_tail={len(ids)-lcp} mem={gpu_mem_line()}")
            else:
                _vlog(1, f"generate.lcp SKIP lcp={lcp} (<64 or ==len(ids)); "
                        f"full prefill of {len(ids)} tokens")
        elif _kv_ssd_disk_reuse is not None:
            # gamma/v7 persistence: no in-memory _pcache (fresh process) but
            # a validated on-disk session matched this exact prefix. The
            # paged KV caches above already resumed at lcp tokens (resume_n);
            # restore the matching GDN recurrent state and skip re-prefilling
            # that prefix, same as the in-memory HIT path above.
            gdn_states = _kv_ssd_disk_reuse["gdn_states"]
            start = _kv_ssd_disk_reuse["lcp"]
            _vlog(1, f"generate.kv_ssd_persist.lcp HIT lcp={start}/{len(ids)} "
                    f"reusing {start} tokens of on-disk KV/GDN, "
                    f"prefill_tail={len(ids)-start} mem={gpu_mem_line()}")
        else:
            _vlog(1, f"generate.lcp NONE (no _pcache); full prefill of {len(ids)} tokens")

        if len(ids) + config.max_tokens > config.max_context:
            raise ValueError(
                f"prompt ({len(ids)}) + max_tokens ({config.max_tokens}) "
                f"exceeds max_context ({config.max_context})")

        # ── Preflight: refuse to start if peak memory looks unsafe ──
        # Better to fail with a clear "your box is too small for this
        # config" than to crash Metal with kIOGPUCommandBufferCallback
        # ErrorOutOfMemory halfway through the first prefill forward
        # (the C++ exception escapes as SIGABRT and the bridge dies).
        try:
            _cur_active = int(mx.get_active_memory())
            _cur_peak = int(mx.get_peak_memory())
        except Exception:
            _cur_active = _cur_peak = 0
        # Realistic activation peak during prefill.
        #
        # The chunked-score path in _attention_full processes blocks one at a
        # time and bounds the per-chunk score matrix to QC=256 queries (not
        # T_chunk, which is typically 1024). The K/V replication per block
        # uses the current cache length, not the worst-case T_total. So the
        # real peak per chunk is one block's worth of (score matrix + K/V
        # replication), which is the dominant transient term but well under
        # the previous "all blocks x T_chunk x T_total" estimate.
        #
        # Per-block transient peak = max(nh, num_kv) * max(QC, T_chunk)
        #                            * min(eff_cap, T_total) * hd * 4
        # which for the 27B (16 full blocks, 32 heads, 256 hd, eff_cap 23552)
        # = 32 * 1024 * 23552 * 256 * 4 = ~770 MB per block (transient, freed
        # at chunk end). The full-block attn_score of the previous code
        # multiplied this by num_full=16, giving 12 GB -- the "conservative"
        # overestimate that the previous comment acknowledged.
        T_chunk = max(1, config.prefill_chunk)
        T_total_est = len(ids) + config.max_tokens
        QC = 256 if T_chunk > 256 else T_chunk
        # Eff-cap used for the K-dim of the score matrix. This caps the
        # transient at the actual KV size the engine will allocate.
        _eff_cap_for_estimate = min(config.max_context,
                                      max(1024,
                                          self.kv_budget_bytes
                                          // max(1, self.kv_bytes_per_token)))
        _T_score = min(max(T_chunk, QC), max(1, _eff_cap_for_estimate))
        # Per-block transient: the score matrix is [max(nh, nk) x QC x eff_cap]
        # in fp32, one chunk at a time, freed at the end of each chunk. NOT
        # multiplied by num_blocks because the engine processes blocks
        # sequentially (one at a time). The Q-dim is QC (256), NOT T_chunk --
        # the chunked-score path in _attention_full loops over QC-sized query
        # windows, so the score matrix per iteration is bounded by QC.
        _attn_score_per_block = (max(self.num_heads, self.num_kv)
                                  * QC
                                  * max(1, _eff_cap_for_estimate)
                                  * 4)
        # f32 residual per block during prefill (T_chunk x hidden x 4). The
        # engine processes blocks sequentially so the actual transient is ONE
        # block's residual at a time, not all num_blocks simultaneously. The
        # previous "all blocks" multiplier was overcounting by num_blocks x.
        _resid_per_chunk = (T_chunk * self.hidden * 4)
        # KV we will actually allocate (eff_cap applied below).
        # gamma/v7: with SSD KV cache, only the hot pages are GPU-resident.
        if config.kv_ssd_enabled:
            from .kvstore import PAGE
            _hot_tokens = config.kv_ssd_hot_pages * PAGE
            _est_kv = (self.kv_bytes_per_token * min(_hot_tokens, _eff_cap_for_estimate))
        else:
            _est_kv = (self.kv_bytes_per_token * _eff_cap_for_estimate)
        _est_peak = (int(getattr(self, "_packed_bytes", 0)) + _est_kv
                     + _attn_score_per_block + _resid_per_chunk
                     + _ACT_HEADROOM_BYTES)
        _ws = int(getattr(self, "_device_ws", 0))
        _vlog(1, f"generate.preflight cur_active={_cur_active} "
                f"cur_peak={_cur_peak} packed={self._packed_bytes} "
                f"est_kv={_est_kv} est_peak={_est_peak} ws={_ws} "
                f"attn_score_per_block={_attn_score_per_block} "
                f"resid_per_chunk={_resid_per_chunk}")
        # 0.95 was over-restrictive: the estimate is conservative by
        # ~10-20% vs the real MLX lazy-materialization peak (the 27B real
        # peak is 9.75 GiB vs 11.84 GiB working set = 1.21x headroom; the
        # conservative estimate was tripping at 13 GiB = 1.10x). 1.10 lets
        # the user's case through while still catching a 10%+ overestimate.
        if _ws and _est_peak > int(_ws * 1.10):
            msg = (
                f"[ATF_OM] estimated peak {_est_peak/(1<<30):.2f} GiB "
                f"exceeds 95% of macOS working set "
                f"({_ws/(1<<30):.2f} GiB) for this model+context. "
                f"packed={self._packed_bytes/(1<<30):.2f} GiB, "
                f"est_kv={_est_kv/(1<<30):.2f} GiB "
                f"(eff_cap={min(config.max_context, max(1024, self.kv_budget_bytes // max(1, self.kv_bytes_per_token)))} "
                f"tokens), max_tokens={config.max_tokens}, "
                f"prompt={len(ids)}. Reduce max_tokens or max_context, "
                f"close other apps, or pick a smaller model. "
                f"Set ATF_OM_SKIP=1 to bypass this check (likely crash).")
            console.print(f"  [red]{msg}[/red]")
            _vlog(1, msg)
            if os.environ.get("ATF_OM_SKIP") != "1":
                raise RuntimeError(msg)

        console.print(f"  [cyan][step] prompt encoded: {len(ids)} tokens, "
                      f"max_tokens={config.max_tokens}, max_context={config.max_context}, "
                      f"exact_ffn={config.exact_ffn}, mem={gpu_mem_line()}[/cyan]")

        # v9.1: cap effective context so worst-case KV stays under the GPU
        # budget (MAX_KV_GPU_BYTES). Prevents any config from wiring more KV
        # than the machine can survive.
        # gamma/v7: when SSD KV cache is active, the GPU budget is no longer
        # the limit — cold pages live on SSD. Use the full max_context.
        if config.kv_ssd_enabled:
            eff_cap = config.max_context
            console.print(f"  [cyan][step] SSD KV: GPU budget bypassed, "
                          f"eff_cap={eff_cap} (full max_context)[/cyan]")
        else:
            eff_cap = min(config.max_context,
                          max(1024, self.kv_budget_bytes // max(1, self.kv_bytes_per_token)))
            if eff_cap < config.max_context:
                console.print(f"[yellow]  Context capped at {eff_cap} tokens "
                              f"(KV budget {self.kv_budget_bytes/(1<<30):.2f} GiB; "
                              f"full {config.max_context} would need "
                              f"{self.kv_bytes_per_token*config.max_context/(1<<30):.1f} GiB)[/yellow]")
        for c in kv_caches.values():
            if not c.is_ssd:
                c.capacity = eff_cap
        console.print(f"  [cyan][step] KV caches initialized, "
                      f"full_attn_blocks={len(self.full_attn_blocks)}, "
                      f"eff_cap={eff_cap}, ssd={config.kv_ssd_enabled}, mem={gpu_mem_line()}[/cyan]")

        t0 = time.time()
        n_prefill = len(ids) - start
        if progress_cb is not None and n_prefill > 0:
            progress_cb(0, n_prefill)
        console.print(f"  [cyan][step] starting prefill: {n_prefill} tokens, "
                      f"chunk={config.prefill_chunk}, mem={gpu_mem_line()}[/cyan]")
        _vlog(1, f"generate.prefill.start n_tokens={n_prefill} "
                f"chunk={config.prefill_chunk} "
                f"moe_blocks_in_prompt={len(self.moe_blocks) if hasattr(self, 'moe_blocks') else 0} "
                f"mem={gpu_mem_line()}")
        logits = self._forward_prefill(ids[start:], kv_caches, gdn_states,
                                       config.exact_ffn,
                                       chunk=config.prefill_chunk,
                                       progress_cb=progress_cb)
        _vlog(2, f"generate.prefill.forward_done mem={gpu_mem_line()}")
        console.print(f"  [cyan][step] prefill forward returned, "
                      f"evaluating logits..., mem={gpu_mem_line()}[/cyan]")
        mx.eval(logits)
        console.print(f"  [cyan][step] logits evaluated OK, mem={gpu_mem_line()}[/cyan]")
        # Level 4: NaN/Inf check on the post-prefill logits. Catches expert
        # dequant blow-ups that only manifest on long prefill of a MoE.
        if _VERBOSE >= 4 or _NAN_TRAP or _NAN_WARN:
            _nan_check(logits, where="post_prefill_logits")
        prefill_t = time.time() - t0
        cached = f" (+{start} cached)" if start else ""
        toks_per_s = (len(ids)-start)/prefill_t if prefill_t > 0 else 0.0
        _vlog(1, f"generate.prefill.done tokens={len(ids)-start} "
                f"t={prefill_t:.2f}s tok/s={toks_per_s:.1f}{cached} "
                f"mem={gpu_mem_line()}")
        _vstate_add("prefill_t", prefill_t)
        _vstate_add("prefill_tokens", len(ids) - start)
        console.print(f"  Prefill: {len(ids)-start} tokens in {prefill_t:.1f}s "
                      f"({(len(ids)-start)/prefill_t:.1f} tok/s){cached}")

        out_ids: list[int] = []
        history = list(ids)
        fed_ids = list(ids)          # every token actually pushed through
        i = 0
        gen_t0 = time.time()
        _vlog(1, f"generate.decode.start max_tokens={config.max_tokens} "
                f"mem={gpu_mem_line()}")
        _decode_step_ms_max = 0.0
        # spec_min_acceptance auto-disable: track the recent acceptance
        # rate across spec cycles. If the model's shallow-layer draft
        # starts producing mostly-rejected tokens (low accept rate
        # means the verify-pass is doing most of the work and the
        # spec overhead is net-negative), fall back to plain decode
        # for the rest of this generation. Configurable via
        # GenConfig.spec_min_acceptance (default 0.30 -- 30% of
        # drafted tokens accepted = the verify-pass pays for the
        # draft's overhead; below that, the draft is hurting more
        # than it helps).
        _spec_ema_accept = 1.0     # optimistic prior so the first
                                    # cycle never triggers auto-disable
        _spec_ema_alpha = 0.2      # EMA smoothing factor
        _spec_disabled = False
        while i < config.max_tokens:
            if (config.spec_draft == "shallow" and not _spec_disabled
                    and think_closed
                    and self._spec_shallow_layers > 0):
                _ptid = fed_ids[-1] if fed_ids else 0
                # Carry the current EMA through to _spec_step so it
                # can update it after observing this cycle's accept
                # rate. Pass it via a tiny mutable container since
                # _spec_step is method-shaped and we want to avoid
                # touching its signature.
                _committed, spec_logits, _accept_rate = self._spec_step(
                    _ptid, kv_caches, gdn_states, config, history,
                    _spec_ema_accept)
                # Update the EMA: new = alpha * observed + (1-alpha) * old
                _spec_ema_accept = (_spec_ema_alpha * _accept_rate
                                    + (1.0 - _spec_ema_alpha) * _spec_ema_accept)
                # Auto-disable: if recent acceptance has dropped below
                # the configured minimum, the draft is hurting more
                # than it helps. Revert to plain decode for the rest
                # of this generation. One-shot decision -- we don't
                # try to re-enable mid-generation (the draft quality
                # is a property of the model+state, not transient).
                if _spec_ema_accept < config.spec_min_acceptance:
                    _spec_disabled = True
                    _vlog(1, f"spec.auto_disable ema_accept="
                            f"{_spec_ema_accept:.3f} < "
                            f"spec_min_acceptance="
                            f"{config.spec_min_acceptance:.3f}; "
                            f"reverting to plain decode for remainder "
                            f"of generation")
                _stop = False
                for _tid in _committed:
                    i += 1
                    if _tid in (getattr(self.tok, "eos_token_id", None), 248044):
                        _stop = True
                        break
                    out_ids.append(_tid)
                    if token_cb is not None:
                        token_cb(_tid)
                    history.append(_tid)
                    if stream and self.tok is not None:
                        console.print(self.tok.decode_token(_tid), end="")
                    fed_ids.append(_tid)
                    _vstate_add("decode_steps", 1)
                    if out_ids and _VERBOSE_STATE["first_token_t"] is None:
                        _VERBOSE_STATE["first_token_t"] = time.time() - (
                            _VERBOSE_STATE.get("generate_t0") or gen_t0)
                    if i >= config.max_tokens:
                        break
                logits = spec_logits
                if _stop:
                    break
                continue
            i += 1
            step_t0 = time.time()
            if not think_closed and budget is not None and i > budget:
                for tid in self.tok.encode("</think>\n\n"):
                    if token_cb is not None:
                        token_cb(tid)
                    # v8: no mx.eval -- _sample below syncs implicitly
                    # gamma/v2: pass the predecessor token id for n-gram
                    # insertion (the last token that was actually pushed
                    # through, or the last prefill token for the first step).
                    _ptid = fed_ids[-1] if fed_ids else 0
                    logits = self._forward_token(tid, kv_caches, gdn_states,
                                                config.exact_ffn, prev_token_id=_ptid)
                    fed_ids.append(tid)
                think_closed = True
            nxt = self._sample(logits, config, history)
            if nxt == THINK_CLOSE:
                think_closed = True
            if not think_closed and nxt == THINK_OPEN:
                if token_cb is not None:
                    token_cb(nxt)
                continue
            if nxt in (getattr(self.tok, "eos_token_id", None), 248044):
                break
            out_ids.append(nxt)
            if token_cb is not None:
                token_cb(nxt)
            history.append(nxt)
            if stream and self.tok is not None:
                console.print(self.tok.decode_token(nxt), end="")
            # gamma/v2: pass the predecessor for n-gram insertion.
            # fed_ids[-1] is the token that produced the nxt sample --
            # i.e. the last token whose residual was actually pushed
            # through the layers.
            _ptid = fed_ids[-1] if fed_ids else 0
            logits = self._forward_token(nxt, kv_caches, gdn_states,
                                        config.exact_ffn, prev_token_id=_ptid)
            fed_ids.append(nxt)
            # Per-step decode timing. At level 3 we log every token; at level
            # 4 we also dump a NaN-check of the post-token logits and the
            # top-1/top-2 token ids.
            step_dt_ms = (time.time() - step_t0) * 1000.0
            if step_dt_ms > _decode_step_ms_max:
                _decode_step_ms_max = step_dt_ms
            if _VERBOSE >= 3:
                _vstate_set("decode_step_ms_max", _decode_step_ms_max)
                # Cheap stats without an NaN check unless level 4.
                if _VERBOSE >= 4 or _NAN_TRAP or _NAN_WARN:
                    st = _nan_check(logits, where=f"decode.logits step={i}", raise_on_nan=False)
                    tag = f" nan={st['nan']} inf={st['inf']} min={st['min']} max={st['max']}"
                else:
                    tag = ""
                _vlog(3, f"decode.step i={i} tok={nxt} "
                        f"t={step_dt_ms:.1f}ms "
                        f"out_count={len(out_ids)}{tag}")
            _vstate_add("decode_steps", 1)
            _vstate_add("decode_t", step_dt_ms / 1000.0)
            if out_ids and _VERBOSE_STATE["first_token_t"] is None:
                _VERBOSE_STATE["first_token_t"] = time.time() - (
                    _VERBOSE_STATE.get("generate_t0") or gen_t0)
                _vlog(1, f"generate.first_token t={_VERBOSE_STATE['first_token_t']:.3f}s "
                        f"after start; tok_id={nxt}")
        if not think_closed:
            # Hit max_tokens mid-reasoning: emit the close tag so the UI
            # (and any following turn) doesn't see a dangling <think>.
            for tid in self.tok.encode("</think>"):
                if token_cb is not None:
                    token_cb(tid)
                out_ids.append(tid)
        if stream:
            console.print()
        # save state for the next call's prefix cache
        self._pcache = (fed_ids, kv_caches, gdn_states)
        # gamma/v7 persistence (ATF_KV_SSD_PERSIST): mirror that state to
        # disk too, so a FRESH process (no in-memory _pcache) can resume
        # this conversation later via the disk-reuse lookup above. The
        # paged .kvmm files are already on disk (persist=True skipped the
        # delete-on-close); this just adds the fed_ids + GDN-state sidecar
        # that makes reusing them safe. Best-effort -- save_manifest() never
        # raises, so a persistence failure can't break the reply that was
        # just generated.
        if config.kv_ssd_enabled and config.kv_ssd_persist and config.model_id:
            try:
                from .kvstore import save_manifest, session_key
                _cache_dir = _kv_ssd_cache_dir or (
                    Path(config.kv_ssd_path) if config.kv_ssd_path
                    else (Path.home() / ".cache" / "atf" / "kvpages"))
                _save_key = (_kv_ssd_disk_reuse["key"] if _kv_ssd_disk_reuse
                            else session_key(config.model_id, ids))
                _ok = save_manifest(_cache_dir, _save_key, config.model_id,
                                    fed_ids, gdn_states)
                _vlog(1, f"generate.kv_ssd_persist.save key={_save_key} "
                        f"n_fed={len(fed_ids)} ok={_ok}")
            except Exception as _exc:
                _vlog(1, f"generate.kv_ssd_persist.save failed: {_exc}")
        gen_t = time.time() - gen_t0
        n_gen = len(out_ids) + 1  # the sampled token that ended the loop
        if gen_t > 0:
            console.print(f"[dim]  Generated {len(out_ids)} tokens in {gen_t:.1f}s "
                          f"= {len(out_ids)/gen_t:.2f} tok/s[/dim]")
        # Level 1: always-on end-of-generate summary. Tells you at a glance
        # whether prefill or decode is the bottleneck, whether MoE was hit,
        # and whether NaN/Inf ever appeared in logits.
        summary = _vstate_summary()
        _vlog(1, f"generate.done gen_tokens={len(out_ids)} "
                f"gen_t={gen_t:.2f}s tok/s={(len(out_ids)/gen_t) if gen_t>0 else 0:.2f} "
                f"prefill_tokens={summary['prefill_tokens']} "
                f"prefill_t={summary['prefill_t']:.2f}s "
                f"decode_steps={summary['decode_steps']} "
                f"moe_blocks_seen={summary['moe_blocks_seen']} "
                f"moe_unique_sum={summary['moe_unique_experts_sum']} "
                f"moe_t={summary['moe_t']:.2f}s "
                f"sample_logits_nan={summary['sample_logits_nan']} "
                f"first_token_t={summary['first_token_t']} "
                f"mem={gpu_mem_line()}")
        return self.tok.decode(out_ids)

    # ─── forward passes ──────────────────────────────────────────────────

    def _forward_prefill(self, token_ids: list[int], kv_caches: dict,
                         gdn_states: dict, exact_ffn: bool = True,
                         chunk: int = 1024, progress_cb=None) -> mx.array:
        """Chunked prefill (v7).

        Processing the whole prompt at once materializes an attention score
        matrix of [nh, T, T] (~274 GB at T=65536). Feeding the prompt in
        chunks keeps peak activation at O(chunk * context) while the
        preallocated KV buffers carry state across chunks; GDN state and
        conv buffers live in gdn_states and carry over unchanged.
        """
        logits = None
        n_chunks = (len(token_ids) + max(1, chunk) - 1) // max(1, chunk)
        _vlog(2, f"_forward_prefill.start total_tokens={len(token_ids)} "
                f"chunk={chunk} n_chunks={n_chunks} mem={gpu_mem_line()}")
        _prefill_t0 = time.time()
        try:
            for s in range(0, len(token_ids), max(1, chunk)):
                piece = token_ids[s:s + max(1, chunk)]
                chunk_idx = s // max(1, chunk)
                _chunk_t0 = time.time()
                x = self.model.embed_batch(piece)
                # gamma/v2: n-gram insertion (per
                # docs/qwen38_flash_next_analysis.md §6.5). For each
                # chunk we need the predecessor token id for every token
                # in the chunk. The first token's predecessor is the
                # last token of the PREVIOUS chunk, or 0 if this is the
                # first chunk. Same logic as the position-id alignment
                # already happening in the caller.
                if self.ngram and self.ngram["insert_layer"] == 0:
                    prev = [token_ids[s - 1] if s > 0 else 0] + piece[:-1]
                    x = self._ngram_insert(x, prev, piece)
                for b in range(self.num_blocks):
                    # gamma/v2: n-gram insertion at the start of the
                    # configured layer (default 2, matching Qwen4).
                    # All tokens in the chunk share the same insertion
                    # point, so the gather is one-shot per chunk.
                    if (self.ngram and self.ngram["insert_layer"] == b):
                        prev = [token_ids[s - 1] if s > 0 else 0] + piece[:-1]
                        x = self._ngram_insert(x, prev, piece)
                    _bt0 = time.time()
                    try:
                        x = self._forward_block(b, x, kv_caches.get(b),
                                                gdn_states, exact_ffn)
                    except Exception as _blk_exc:
                        # Metal OOM is a std::runtime_error that aborts the
                        # whole process; catch it here so we get a clean
                        # [atf-verbose] line identifying the failing block
                        # before re-raising.
                        _vlog(1, f"_forward_prefill.OOM chunk={chunk_idx} "
                                f"b={b} cur_active="
                                f"{int(mx.get_active_memory()) if hasattr(mx, 'get_active_memory') else '?'} "
                                f"mem={gpu_mem_line()} "
                                f"err={type(_blk_exc).__name__}: {_blk_exc}")
                        raise
                    _bdt = (time.time() - _bt0) * 1000.0
                    _vstate_add("prefill_block_ms_total", _bdt)
                    _vstate_add("prefill_block_calls", 1)
                    if _VERBOSE >= 3:
                        kind = "attn" if b in self.full_attn_blocks else "gdn"
                        _vlog(3, f"_forward_prefill.block b={b:3d} ({kind}) "
                                f"t={_bdt:.1f}ms")
                    # gamma/v5 hotfix: this loop previously had NO
                    # console.print() checkpoint between "starting prefill"
                    # and the whole chunk finishing -- for a single-chunk
                    # prompt (<=1024 tokens, the common case) the UI's
                    # status panel went completely dark for the full
                    # duration of a 40-90-block forward pass.
                    #
                    # gamma/v6 follow-up: the v5 fix throttled to a
                    # 10-second wall-clock interval plus a final-block
                    # fallback. For a 64-block model where the entire
                    # prefill finishes in well under 10s, that meant
                    # only the LAST block ever emitted -- effectively
                    # the same "silent panel" problem the fix was meant
                    # to solve. The chunk-level \\r-progress above
                    # already covers token progress; this loop covers
                    # block progress, which is what the user actually
                    # sees when the prompt is tiny (chunk count == 1).
                    #
                    # Cadence:
                    #   * ATF_PREFILL_HEARTBEAT=0  -> disable entirely
                    #   * ATF_PREFILL_HEARTBEAT_BLOCKS=N -> emit every Nth
                    #     block (default: ~8 evenly-spaced updates,
                    #     regardless of num_blocks; floor of 1, ceiling
                    #     of num_blocks).
                    #   * 10-second wall-clock backstop still fires if a
                    #     single block hangs longer than the block
                    #     cadence would otherwise produce.
                    _hb_enabled = os.environ.get("ATF_PREFILL_HEARTBEAT", "1") \
                            != "0"
                    if _hb_enabled:
                        try:
                            _hb_every = int(os.environ.get(
                                "ATF_PREFILL_HEARTBEAT_BLOCKS", "0"))
                        except ValueError:
                            _hb_every = 0
                        if _hb_every <= 0:
                            # Adapt to model size: aim for ~8 updates
                            # across the pass. Floor at 1 (every block)
                            # so tiny models still show progress; cap at
                            # num_blocks to avoid div-by-zero and to
                            # guarantee the last block always fires.
                            _hb_every = max(1, self.num_blocks // 8)
                        _hb_every = min(_hb_every, max(1, self.num_blocks))
                        _now_abs = time.time()
                        _last_abs = getattr(
                            self, "_last_prefill_heartbeat_abs", 0.0)
                        _hb_due = (
                            # block-cadence boundary (every Nth block,
                            # plus the final block so the user always
                            # sees completion)
                            (b + 1) % _hb_every == 0
                            or b == self.num_blocks - 1
                            # wall-clock backstop (v5 behavior)
                            or (_now_abs - _last_abs) > 10.0)
                        if _hb_due:
                            self._last_prefill_heartbeat_abs = _now_abs
                            # Reset the elapsed-since-chunk sentinel too
                            # so any callers reading it stay sane.
                            self._last_prefill_heartbeat = \
                                    _now_abs - _chunk_t0
                            console.print(f"  [dim][step] prefill block "
                                          f"{b+1}/{self.num_blocks} "
                                          f"({(b+1)/self.num_blocks*100:.0f}%), "
                                          f"last_block={_bdt:.0f}ms "
                                          f"mem={gpu_mem_line()}[/dim]")
                            # gamma/v6: forward fine-grained per-block progress
                            # to the bridge so the UI bar actually moves during
                            # single-chunk prefills (the chunk-level callback at
                            # the bottom of this function only fires once for
                            # prompts under 1024 tokens, which is most of them).
                            # We map (b+1)/num_blocks of the current chunk's
                            # token range so the renderer sees a smooth 0->100%.
                            if progress_cb is not None:
                                _tok_done = s + ((b + 1) / self.num_blocks) * len(piece)
                                progress_cb(int(_tok_done), len(token_ids))
                if os.environ.get("ATF_PREFILL_EVAL_PER_BLOCK"):
                    # Diagnostic: force sync per block so a GPU wedge is
                    # attributed to one block instead of the whole graph.
                    import sys as _sys
                    print(f"[prefill-eval] block {b}", file=_sys.stderr,
                          flush=True)
                    dump = os.environ.get("ATF_DUMP_BLOCK_X")
                    if dump and b == int(os.environ.get("DIAG_BLOCK", "41")):
                        np.save(dump, np.array(x))   # forces sync + captures
                        print(f"[prefill-eval] dumped x@{b} -> {dump}",
                              file=_sys.stderr, flush=True)
                    mx.eval(x)
                # Level 4: NaN check on every block's residual output. Catches
                # MoE/dequant blow-ups before they corrupt later blocks.
                if _VERBOSE >= 4 or _NAN_TRAP:
                    _nan_check(x, where=f"prefill.b{b}.x")
            x = _rmsnorm(x, self.model.w("output_norm.weight"))
            logits = self.model.mm("output.weight", x[-1])
            mx.eval(logits)
            if progress_cb is not None:
                done = min(s + max(1, chunk), len(token_ids))
                progress_cb(done, len(token_ids))
                # v17: visible heartbeat on stdout (status log / API log tail);
                # carriage return keeps it to one line per prompt.
                print(f"\r  Prefilling {done}/{len(token_ids)} tokens",
                      end="", flush=True)
            _chunk_dt = time.time() - _chunk_t0
            _vstate_add("prefill_chunks", 1)
            _vstate_add("prefill_tokens", len(piece))
            if _VERBOSE >= 2:
                tok_per_s = (len(piece) / _chunk_dt) if _chunk_dt > 0 else 0.0
                _vlog(2, f"_forward_prefill.chunk idx={chunk_idx}/{n_chunks} "
                        f"tokens=[{s},{s+len(piece)}) of {len(token_ids)} "
                        f"t={_chunk_dt:.2f}s tok/s={tok_per_s:.1f} "
                        f"mem={gpu_mem_line()}")
        except Exception as _prefill_exc:
            _vlog(1, f"_forward_prefill.EXCEPTION "
                    f"err={type(_prefill_exc).__name__}: {_prefill_exc} "
                    f"cur_active={int(mx.get_active_memory()) if hasattr(mx, 'get_active_memory') else '?'} "
                    f"mem={gpu_mem_line()}")
            raise
        if progress_cb is not None:
            print("", flush=True)   # newline before the final Prefill summary
        _vlog(2, f"_forward_prefill.done total={time.time()-_prefill_t0:.2f}s "
                f"n_chunks={n_chunks}")
        # Defensive: if the chunked loop never ran (e.g. the v10 sys_baseline
        # LCP overshoot we just fixed in the caller), logits stayed None and
        # the first _sample crashed with a confusing
        # "LM head returned invalid logits shape ()" -- surface the real
        # cause here so the bridge shows a useful error instead.
        if logits is None:
            raise RuntimeError(
                "_forward_prefill received an empty token list (len==0); "
                "the LCP cache matched the entire prompt and left nothing "
                "to prefill. This is a v10 sys_baseline bug — file a bug "
                "with the prompt and sys_text that triggered it."
            )
        return logits

    def _forward_token(self, token_id: int, kv_caches: dict, gdn_states: dict,
                       exact_ffn: bool = True,
                       prev_token_id: int | None = None) -> mx.array:
        x = self.model.embed(token_id)[None, :]
        diag = os.environ.get("ATF_DECODE_EVAL_PER_BLOCK")
        if diag:
            import sys as _sys
            _t_prev = time.time()
        for b in range(self.num_blocks):
            # gamma/v2: n-gram insertion at the configured layer (T=1
            # path, so the pair-id gather is a single index lookup).
            if (self.ngram and self.ngram["insert_layer"] == b):
                # caller passes the predecessor (the token generated
                # just before this one); for the first decode step
                # after prefill, the caller's last prefill token id is
                # the right predecessor.
                ptid = 0 if prev_token_id is None else prev_token_id
                x = self._ngram_insert(x, [ptid], [token_id])
            _bt0 = time.time()
            x = self._forward_block(b, x, kv_caches.get(b), gdn_states, exact_ffn)
            _bdt = (time.time() - _bt0) * 1000.0
            _vstate_add("decode_block_ms_total", _bdt)
            _vstate_add("decode_block_calls", 1)
            if _VERBOSE >= 3:
                kind = "attn" if b in self.full_attn_blocks else "gdn"
                _vlog(3, f"_forward_token.block b={b:3d} ({kind}) "
                        f"t={_bdt:.1f}ms")
            if _VERBOSE >= 4 or _NAN_TRAP:
                _nan_check(x, where=f"decode.b{b}.x", raise_on_nan=False)
            if diag:
                mx.eval(x)
                _t_now = time.time()
                kind = "full-attn" if b in self.full_attn_blocks else "GDN"
                print(f"[decode-eval] block {b:3d} ({kind:9s}) "
                      f"{(_t_now - _t_prev) * 1000:8.2f} ms",
                      file=_sys.stderr, flush=True)
                _t_prev = _t_now
        x = _rmsnorm(x, self.model.w("output_norm.weight"))
        out = self.model.mm("output.weight", x[-1])
        if diag:
            mx.eval(out)
            print(f"[decode-eval] lm_head       {(time.time() - _t_prev) * 1000:8.2f} ms",
                  file=_sys.stderr, flush=True)
        return out

    # ─── speculative decoding (gamma/v3.1, shallow self-draft) ──────────
    # SPEC_DECODING_PROPOSAL.md Phase 2. Medusa/EAGLE-style self-
    # speculative decoding: draft K tokens via an early-exit forward
    # through only the first self._spec_shallow_layers blocks (reusing
    # this model's own LM head on that intermediate hidden state), then
    # verify all K in ONE batched forward through every block. No second
    # model, no residency problem, no cross-tokenizer-alignment risk (see
    # the proposal's §2a for why this was chosen over a two-engine
    # 9B-drafts-27B design). Accept/reject uses atf/spec.py's pure
    # functions (Phase 1).

    @staticmethod
    def _snapshot_state(kv_caches: dict, gdn_states: dict):
        """O(num_blocks) snapshot of KV-cache lengths + GDN recurrent
        state, for rolling back a speculative draft/verify cycle.

        Static (no `self` use) so tests/test_spec_shallow_integration.py
        can exercise the real snapshot/restore logic directly against
        real KVCache instances and plain dicts -- no Engine instance and
        no fake/mock model required.

        IMPORTANT: gdn_states[b] is an inner dict that _gated_deltanet
        mutates IN PLACE ("st = gdn_states[b]; st['state'] = ...") rather
        than rebinding gdn_states[b] itself to a new dict. A naive
        dict(gdn_states) shallow copy therefore does NOT protect against
        later mutation -- the copied dict still points at the SAME inner
        dict objects, whose 'state'/'conv_buf' values change underneath
        you as soon as another _gated_deltanet call runs. This copies one
        level deeper (a fresh dict per block) so the snapshot is real. The
        mx.array values inside are safe to share by reference: they are
        never mutated in place, only replaced -- MLX arrays are immutable
        under every op this engine uses (see SPEC_DECODING_PROPOSAL.md
        §2b and §7 Risk 5).
        """
        kv_n = {b: c.n for b, c in kv_caches.items()}
        gdn_snap = {b: dict(v) for b, v in gdn_states.items()}
        return kv_n, gdn_snap

    @staticmethod
    def _restore_state(kv_caches: dict, gdn_states: dict,
                       kv_n: dict, gdn_snap: dict) -> None:
        """Undo everything a draft/verify cycle did to kv_caches/gdn_states,
        back to the point _snapshot_state was called. Static for the same
        testability reason as _snapshot_state."""
        for b, n in kv_n.items():
            if b in kv_caches:
                kv_caches[b].n = n
        gdn_states.clear()
        gdn_states.update(gdn_snap)

    def _draft_k_tokens(self, seed_token_id: int, k: int, kv_caches: dict,
                        gdn_states: dict, config: GenConfig,
                        history: list[int]):
        """Draft K tokens via a shallow early-exit forward.

        Runs only the first self._spec_shallow_layers blocks (NOT the
        full self.num_blocks), then applies the same output_norm +
        lm_head used for the real final layer directly to that
        intermediate hidden state. This is an approximation -- the model
        was never trained to produce sensible logits at this depth -- but
        it is a known, legitimate technique (the Medusa/EAGLE self-
        speculative-decoding family); see SPEC_DECODING_PROPOSAL.md §2a
        option 3 for why ATF picked it over a second draft model.

        MUTATES kv_caches/gdn_states for blocks 0..shallow_layers-1 as it
        goes (each drafted token needs the previous draft token's state
        to continue autoregressively). Callers MUST snapshot before
        calling and restore after (see _spec_step) -- the state this
        leaves behind is never the real, correct state; only the
        subsequent full-stack verify pass produces that.

        Returns (draft_tokens: list[int], draft_logits: np.ndarray[K,vocab]).
        """
        L = self._spec_shallow_layers
        draft_tokens: list[int] = []
        draft_logits: list[np.ndarray] = []
        tok_id = seed_token_id
        for _ in range(k):
            x = self.model.embed(tok_id)[None, :]
            for b in range(L):
                x = self._forward_block(b, x, kv_caches.get(b), gdn_states,
                                        config.exact_ffn)
            x_final = _rmsnorm(x, self.model.w("output_norm.weight"))
            step_logits = self.model.mm("output.weight", x_final[-1])
            # Phase 0 finding: clamp the LM-head padding rows before
            # sampling. The shallow-layer logits are noisier than the
            # full-stack verify pass, so a noisy softmax here is the
            # most likely place to ever pick a padding id (which the
            # tokenizer panics on at decode time). clamp_padding is
            # a no-op if the input vocab is smaller than the
            # padding-start id (defensive for Qwen2-derivative vocabs).
            from .spec import clamp_padding
            step_logits_clamped = clamp_padding(step_logits)
            drafted = self._sample(step_logits_clamped, config,
                                    history + draft_tokens)
            draft_tokens.append(drafted)
            # Save the *clamped* logits (so the accept test's
            # p_draft(padding) is exactly 0.0, not a near-zero
            # float -- avoids float underflow corner cases).
            draft_logits.append(np.asarray(step_logits_clamped,
                                            dtype=np.float64).reshape(-1))
            tok_id = drafted
        return draft_tokens, np.stack(draft_logits, axis=0)

    def _forward_chunk_all_logits(self, token_ids: list[int], kv_caches: dict,
                                  gdn_states: dict, exact_ffn: bool) -> mx.array:
        """Like _forward_prefill, but for a single small chunk (no
        sub-chunking -- callers pass a handful of tokens, e.g. a
        speculative-decoding draft/correction batch) and returns the
        LM-head logits at EVERY position (shape [T, vocab]), not just the
        last. _forward_prefill only returns the final position's logits
        (the next-token prediction after a full prompt prefill);
        speculative verification needs one target-logits row per drafted
        position. Runs through ALL self.num_blocks (unlike
        _draft_k_tokens, which only runs the shallow subset), so this is
        the expensive-but-real forward pass, batched over T tokens in one
        call for the whole speedup speculative decoding is for.
        """
        x = self.model.embed_batch(token_ids)
        for b in range(self.num_blocks):
            x = self._forward_block(b, x, kv_caches.get(b), gdn_states, exact_ffn)
        x = _rmsnorm(x, self.model.w("output_norm.weight"))
        return self.model.mm("output.weight", x)   # [T, vocab]

    def _spec_step(self, seed_token_id: int, kv_caches: dict, gdn_states: dict,
                   config: GenConfig, history: list[int],
                   ema_accept: float = 1.0):
        """One draft -> verify -> accept/reject cycle.

        Per SPEC_DECODING_PROPOSAL.md §3, with the rollback simplified
        relative to the proposal's own pseudocode (see _snapshot_state's
        docstring for why a naive shallow dict-copy doesn't actually
        protect GDN state) and relative to per-step GDN snapshotting
        (which the proposal's §3a describes but which needs different
        code for each of the three _gated_deltanet execution paths --
        eager loop, ATF_GDN_SCAN, ATF_COMPILE_STEP -- to hook into): on
        a full accept, the one batched verify forward already leaves
        kv_caches/gdn_states in exactly the right state, so nothing
        further is needed. On a partial reject, this discards that
        forward's state entirely and reruns a second, smaller batched
        forward over just the correct committed prefix (accepted tokens
        + the residual-sampled correction) -- always correct, at the
        cost of a second (cheaper) forward pass on the less-common
        rejection path. The common (full-accept) path pays no extra cost.

        Returns (committed_tokens, next_logits, accept_rate):
          - committed_tokens: list[int] of tokens committed this cycle.
          - next_logits: mx.array target distribution for the token
            AFTER the last committed one, ready to feed back into
            the normal decode loop exactly like _forward_token's
            return value would.
          - accept_rate: float in [0, 1] -- this cycle's
            acceptance rate (committed_count / k). The caller
            (generate) folds it into an EMA and may auto-disable
            spec if the EMA falls below config.spec_min_acceptance.
            On a full reject path that does the redo forward, the
            accept_rate is reported as 0 (this is the "really bad
            draft" case that auto-disable is designed to catch).
        """
        from .spec import spec_decode_step

        k = max(1, config.spec_k)
        kv_n0, gdn0 = self._snapshot_state(kv_caches, gdn_states)

        draft_tokens, draft_logits = self._draft_k_tokens(
            seed_token_id, k, kv_caches, gdn_states, config, history)

        # Undo the draft's advance of the shallow blocks -- the verify
        # pass below re-derives everything from scratch through ALL
        # blocks, so the draft's partial state must not linger.
        self._restore_state(kv_caches, gdn_states, kv_n0, gdn0)

        # Verify the FULL sequence [seed, draft_1, ..., draft_K] through
        # the full model. The verify forward returns K+1 logit rows;
        # row 0 is the target distribution at the position immediately
        # after seed (i.e. what the drafter should have produced at
        # position 1), row 1 is the target after [seed, draft_1], etc.
        # We slice off row 0 (it's the same as the seed's next-token
        # distribution the caller already has from the previous
        # iteration) and use verify[1..K] for the accept test.
        verify_full_input = [seed_token_id] + list(draft_tokens)
        verify_logits_all = self._forward_chunk_all_logits(
            verify_full_input, kv_caches, gdn_states, config.exact_ffn)
        mx.eval(verify_logits_all)
        # Slice off verify row 0 (target distribution at the position
        # immediately after seed; not used -- we already sampled that
        # position in the previous iteration, and there's no draft
        # token to evaluate against it). Use verify[1..K] for the
        # accept test against draft_tokens[0..K-1].
        # Clamp padding rows in verify logits too. The full-stack
        # 27B forward is far less likely than the drafter to put
        # mass on padding rows (those rows are zero-initialized in
        # the .atf), but a numerically-noisy softmax can still
        # produce a tiny positive value that would otherwise get
        # the drafter's id through accept_token as a degenerate
        # case (p_target/p_draft < 1 always, so always-reject).
        from .spec import clamp_padding
        verify_logits = clamp_padding(
            np.asarray(verify_logits_all, dtype=np.float64)[1:])

        rand_values = [float(np.random.random()) for _ in range(k)]
        accepted, n_accepted = spec_decode_step(
            draft_tokens, draft_logits, verify_logits, rand_values)

        k_actual = max(1, k)
        if n_accepted == k_actual:
            # Full accept: the verify forward's state IS the correct
            # state (it processed exactly the tokens we're committing:
            # [seed, draft_1, ..., draft_K] -- K+1 positions). Its
            # logits at the last position are already the target
            # distribution for the token after the last committed one.
            # Cache is correctly advanced by K+1; nothing more to do.
            next_logits = verify_logits_all[-1]
            return accepted, next_logits, 1.0

        # Partial reject: the verify forward advanced state by K+1
        # tokens, but only n_accepted + 1 of them (the accepted prefix
        # + one correction) are actually being committed. Discard that
        # state and redo a smaller, correct forward over just the
        # committed sequence -- [seed, accepted_draft_1, ..., correction]
        # -- with the seed prepended for the same reason as above
        # (verify row 0 must be the target distribution at the position
        # immediately after the seed, not after the first accepted draft).
        self._restore_state(kv_caches, gdn_states, kv_n0, gdn0)
        redo_input = [seed_token_id] + list(accepted)
        redo_logits_all = self._forward_chunk_all_logits(
            redo_input, kv_caches, gdn_states, config.exact_ffn)
        mx.eval(redo_logits_all)
        next_logits = redo_logits_all[-1]
        # Report this cycle's accept rate: accepted_count / k.
        # Excluding the residual correction token (the +1 in the
        # committed list) so the rate reflects pure draft-quality.
        draft_accepts = max(0, n_accepted)
        return accepted, next_logits, draft_accepts / k_actual

    def _forward_block(self, b: int, x: mx.array, cache: KVCache | None,
                       gdn_states: dict, exact_ffn: bool = True) -> mx.array:
        prefix = f"blk.{b}."
        attn_in = _rmsnorm(x, self.model.w(f"{prefix}attn_norm.weight"))

        # gamma/v3: QSA dispatch comes FIRST (before full_attn check)
        # because QSA blocks also have attn_q.weight (same projection
        # family as full-attn for the main path) -- the indexer
        # presence is the discriminator. See __init__ autodetect
        # comment for why the order matters.
        if b in self.qsa_blocks:
            if cache is None:
                cache = KVCache()
            attn_out = self._attention_qsa(b, attn_in, cache)
        elif b in self.full_attn_blocks:
            if cache is None:
                cache = KVCache()
            attn_out = self._attention_full(b, attn_in, cache)
        else:
            attn_out = self._gated_deltanet(b, attn_in, gdn_states)

        x = x + attn_out
        ffn_in = _rmsnorm(x, self.model.w(f"{prefix}post_attention_norm.weight"))
        x = x + self._ffn(b, ffn_in, exact_ffn)
        return x

    # ─── n-gram embedding (gamma/v2) ─────────────────────────────────────
    def _ngram_insert(self, x: mx.array, prev_token_ids: list[int],
                      cur_token_ids: list[int]) -> mx.array:
        """Add the n-gram contribution to the residual stream.

        Per docs/qwen38_flash_next_analysis.md §6.5:

            For each token t in this chunk:
                pair_id = encode(t-1, t)             # bigram: pair of consecutive token ids
                ngram_act = take(ngram_table, pair_id)  # [T, ngram_dim], LOD-dequantized
            ngram_act = rmsnorm(ngram_act, ngram_emb_norm)
            ngram_act = ngram_act @ ngram_emb_proj.weight   # [T, hidden]
            x = x + ngram_act                              # residual add

        Args:
            x: [T, hidden] residual stream entering layer ngram_insert_layer.
            prev_token_ids: token id of position t-1 for each of the T tokens
                (length T). For t=0 in the chunk there is no predecessor; the
                pair id is 0 (or any out-of-vocab id; the table is designed to
                be padded so id 0 is the BOS/null row).
            cur_token_ids: the T token ids in this chunk.

        Returns:
            x with the n-gram contribution added (same shape, same dtype).
        """
        if not self.ngram:
            return x
        if not (self.ngram["has_table"] and self.ngram["has_proj"]
                and self.ngram["has_norm"]):
            # Partial setup: silently no-op rather than crash. The console
            # init line already reported which pieces are missing.
            return x
        m = self.model
        T = x.shape[0]
        # Build pair ids. For bigram (context==2) we use (t-1, t). For
        # trigram (context==3) we use (t-2, t-1, t) hashed to a single id
        # (vocab is sized for it). The hash is a deterministic mix
        # -- a real Qwen4 GGUF will precompute the table for its hash
        # function, so this is the convention the analysis doc pins.
        vocab = self.ngram["vocab"]
        ctx = self.ngram["context"]
        pair_ids = np.zeros((T,), dtype=np.int64)
        # Pair-id hash. The Qwen4 release pins the convention that
        # pair (a, b) and (b, a) land in DIFFERENT rows of the n-gram
        # table -- the table is a learned embedding, not a symmetric
        # kernel. We use two distinct large primes to break the
        # symmetry that a single-multiplier hash has for some inputs.
        # (See the parity test for the test pin.)
        if ctx == 2:
            for i, (p, c) in enumerate(zip(prev_token_ids, cur_token_ids)):
                pair_ids[i] = (int(p) * 2000003 + int(c) * 3000017) % vocab
        elif ctx == 3:
            # Need two predecessors; pad with 0 if not available. Caller
            # is responsible for providing the right prev list.
            for i, c in enumerate(cur_token_ids):
                p1 = prev_token_ids[i] if i < len(prev_token_ids) else 0
                p2 = prev_token_ids[i - 1] if i >= 1 else 0
                pair_ids[i] = (int(p1) * 2000003
                               + int(p2) * 4000033
                               + int(c) * 5000011) % vocab
        else:
            raise ValueError(f"ngram_context must be 2 or 3, got {ctx}")
        # mx.take along axis 0 of the table. The table is loaded as f32
        # by model.w() -- the dense section is int8 dequantized; the
        # expert-section table is LOD-paged. Either way, .w() returns
        # a dequantized f16/f32 mx.array.
        # w_ngram() routes the big table through the LRU-paging path;
        # the proj/norm are tiny dense tensors (also via w_ngram, which
        # falls through to w() for those).
        table = m.w_ngram("ngram_emb.weight")
        norm_w = m.w_ngram("ngram_emb_norm.weight")
        proj_w = m.w_ngram("ngram_emb_proj.weight")
        idx = mx.array(pair_ids)
        ngram_act = table[idx]                          # [T, ngram_dim]
        ngram_act = _rmsnorm(ngram_act, norm_w)         # [T, ngram_dim]
        # proj_w is [ngram_dim, hidden] (engine-facing). @ broadcasts.
        ngram_act = ngram_act @ proj_w                  # [T, hidden]
        return x + ngram_act

    # ─── full attention ──────────────────────────────────────────────────

    def _attention_full(self, b: int, x: mx.array, cache: KVCache) -> mx.array:
        m = self.model
        prefix = f"blk.{b}."
        T = x.shape[0]
        nh, nk, hd = self.num_heads, self.num_kv, self.head_dim

        qg, k, v = m.qmm_fuse([f"{prefix}attn_q.weight", f"{prefix}attn_k.weight",
                              f"{prefix}attn_v.weight"], x)
        qg = qg.reshape(T, nh, 2, hd)                     # [T, nh * 2 * hd]
        q = qg[:, :, 0, :]                               # per-head interleaved q|gate
        gate = qg[:, :, 1, :]

        k = k.reshape(T, nk, hd)
        v = v.reshape(T, nk, hd)

        q = _rmsnorm(q, m.w(f"{prefix}attn_q_norm.weight"))
        k = _rmsnorm(k, m.w(f"{prefix}attn_k_norm.weight"))

        start_pos = cache.n
        positions = mx.arange(start_pos, start_pos + T)
        q = _apply_rope(q, positions, self.rope_freqs)
        k = _apply_rope(k, positions, self.rope_freqs)

        rep = nh // nk
        cache._ensure(cache.capacity or self.max_context_default(), nk, hd)

        # v9.1: feed k/v through the cache in bounded query chunks. Computing
        # the whole [nh, T, S+T] fp32 score matrix at once peaks at several
        # GB once the history grows (24 heads x 1024 queries x 32k+ context
        # x 4 B ~= 3+ GB per block); chunking caps it at O(QC) instead.
        QC = 256 if T > 256 else T
        outs = []
        for qs in range(0, T, QC):
            qe = min(qs + QC, T)
            qc, kc, vc = q[qs:qe], k[qs:qe], v[qs:qe]
            Tq = qe - qs
            start_pos = cache.n
            cache.append(kc, vc)
            k_full = cache.keys()
            v_full = cache.values()
            k_exp = mx.repeat(k_full.astype(mx.float32), rep, axis=1)
            v_exp = mx.repeat(v_full.astype(mx.float32), rep, axis=1)

            q_h = qc.transpose(1, 0, 2)
            k_h = k_exp.transpose(1, 0, 2)
            v_h = v_exp.transpose(1, 0, 2)

            scores = mx.matmul(q_h, k_h.transpose(0, 2, 1)) / math.sqrt(hd)
            total = start_pos + Tq
            masked = mx.arange(total)[None, :] > (start_pos + mx.arange(Tq))[:, None]
            scores = mx.where(masked[None], mx.array(float("-inf")), scores)
            attn = _softmax(scores, axis=-1)
            ctx = mx.matmul(attn, v_h).transpose(1, 0, 2).reshape(Tq, -1)
            outs.append(m.qmm(f"{prefix}attn_output.weight",
                              ctx * mx.sigmoid(gate[qs:qe]).reshape(Tq, -1)))
        return mx.concatenate(outs, axis=0)

    # ─── QSA (Qwen Sparse Attention, gamma/v3) ───────────────────────────
    def _attention_qsa(self, b: int, x: mx.array, cache: KVCache) -> mx.array:
        """Qwen Sparse Attention forward, parallel to _attention_full.

        Per docs/qwen38_flash_next_analysis.md §3.3:

            1. Compute per-block indexer scores. The indexer is a tiny
               MQA: qsa_indexer_heads (4) query heads + qsa_kv_heads (1)
               shared key head, each head_dim 128 (per Qwen3.8-Flash-Next
               model card; values not measured on ATF). The score is
               q_idx @ k_idx_per_block, producing [num_blocks] fp16
               scores (one per KV block of size qsa_block_size tokens).
            2. Select top-K blocks: topk(scores, k=qsa_budget_blocks).
               Qwen4 spec: 512 blocks. Per-token attention cost is now
               bounded by the top-K fetch, not by context length.
            3. Sparse-gather K/V from the cache for just those blocks.
               The fetched set is at most qsa_budget_blocks * qsa_block_size
               tokens (Qwen4: 512 * 16 = 8192).
            4. Standard softmax attention over the fetched set: q @
               k_fetched.T, softmax, @ v_fetched. Same as _attention_full
               but over the small fetched subset.

        Args:
            b: block index (must be in self.qsa_blocks).
            x: [T, hidden] residual stream after attn_norm.
            cache: KVCache for this block (shared with full-attn path
                structurally; the cache stores the full history, the
                QSA path just reads a sparse subset per query).

        Returns:
            [T, hidden] attention output, ready to be added to the
            residual stream.
        """
        m = self.model
        prefix = f"blk.{b}."
        T = x.shape[0]
        nh, nk, hd = self.num_heads, self.num_kv, self.head_dim
        indexer_h = self.qsa_indexer_heads
        indexer_kv = self.qsa_kv_heads
        budget = self.qsa_budget_blocks
        block_size = self.qsa_block_size

        # --- Stage 1: indexer scores (per query) ---
        # The indexer is a small MQA. q_proj outputs [T, indexer_h * hd],
        # k_proj outputs [T, indexer_kv * hd]. Per the model card, the
        # indexer head_dim is 128 -- the constant the analysis doc
        # pins. We use the same head_dim as the main attention unless
        # a future variant has a separate one.
        indexer_hd = 128  # Qwen3.8-Flash-Next model card; pinned assumption
        q_idx = m.w_qsa_indexer(f"{prefix}attn_indexer.q_proj.weight")
        k_idx = m.w_qsa_indexer(f"{prefix}attn_indexer.k_proj.weight")
        # Apply QK-norm if present (the Qwen4 spec includes it). The
        # tensors are optional; if absent, the indexer is unnormalized
        # which is a slight quality regression but not a correctness one.
        if m.has(f"{prefix}attn_indexer.q_norm.weight"):
            q_idx_n = m.w_qsa_indexer(f"{prefix}attn_indexer.q_norm.weight")
            q_idx = _rmsnorm(q_idx.reshape(T, indexer_h, indexer_hd), q_idx_n).reshape(T, -1)
        if m.has(f"{prefix}attn_indexer.k_norm.weight"):
            k_idx_n = m.w_qsa_indexer(f"{prefix}attn_indexer.k_norm.weight")
            k_idx = _rmsnorm(k_idx.reshape(T, indexer_kv, indexer_hd), k_idx_n).reshape(T, -1)

        # --- Main attention Q (per-head interleaved q|gate) ---
        qg, k, v = m.qmm_fuse([f"{prefix}attn_q.weight",
                              f"{prefix}attn_k.weight",
                              f"{prefix}attn_v.weight"], x)
        qg = qg.reshape(T, nh, 2, hd)
        q = qg[:, :, 0, :]
        gate = qg[:, :, 1, :]
        k = k.reshape(T, nk, hd)
        v = v.reshape(T, nk, hd)
        q = _rmsnorm(q, m.w(f"{prefix}attn_q_norm.weight"))
        k = _rmsnorm(k, m.w(f"{prefix}attn_k_norm.weight"))

        # --- Stage 1 (cont): per-block indexer scores ---
        # For each block of size block_size in the KV cache, the indexer
        # score is the mean of q_idx[t] @ k_idx[block_start:block_end].T
        # (this matches the Qwen4 spec: block-level scoring, not per-token).
        # We compute the full [T, num_blocks] score matrix, then top-K
        # select for each query. At prefill, this is O(T * num_blocks)
        # which is bounded by (chunk * num_blocks); at decode, T=1 and
        # num_blocks = cache.n // block_size.
        cache._ensure(cache.capacity or self.max_context_default(), nk, hd)
        start_pos = cache.n
        positions = mx.arange(start_pos, start_pos + T)
        # The indexer k must also be RoPE'd (Qwen4 spec) -- but with the
        # indexer head_dim (128) and the same rope_freqs. For v3, we
        # skip indexer RoPE because the indexer is much smaller and the
        # RoPE implementation _apply_rope expects head_dim that matches
        # self.rope_freqs shape. Future v3.x can add it.
        k_main_with_rope = k  # placeholder so the rest is uniform
        # Append main K/V to cache
        cache.append(k, v)
        num_blocks = max(1, cache.n // block_size)
        # Indexer score matrix: [T, num_blocks]. For decode (T=1), this
        # is a single row; for prefill (T>1), this is a banded matrix
        # (causal -- block b is only visible to query at position t if
        # b*block_size <= t).
        k_idx_r = k_idx.reshape(cache.n, indexer_kv, indexer_hd)
        # Per-block k: [num_blocks, indexer_kv, indexer_hd] = mean over the
        # block. This is the "block-level" key that the Qwen4 spec describes.
        k_block = k_idx_r[:num_blocks * block_size].reshape(
            num_blocks, block_size, indexer_kv, indexer_hd
        ).mean(axis=1)  # [num_blocks, indexer_kv, indexer_hd]
        # q_idx: [T, indexer_h, indexer_hd] -- broadcast with k_block.
        q_idx_r = q_idx.reshape(T, indexer_h, indexer_hd)
        # Per-block score: [T, num_blocks] = mean over heads of q @ k.T
        scores = (q_idx_r @ k_block.transpose(1, 0, 2)  # [T, indexer_h, num_blocks]
                  ).mean(axis=1) / math.sqrt(indexer_hd)
        # Causal mask: block b is only visible if b*block_size <= t (the
        # current query position). For decode (T=1) this is just
        # b*block_size <= start_pos; for prefill it's a banded mask.
        block_starts = mx.arange(num_blocks) * block_size
        causal_mask = block_starts[None, :] <= (start_pos + mx.arange(T))[:, None]
        scores = mx.where(causal_mask, scores, mx.array(float("-inf")))

        # --- Stage 2: top-K block selection ---
        # budget blocks per query. argpartition gives O(n) top-K.
        k_actual = min(budget, num_blocks)
        # If num_blocks is small (early in generation), k_actual = num_blocks
        # and we attend to everything; if k_actual < num_blocks, sparse.
        if k_actual >= num_blocks:
            # No sparsification needed -- attend to all blocks.
            selected = mx.arange(num_blocks)[None, :].repeat(T, axis=0)  # [T, num_blocks]
        else:
            # argpartition over the last axis, take the top k_actual. The
            # returned indices are NOT sorted; we sort them for the
            # gather to produce a contiguous block range (for cache locality).
            topk_idx = mx.argpartition(-scores, k_actual - 1, axis=-1)[:, :k_actual]
            # Sort by block index (so the gathered K/V is contiguous)
            selected = mx.sort(topk_idx, axis=-1)  # [T, k_actual]

        # --- Stage 3: sparse-gather K/V from cache ---
        # k_full: [num_blocks * block_size, nk, hd] = cache.keys()
        # We want k_gathered[t, s, head, dim] = k_full[selected[t, s//block_size] * block_size + (s % block_size), head, dim]
        # This is a gather over a 2-D index -- mlx.gather_nd or manual
        # via take + reshape. For v3, we do it via take() of a flat
        # block index, then expand to per-token.
        k_full = cache.keys()    # [cache.n, nk, hd]
        v_full = cache.values()  # [cache.n, nk, hd]
        # Compute flat per-token indices: block_id * block_size + offset_within_block
        offset_within = mx.arange(block_size)  # [block_size]
        # selected: [T, k_actual] of block ids -> [T, k_actual, block_size] of token ids
        token_idx = (selected[:, :, None] * block_size
                     + offset_within[None, None, :]).reshape(T, k_actual * block_size)
        # Clamp to cache size (defensive -- selected could include a partial last block)
        token_idx = mx.minimum(token_idx, cache.n - 1)
        # Gather
        k_gathered = k_full[token_idx.reshape(-1)].reshape(T, k_actual * block_size, nk, hd)
        v_gathered = v_full[token_idx.reshape(-1)].reshape(T, k_actual * block_size, nk, hd)

        # --- Stage 4: standard attention over the fetched set ---
        # Same pattern as _attention_full but over a small fetched set
        # (max budget * block_size = 8192 tokens for Qwen4) -- no need
        # for chunked scores.
        rep = nh // nk
        q_h = q.transpose(1, 0, 2)  # [nh, T, hd]
        k_h = mx.repeat(k_gathered.transpose(1, 0, 2).astype(mx.float32), rep, axis=1)
        v_h = mx.repeat(v_gathered.transpose(1, 0, 2).astype(mx.float32), rep, axis=1)
        attn_scores = mx.matmul(q_h, k_h.transpose(0, 2, 1)) / math.sqrt(hd)
        # Causal mask: only attend to positions <= current.
        fetched_len = k_actual * block_size
        causal = (mx.arange(fetched_len)[None, :]
                  <= (start_pos + mx.arange(T))[:, None])
        attn_scores = mx.where(causal[None], attn_scores, mx.array(float("-inf")))
        attn = _softmax(attn_scores, axis=-1)
        ctx = mx.matmul(attn, v_h).transpose(1, 0, 2).reshape(T, -1)
        return m.qmm(f"{prefix}attn_output.weight",
                     ctx * mx.sigmoid(gate).reshape(T, -1))

    # ─── Gated DeltaNet ──────────────────────────────────────────────────

    def _gdn_step_compiled(self, q_t: mx.array, k_t: mx.array, v_t: mx.array,
                           beta_t: mx.array, g_t: mx.array,
                           state: mx.array) -> tuple[mx.array, mx.array]:
        """mx.compile'd single-token GDN delta-rule recurrence (Win A).

        Exactly the T==1 body of the per-timestep loop in
        _gated_deltanet below, factored out so mx.compile can trace it
        once (shapes are fixed for the lifetime of this Engine -- they're
        derived from the model header in __init__ and never change) and
        reuse the compiled graph for every decode step of every GDN block
        for the rest of generation. Pure function of its arguments, no
        Python-side mutation, so it needs none of MLX's in-place-update
        machinery -- just ordinary compiled tracing of new-array-out ops,
        identical to what the uncompiled loop already does.
        """
        fn = self._gdn_step_fn
        if fn is None:
            @mx.compile
            def _step(q_t, k_t, v_t, beta_t, g_t, state):
                state = state * mx.exp(g_t)[:, None, None]
                # state layout [Hv, Dv, Dk]; retrieval/insertion contract
                # key dim along the LAST axis (matches the uncompiled loop).
                kv_mem = mx.sum(state * k_t[:, None, :], axis=-1)
                delta = (v_t - kv_mem) * beta_t[:, None]
                state = state + delta[:, :, None] * k_t[:, None, :]
                y = mx.sum(state * q_t[:, None, :], axis=-1)
                return y, state
            fn = _step
            self._gdn_step_fn = fn
        return fn(q_t, k_t, v_t, beta_t, g_t, state)

    def _get_gdn_full_compiled_fn(self):
        """Build and cache the mx.compile'd full single-token GDN step function (Win A Expanded)."""
        fn = self._gdn_full_step_fn
        if fn is not None:
            return fn

        n_kh = self.gdn_n_k_heads
        n_vh = self.gdn_n_v_heads
        dk = self.gdn_head_k_dim
        dv = self.gdn_head_v_dim
        kd = n_kh * dk
        rep = n_vh // n_kh if n_kh else 1
        is_raw = bool(getattr(self.model, "raw", None))

        @mx.compile
        def _full_step(x, qkv_raw, z, conv_buf, state,
                       conv_w, alpha_w, beta_w, dt_bias, ssm_a, ssm_norm):
            alpha_raw = x @ alpha_w
            beta_raw = x @ beta_w
            alpha_biased = alpha_raw + dt_bias
            alpha_sp = mx.log(1.0 + mx.exp(alpha_biased))
            g = ssm_a[None, :] * alpha_sp
            beta = mx.sigmoid(beta_raw)

            ksize = conv_w.shape[0]
            full_seq = mx.concatenate([conv_buf, qkv_raw], axis=0)
            conv_out = mx.zeros((1, qkv_raw.shape[1]), dtype=mx.float32)
            for i in range(ksize):
                conv_out = conv_out + conv_w[i][None, :] * full_seq[i:i + 1]
            next_conv_buf = full_seq[-(ksize - 1):] if ksize > 1 else mx.zeros((0, qkv_raw.shape[1]))
            conv_out = conv_out * mx.sigmoid(conv_out)

            q = conv_out[:, :kd].reshape(1, n_kh, dk)
            k = conv_out[:, kd:2 * kd].reshape(1, n_kh, dk)
            v = conv_out[:, 2 * kd:].reshape(1, n_vh, dv)

            q = _l2norm(q)
            k = _l2norm(k)
            q = q * (dk ** -0.5)

            if rep > 1:
                if is_raw:
                    q = mx.tile(q, (1, rep, 1))
                    k = mx.tile(k, (1, rep, 1))
                else:
                    q = mx.repeat(q, rep, axis=-2)
                    k = mx.repeat(k, rep, axis=-2)

            q0 = q[0]
            k0 = k[0]
            v0 = v[0]
            g0 = g[0]
            beta0 = beta[0]

            st = state * mx.exp(g0)[:, None, None]
            kv_mem = mx.sum(st * k0[:, None, :], axis=-1)
            delta = (v0 - kv_mem) * beta0[:, None]
            next_state = st + delta[:, :, None] * k0[:, None, :]
            yt = mx.sum(next_state * q0[:, None, :], axis=-1)
            y = yt[None, :, :]

            y = _rmsnorm(y, ssm_norm)
            zg = z.reshape(1, n_vh, dv)
            y = y * (zg * mx.sigmoid(zg))
            return y.reshape(1, -1), next_conv_buf, next_state

        self._gdn_full_step_fn = _full_step
        return _full_step

    def _gated_deltanet(self, b: int, x: mx.array, gdn_states: dict) -> mx.array:
        m = self.model
        prefix = f"blk.{b}."
        n_kh, n_vh = self.gdn_n_k_heads, self.gdn_n_v_heads
        dk, dv = self.gdn_head_k_dim, self.gdn_head_v_dim
        T = x.shape[0]

        qkv_raw, z = m.qmm_fuse(
            [f"{prefix}attn_qkv.weight", f"{prefix}attn_gate.weight"], x)

        # Win A+B: Fused single-token compiled GDN step with pinned weights
        if T == 1 and self._compile_full_gdn and b in self._gdn_pinned:
            if b not in gdn_states:
                conv_w = self._gdn_pinned[b][0]
                ksize = conv_w.shape[0]
                cch = conv_w.shape[1]
                gdn_states[b] = {
                    "conv_buf": mx.zeros((ksize - 1, cch), dtype=mx.float32),
                    "state": mx.zeros((n_vh, dv, dv), dtype=mx.float32),
                }
            st = gdn_states[b]
            conv_w, alpha_w, beta_w, dt_bias, ssm_a, ssm_norm = self._gdn_pinned[b]
            step_fn = self._get_gdn_full_compiled_fn()
            y_flat, next_cb, next_st = step_fn(
                x, qkv_raw, z, st["conv_buf"], st["state"],
                conv_w, alpha_w, beta_w, dt_bias, ssm_a, ssm_norm
            )
            st["conv_buf"] = next_cb
            st["state"] = next_st
            return m.qmm(f"{prefix}ssm_out.weight", y_flat)

        # decay/beta gates are precision-critical: read the f16-stored
        # projections through w() instead of bundling them into the 4-bit
        # qmm group (weights stored GGUF-style [in,out]).
        alpha_raw = x @ m.w(f"{prefix}ssm_alpha.weight")
        beta_raw = x @ m.w(f"{prefix}ssm_beta.weight")

        alpha_biased = alpha_raw + m.w(f"{prefix}ssm_dt.bias")
        alpha_sp = mx.log(1.0 + mx.exp(alpha_biased))
        g = m.w(f"{prefix}ssm_a")[None, :] * alpha_sp      # negative decay log

        beta = mx.sigmoid(beta_raw)

        conv_w = m.w(f"{prefix}ssm_conv1d.weight")
        ksize, cch = conv_w.shape
        if b not in gdn_states:
            gdn_states[b] = {
                "conv_buf": mx.zeros((ksize - 1, cch), dtype=mx.float32),
                "state": mx.zeros((n_vh, dv, dv), dtype=mx.float32),
            }
        st = gdn_states[b]

        full_seq = mx.concatenate([st["conv_buf"], qkv_raw], axis=0)
        conv_out = mx.zeros((T, cch), dtype=mx.float32)
        for i in range(ksize):
            conv_out = conv_out + conv_w[i][None, :] * full_seq[i:i + T]
        st["conv_buf"] = full_seq[-(ksize - 1):] if ksize > 1 else mx.zeros((0, cch))
        conv_out = conv_out * mx.sigmoid(conv_out)

        kd = n_kh * dk
        q = conv_out[:, :kd].reshape(T, n_kh, dk)
        k = conv_out[:, kd:2 * kd].reshape(T, n_kh, dk)
        v = conv_out[:, 2 * kd:].reshape(T, n_vh, dv)

        q = _l2norm(q)
        k = _l2norm(k)
        q = q * (dk ** -0.5)

        if n_kh != n_vh:
            rep = n_vh // n_kh
            # Head-expansion pairing is SOURCE-DEPENDENT:
            #  - MLX/HF checkpoints store v-heads grouped [v0..v31], so
            #    v-head 2j/2j+1 pair with k-head j (adjacent repeat,
            #    matching mlx_lm mx.repeat semantics).
            #  - llama.cpp GGUF checkpoints store a permuted v-head order
            #    under which the historical blocked tile pairing is the one
            #    that works (verified empirically: repeat scrambles GGUF
            #    outputs, tile keeps them coherent).
            if self.model.raw:
                q = mx.tile(q, (1, rep, 1))
                k = mx.tile(k, (1, rep, 1))
            else:
                q = mx.repeat(q, rep, axis=-2)
                k = mx.repeat(k, rep, axis=-2)

        if os.environ.get("ATF_GDN_SCAN") == "1":
            # v8 fast path (~6x/block): exact algorithm, but an fp32
            # conditioning issue on real weights is still open -- opt-in only.
            y, state = _gdn_chunk_scan(q, k, v, beta, g, st["state"])
            st["state"] = state
        elif self._compile_gdn_step:
            # v20: extended from decode-only (T==1) to prefill's multi-token
            # chunks too. Same per-timestep body as the uncompiled loop
            # below (state_t depends on state_{t-1}, so this is still O(T)
            # sequential -- mx.compile does NOT parallelize across t, it
            # only fuses each step's ~6 elementwise/reduce ops into one
            # command buffer instead of ~6 separate eager launches). Math
            # and iteration order are identical to the uncompiled loop;
            # only dispatch overhead changes. Opt-in via ATF_COMPILE_STEP=1
            # -- verify with tests/test_perf_parity_compile_step.py
            # (byte-identical output, temperature=0, before trusting this
            # for real use) before relying on it, same as the original
            # decode-only Win A.
            state = st["state"]
            ys = []
            for t in range(T):
                y_t, state = self._gdn_step_compiled(q[t], k[t], v[t],
                                                     beta[t], g[t], state)
                ys.append(y_t)
            st["state"] = state
            y = mx.stack(ys, axis=0)
        else:
            state = st["state"]
            ys = []
            for t in range(T):
                state = state * mx.exp(g[t])[:, None, None]
                # state layout [Hv, Dv, Dk]; retrieval/insertion contract key
                # dim along the LAST axis (matches gated_delta step ops).
                kv_mem = mx.sum(state * k[t][:, None, :], axis=-1)   # [Hv, Dv]
                delta = (v[t] - kv_mem) * beta[t][:, None]
                state = state + delta[:, :, None] * k[t][:, None, :]
                ys.append(mx.sum(state * q[t][:, None, :], axis=-1)) # [Hv, Dv]
            st["state"] = state
            y = mx.stack(ys, axis=0)

        y = _rmsnorm(y, m.w(f"{prefix}ssm_norm.weight"))
        zg = z.reshape(T, n_vh, dv)
        y = y * (zg * mx.sigmoid(zg))
        return m.qmm(f"{prefix}ssm_out.weight", y.reshape(T, -1))

    # ─── FFN (dense SwiGLU; experts merged to 3 dense matmuls in v8) ────

    def _ffn(self, b: int, x: mx.array, exact_ffn: bool = True) -> mx.array:
        if b in self.moe_blocks:
            return self._ffn_moe(b, x)
        m = self.model
        n_exp = self.num_experts

        if exact_ffn or self.top_k >= n_exp:
            # v8: the 8 chunk-experts are an storage-level split of ONE dense
            # FFN (gate/up split along the intermediate dim, down along the
            # output dim). Merging them into 3 dense 4-bit matrices once and
            # doing 3 quantized matmuls replaces 24 per block -- identical
            # math, 8x fewer kernel launches.
            m.merge_ffn(b)
            g, u = m.qmm_fuse([f"blk.{b}.ffn_gate.weight.dense",
                              f"blk.{b}.ffn_up.weight.dense"], x)
            inter = mx.sigmoid(g) * g * u
            return m.qmm(f"blk.{b}.ffn_down.weight.dense", inter)

        router = m.router[b]                      # [hidden, n_exp]
        scores = mx.mean(x[:, :, None] * router[None], axis=1)
        topk = mx.argpartition(-scores, self.top_k, axis=-1)[:, :self.top_k]
        rows = []
        for t in range(x.shape[0]):
            acc = mx.zeros((self.hidden,), dtype=mx.float32)
            for e in topk[t]:
                g = m.qmm(f"blk.{b}.ffn_gate.weight.e{int(e)}", x[t][None])[0]
                u = m.qmm(f"blk.{b}.ffn_up.weight.e{int(e)}", x[t][None])[0]
                inter = mx.sigmoid(g) * g * u
                acc = acc + m.qmm(f"blk.{b}.ffn_down.weight.e{int(e)}", inter[None])[0]
            rows.append(acc)
        return mx.stack(rows, axis=0)

    # ─── true-MoE FFN (v19.1) ─────────────────────────────────────────────
    #
    # scores = x @ router_b ([T, n_exp]); softmax in fp32; top-k with
    # k = min(top_k, n_exp); out += weight_e * down_e(silu(gate_e(x)) * up_e(x))
    # plus an always-active shared expert whose output is scaled per channel
    # by sigmoid(ffn_gate_inp_shexp). Tokens are grouped per expert
    # (vectorized across T; no per-token Python loop on the hot path).

    def _ffn_moe(self, b: int, x: mx.array) -> mx.array:
        m = self.model
        p = f"blk.{b}."
        T = x.shape[0]
        _moe_debug = os.environ.get("ATF_MOE_DEBUG") == "1"
        _t_block0 = time.time()
        _misses_before = m.cache_misses if hasattr(m, "cache_misses") else 0
        if _moe_debug:
            import sys as _sys
        _vlog(3, f"_ffn_moe.enter b={b} T={T} top_k={self.top_k} "
                f"mem={gpu_mem_line()}")

        # ── shared expert (always active) + per-token scalar sigmoid gate
        # v4 (2026-09-03): if this file has REAL trained shexp records (the
        # genuine Qwen3.5-MoE shared expert, as in real pretrained MoE
        # checkpoints like the 35B-A3B model), use them exactly as before --
        # plain f32 matmuls via w(), since the shexp intermediate dim is
        # often smaller than the 64-group MLX quantize minimum and raw-GGUF
        # shexp dtypes vary, so w() handles every backend/dtype uniformly.
        # UNCHANGED for any file that has these records -- this branch is
        # byte-identical to the pre-v4 code.
        #
        # If the file has NO shexp records (a convert_moe.py "reuse" mode
        # conversion, which cannot train a real shared expert), fall back to
        # computing the shared-expert term directly from the block's own
        # ORIGINAL dense FFN tensors -- the exact same qmm()/merge_ffn()
        # path the plain dense (non-MoE) forward pass a few methods below
        # already uses. This reuses an already-proven-correct,
        # Metal-accelerated dispatch instead of an untrained, narrow,
        # CPU-dequantized synthetic slice. Zero extra file bytes (no new
        # tensors read), and makes the shared expert mathematically equal
        # to the block's normal dense FFN output -- as coherent as the
        # underlying dense model, at the cost of NOT reducing this block's
        # active compute for the shared term (routed experts on top are
        # still small/sparse). See notes.md 2026-09-03.
        if m.has(f"{p}ffn_gate_shexp.weight"):
            gs = x @ m.w(f"{p}ffn_gate_shexp.weight")
            us = x @ m.w(f"{p}ffn_up_shexp.weight")
            sh_inter = mx.sigmoid(gs) * gs * us
            sh = sh_inter @ m.w(f"{p}ffn_down_shexp.weight")
            # Qwen3_5MoeSparseMoeBlock.shared_expert_gate is Linear(hidden, 1).
            # The GGUF tensor ffn_gate_inp_shexp.weight stores the row-vector W
            # of shape [hidden]; the reference treats it as [1, hidden] so the
            # matmul `x @ W` produces a [T, 1] per-token scalar. Sigmoid is then
            # the per-token gate that scales the shared-expert output. This is
            # a real TRAINED weight for a genuine MoE checkpoint, so it's a
            # small, calibrated dot product -- safe to apply here.
            sh_gate = mx.sigmoid(x @ m.w(f"{p}ffn_gate_inp_shexp.weight"))  # [T, 1]
            sh = sh * sh_gate[:, None]   # broadcast [T, 1] * [T, hidden]
            if _VERBOSE >= 4 or _NAN_TRAP:
                _nan_check(sh, where=f"moe.b{b}.shared_out")
        else:
            m.merge_ffn(b)
            gs, us = m.qmm_fuse([f"{p}ffn_gate.weight.dense",
                                f"{p}ffn_up.weight.dense"], x)
            sh_inter = mx.sigmoid(gs) * gs * us
            sh = m.qmm(f"{p}ffn_down.weight.dense", sh_inter)
            # v4.2 (2026-09-03): do NOT apply ffn_gate_inp_shexp gating here.
            # convert_moe.py writes that tensor as a CONSTANT 10.0 across all
            # hidden dims (not a real trained Linear(hidden,1) weight), so
            # `x @ sgate` = 10 * sum(x) -- a large-magnitude number whose sign
            # depends on sum(x), not the small calibrated dot product a real
            # gate would produce. sigmoid() of that SATURATES HARD to exactly
            # 0 or 1 per token per block, rather than staying smoothly near
            # 1.0 as the constant was meant to approximate -- likely zeroing
            # out entire blocks' FFN contribution unpredictably and
            # corrupting the residual stream (confirmed by real test:
            # coherent first token, "Hello"/"The", then immediate EOS). sh is
            # already exactly the dense model's own FFN output here -- no
            # gate is needed or correct to apply; return it unmodified.
            if _VERBOSE >= 4 or _NAN_TRAP:
                _nan_check(sh, where=f"moe.b{b}.reuse_shared_out")
            return sh

        # v4.1 (2026-09-03): in reuse mode (no real shexp records), sh is
        # ALREADY the complete, correct dense FFN output -- the routed
        # experts below are untrained slices of that SAME dense tensor,
        # selected by a zero-init (non-functional) router and renormalized
        # to sum to ~1.0 weight, i.e. the same order of magnitude as sh's
        # own ~1.0 gate. Adding them on top roughly doubles this block's
        # FFN contribution to the residual stream every block, compounding
        # over 64 blocks -- confirmed by real test to produce garbage
        # output. They add no real information here (same tensor sh
        # already used in full) so skip them entirely and return the
        # correct dense-equivalent output directly. Real trained MoE files
        # (m.has(...) True) are unaffected -- they still fall through to
        # the router/routed-expert path below exactly as before.
        # (v4.2 note: the reuse-mode return now happens above, in the else
        # branch, before this comment is even reached -- this comment/dead
        # code path only applies to the has()==True branch now, which never
        # takes it since has()==True doesn't set the reuse early-return
        # condition. Kept as a no-op guard for safety.)
        if not m.has(f"{p}ffn_gate_shexp.weight"):
            return sh

        # ── router: softmax fp32 over all experts, then top-k ────────────
        scores = _softmax(x @ m.moe_router(b), axis=-1)          # [T, n_exp]
        k = min(self.top_k, scores.shape[1])
        idx = mx.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]  # [T, k]
        wts = mx.take_along_axis(scores, idx, axis=-1)             # [T, k]
        # Qwen3_5MoeTopKRouter renormalizes top-k weights so the routed
        # contribution is on the same scale as the shared expert (~1.0).
        wts = wts / wts.sum(axis=-1, keepdims=True)
        if _VERBOSE >= 4 or _NAN_TRAP:
            _nan_check(scores, where=f"moe.b{b}.router_scores")
            _nan_check(wts, where=f"moe.b{b}.router_weights")

        # Host-side grouping only over the tiny [T, k] index matrix; all
        # tensor math stays on GPU. Within one top-k slot every token
        # appears at most once, so per-slot row assignment cannot collide.
        idx_np = np.asarray(idx).astype(np.int64)            # [T, k]
        # One dense accumulator per top-k slot: within a slot every token
        # appears at most once, so per-slot row writes cannot collide; the
        # slot buffers are then summed (a token's k contributions ADD).
        routed = mx.zeros_like(x)
        _new_misses_total = 0
        for s in range(k):
            col = idx_np[:, s]
            wcol = wts[:, s]
            acc_s = mx.zeros_like(x)
            for e in np.unique(col).tolist():
                rows = mx.array(np.nonzero(col == e)[0].astype(np.uint32))
                Xe = mx.take(x, rows, axis=0)                # [n_sel, hidden]
                # gamma/v5 fast-MoE fix: qmm() directly instead of
                # moe_expert(). moe_expert()/_dequant_expert_slice()
                # dequantized the whole expert on CPU via the slow
                # pure-Python gguf reference decoder (fine for correctness,
                # ~100-1000x slower than the Metal kernels already used
                # for every dense weight in this engine). qmm() resolves
                # "blk.{b}.moe_{proj}.weight.e{e}" through the SAME
                # per-expert-slice addressing (see _raw_base_uncached),
                # but computes Xe @ W_slice directly through the existing
                # fast dispatch (_qmm_gemm batched-GEMM when n_sel is
                # large enough, else the gguf_matmul per-row Metal
                # kernel) -- no CPU dequant, no full fp32 weight matrix
                # ever materialized. cache_misses bookkeeping below no
                # longer applies (qmm's own _gemm_cache tracks reuse
                # separately) so _new_misses_total is left at 0 here.
                g = m.qmm(f"blk.{b}.ffn_gate.weight.e{e}", Xe)
                u = m.qmm(f"blk.{b}.ffn_up.weight.e{e}", Xe)
                inter = mx.sigmoid(g) * g * u
                y = m.qmm(f"blk.{b}.ffn_down.weight.e{e}", inter)  # [n_sel, hidden]
                if _VERBOSE >= 4 or _NAN_TRAP:
                    _nan_check(y, where=f"moe.b{b}.e{e}.y", raise_on_nan=False)
                wsel = mx.take(wcol, rows, axis=0)
                acc_s[rows] = y * wsel[:, None]
            routed = routed + acc_s
        result = routed + sh
        if _VERBOSE >= 4 or _NAN_TRAP:
            _nan_check(result, where=f"moe.b{b}.result")
        _n_unique_total = len(np.unique(idx_np))
        _elapsed = time.time() - _t_block0
        _vstate_add("moe_blocks_seen", 1)
        _vstate_add("moe_unique_experts_sum", _n_unique_total)
        if _n_unique_total > _VERBOSE_STATE.get("moe_unique_experts_max", 0):
            _vstate_set("moe_unique_experts_max", _n_unique_total)
        _vstate_add("moe_cache_misses_sum", _new_misses_total)
        _vstate_add("moe_t", _elapsed)
        _vlog(2, f"_ffn_moe.done b={b} T={T} top_k={k} "
                f"unique_experts={_n_unique_total} "
                f"new_cache_misses={_new_misses_total} "
                f"t={_elapsed*1000:.1f}ms")
        if _moe_debug:
            mx.eval(result)
            print(f"[moe-debug] block {b:3d}: T={T} top_k={k} "
                  f"unique_experts_needed={_n_unique_total} "
                  f"new_cache_misses={_new_misses_total} "
                  f"block_total={_elapsed*1000:8.1f}ms",
                  file=_sys.stderr, flush=True)
        return result
    # ─── sampling ────────────────────────────────────────────────────────

    def _sample(self, logits: mx.array, config: GenConfig, history: list[int]) -> int:
        # Decode projections may arrive as [1, vocab] from a backend kernel;
        # sampling always operates on one flat vocabulary row.
        _st0 = time.time()
        lg = np.asarray(logits, dtype=np.float64).reshape(-1)
        if lg.size <= 1:
            raise ValueError(
                f"LM head returned invalid logits shape {np.shape(logits)}"
            )

        # NaN/Inf trap on the input to sampling. Catches the classic
        # "model produced garbage, sampling crashed later" case early.
        _has_nan = bool(np.isnan(lg).any()) or bool(np.isinf(lg).any())
        if _has_nan:
            _vstate_add("sample_logits_nan", 1)
            if _NAN_WARN:
                _sys.stderr.write(
                    f"[atf-nan] _sample.input logits_nan={int(np.isnan(lg).sum())} "
                    f"logits_inf={int(np.isinf(lg).sum())}\n")
                _sys.stderr.flush()
            if _NAN_TRAP:
                raise RuntimeError(
                    f"[ATF_NAN_TRAP] _sample received NaN/Inf logits "
                    f"nan={int(np.isnan(lg).sum())} inf={int(np.isinf(lg).sum())}")

        if config.repeat_penalty != 1.0 and history:
            recent = history[-64:]
            for tid in set(recent):
                if lg[tid] > 0:
                    lg[tid] /= config.repeat_penalty
                else:
                    lg[tid] *= config.repeat_penalty

        if config.temperature <= 0.0:
            argmax_id = int(np.argmax(lg))
            _vstate_add("sample_t", time.time() - _st0)
            if _VERBOSE >= 3:
                _vlog(3, f"_sample.greedy tok={argmax_id} "
                        f"logit={float(lg[argmax_id]):.3f} "
                        f"nan_input={_has_nan} t={(time.time()-_st0)*1000:.1f}ms")
            return argmax_id

        lg = lg / config.temperature
        if config.top_p < 1.0:
            order = np.argsort(-lg)
            sorted_lg = lg[order]
            cum = np.cumsum(_softmax_np(sorted_lg))
            cutoff = cum > config.top_p
            cutoff[0] = False
            lg[order[cutoff]] = float("-inf")
        probs = _softmax_np(lg)
        tok = int(np.random.choice(len(probs), p=probs))
        _vstate_add("sample_t", time.time() - _st0)
        if _VERBOSE >= 3:
            order_top = np.argsort(-probs)[:5]
            top = [(int(t), float(probs[t])) for t in order_top]
            _vlog(3, f"_sample.tok={tok} top5={top} "
                    f"temperature={config.temperature} top_p={config.top_p} "
                    f"nan_input={_has_nan} t={(time.time()-_st0)*1000:.1f}ms")
        return tok


def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─── v8: exact chunk-parallel GatedDeltaNet scan ─────────────────────────
# Replaces the per-timestep Python loop. Validated against the sequential
# reference to 1.5e-07 worst relative error over 60 randomized trials
# (H,T,chunk,decay varied; see docs/v8_gdn_scan_validation.py).
#
# Per chunk of C tokens (all exponents <= 0 => overflow-safe), with raw
# deltas as unknowns:
#   L_t     = cumsum(g)[t]                                  (log decay)
#   M[t,s]  = 1 if t==s else beta_t * exp(L_t - L_s)*(k_s . k_t),  s < t
#   B_t     = beta_t * (v_t - exp(L_t) * S0 k_t)
#   Delta   = solve(M, B)
#   S_out   = exp(L_C) * S0 + sum_s exp(L_C - L_s) * Delta_s k_s^T
#   Y_t     = exp(L_t)*q_t.S0 + sum_{s<=t} exp(L_t - L_s)*(q_t.Delta_s) k_s
#
# Shapes (batched over heads): q,k:[T,H,dk] v:[T,H,dv] beta,g:[T,H]
# S0,[H,dv,dk] -> Y:[T,H,dk], S_out:[H,dv,dk].
def _gdn_chunk_scan(q, k, v, beta, g, S0):
    T = q.shape[0]
    C = Engine.GDN_CHUNK
    S = S0.astype(mx.float32)
    ys = []
    eye_c = {}
    for s in range(0, T, C):
        e = min(s + C, T)
        c = e - s
        qc = q[s:e].astype(mx.float32).transpose(1, 0, 2)   # [H,c,dk]
        kc = k[s:e].astype(mx.float32).transpose(1, 0, 2)   # [H,c,dk]
        vc = v[s:e].astype(mx.float32).transpose(1, 0, 2)   # [H,c,dv]
        bc = beta[s:e].astype(mx.float32).transpose(1, 0)   # [H,c]
        L = mx.cumsum(g[s:e].astype(mx.float32), axis=0).transpose(1, 0)  # [H,c]

        KK = kc @ kc.transpose(0, 2, 1)                     # [H,t,s]: k_t.k_s
        # Real models can emit very large per-token decays (|g| > 300), making
        # upper-triangle exponents hugely positive -> exp() overflows to inf
        # and inf*0 == NaN inside the masked product. Clamp exponents at <= 0;
        # the masked-out region is multiplied away afterwards anyway.
        ratio = mx.exp(mx.minimum(L[:, :, None] - L[:, None, :], 0.0))   # <= 1
        if c not in eye_c:
            eye_c[c] = mx.eye(c, dtype=mx.float32)
        offdiag = mx.arange(c)[None, :] < mx.arange(c)[:, None]
        M = eye_c[c] + bc[:, :, None] * ratio * KK * offdiag

        Sk = kc @ S.transpose(0, 2, 1)                      # [H,c,dv]: S0 @ k_t
        B = bc[:, :, None] * (vc - mx.exp(L)[:, :, None] * Sk)
        if c == 1:
            Dl = B                                          # M is [1,1] identity
        else:
            # MLX linalg.solve runs on CPU only; used for prefill chunks.
            # The copy back is tiny ([H,c,dv]) relative to the ~64x reduction
            # in dispatched ops this scan buys.
            Dl = mx.linalg.solve(M, B, stream=mx.cpu)

        QD = qc @ Dl.transpose(0, 2, 1)                     # [H,t,s]: q_t.Delta_s
        orat = mx.exp(mx.minimum(L[:, :, None] - L[:, None, :], 0.0))
        tril = (mx.arange(c)[None, :] <= mx.arange(c)[:, None]).astype(mx.float32)
        w = mx.exp(L[:, -1:] - L)                           # exp(L_C - L_s) <= 1
        Sold = S
        S = (mx.exp(L[:, -1])[:, None, None] * S
             + (kc.transpose(0, 2, 1) @ (Dl * w[:, :, None])).transpose(0, 2, 1))
        # v20 fix: this needs Sold.transpose(0, 2, 1), matching the Sk
        # computation above (`kc @ S.transpose(0, 2, 1)`) -- state is stored
        # [H, dv, dk], and q (indexed by dk) must contract against dk, i.e.
        # Sold's LAST axis, not its middle (dv) axis. The untransposed
        # `qc @ Sold` only avoided a shape error because dk == dv is forced
        # equal elsewhere in this codebase (gdn_head_k_dim = gdn_head_v_dim
        # in Engine.__init__) -- it silently contracted over the wrong axis
        # instead of raising, producing shape-valid but numerically wrong
        # output. This was the cause of the garbled ATF_GDN_SCAN=1 output
        # reported and reverted earlier this session -- see LOOP_ANALYSIS.md.
        # y_t[d] = sum_o q_t[o] S_t[o,d]  ->  qc @ Sold^T ([H,c,dk] @ [H,dk,dv])
        yt = (mx.exp(L)[:, :, None] * (qc @ Sold.transpose(0, 2, 1))
              + ((QD * orat * tril) @ kc))                  # [H,c,dk]
        ys.append(yt.transpose(1, 0, 2))
    return mx.concatenate(ys, axis=0), S
