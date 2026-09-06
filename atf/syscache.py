"""Disk persistence for system-prompt KV state (v10, syscache).

After the first request that includes a system prompt, the engine's _pcache
holds the KV state for the system portion. We serialize that to disk keyed
by (model_id, system_hash) so subsequent server restarts can load it back,
eliminating the multi-minute system prefill from the very first request.

Cache file: ~/.cache/atf/syscache/<model_id>/<sys_hash>.npz
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import hashlib
import numpy as np
from pathlib import Path

log = logging.getLogger("atf.syscache")

CACHE_ROOT = Path(
    os.environ.get("ATF_SYSCACHE_DIR",
                   Path.home() / ".cache" / "atf" / "syscache")
)

# In-memory mirror: per-process cache of (sys_ids, sys_kv, sys_gdn)
# so we don't have to re-load from disk on every request after the first.
_MEMO: dict[str, tuple] = {}


def hash_system(sys_text: str, tools) -> str:
    """Cache key: SHA-256(system_text + tools)[:24]."""
    h = hashlib.sha256()
    h.update(sys_text.encode("utf-8"))
    for t in (tools or []):
        h.update(json.dumps(t, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:24]


def cache_path(model_id: str, sys_hash: str) -> Path:
    """Path to the cache file for a (model_id, system_hash) pair."""
    safe_model = model_id.replace("/", "_").replace(":", "_")
    return CACHE_ROOT / safe_model / f"{sys_hash}.npz"


def save_cache(model_id: str, sys_hash: str, sys_ids: list[int],
               sys_kv: dict, sys_gdn: dict) -> bool:
    """Serialize the system-prompt KV state to disk.

    sys_kv: dict[block_idx -> KVCache]  (KVCache has k_buf, v_buf, n, n_kv, head_dim)
    sys_gdn: dict[block_idx -> dict]     (each value has 'state' and 'conv_buf' arrays)

    Returns True on success. Errors are logged but not raised.
    """
    try:
        path = cache_path(model_id, sys_hash)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {}
        metadata = {
            "blocks": [],
            "n_sys": len(sys_ids),
            "saved_at": time.time(),
        }

        # Serialize KVCache objects
        for block_idx, cache in sys_kv.items():
            try:
                k_np = np.array(cache.k_buf[:cache.n])  # [n, n_kv, head_dim]
                v_np = np.array(cache.v_buf[:cache.n])
                arrays[f"k_{block_idx}"] = k_np.astype(np.float16)
                arrays[f"v_{block_idx}"] = v_np.astype(np.float16)
                metadata["blocks"].append({
                    "idx": int(block_idx),
                    "n": int(cache.n),
                    "n_kv": int(cache.n_kv),
                    "head_dim": int(cache.head_dim),
                })
            except Exception as exc:
                log.warning("syscache: failed to serialize block %d: %s",
                            block_idx, exc)
                return False

        # Serialize GDN states
        gdn_meta = []
        for block_idx, state in (sys_gdn or {}).items():
            try:
                if "state" in state:
                    arrays[f"gdn_{block_idx}"] = np.array(state["state"]).astype(np.float32)
                if "conv_buf" in state and state["conv_buf"].size > 0:
                    arrays[f"conv_{block_idx}"] = np.array(state["conv_buf"]).astype(np.float32)
                gdn_meta.append({
                    "idx": int(block_idx),
                    "has_state": "state" in state,
                    "conv_size": int(state["conv_buf"].size) if "conv_buf" in state else 0,
                })
            except Exception as exc:
                log.warning("syscache: failed to serialize GDN block %d: %s",
                            block_idx, exc)
        metadata["gdn_blocks"] = gdn_meta

        arrays["sys_ids"] = np.array(sys_ids, dtype=np.int32)

        # Write atomically
        tmp_path = path.with_suffix(".tmp")
        np.savez_compressed(tmp_path, **arrays,
                            _metadata=np.array([json.dumps(metadata)], dtype=object))
        tmp_path.replace(path)

        size_mb = path.stat().st_size / (1 << 20)
        log.info("syscache: SAVED  %s  blocks=%d  n_sys=%d  size=%.1fMB",
                 path.name, len(metadata["blocks"]), len(sys_ids), size_mb)
        # Invalidate memo
        _MEMO.pop(f"{model_id}:{sys_hash}", None)
        return True

    except Exception as exc:
        log.warning("syscache: save failed: %s", exc)
        return False


def load_cache(model_id: str, sys_hash: str):
    """Load the system-prompt KV state from disk.

    Returns (sys_ids, sys_kv, sys_gdn) on success, or None.
    sys_kv: dict[block_idx -> KVCache] (MLX-backed, ready for engine._pcache)
    sys_gdn: dict[block_idx -> dict]   (with 'state' and/or 'conv_buf' as MLX arrays)
    """
    memo_key = f"{model_id}:{sys_hash}"
    if memo_key in _MEMO:
        return _MEMO[memo_key]

    try:
        path = cache_path(model_id, sys_hash)
        if not path.exists():
            return None

        start = time.time()
        # Lazy import to avoid MLX cost if not used
        import mlx.core as mx
        from atf.engine import KVCache

        loaded = np.load(path, allow_pickle=True)
        metadata_raw = loaded.get("_metadata", None)
        if metadata_raw is None:
            return None
        metadata = json.loads(str(metadata_raw[0]))

        sys_ids = loaded["sys_ids"].tolist()

        sys_kv = {}
        for bm in metadata["blocks"]:
            idx = bm["idx"]
            try:
                k_arr = loaded[f"k_{idx}"]  # numpy [n, n_kv, head_dim] fp16
                v_arr = loaded[f"v_{idx}"]
                cache = KVCache()
                cache.n_kv = bm["n_kv"]
                cache.head_dim = bm["head_dim"]
                cache.n = bm["n"]
                cache.k_buf = mx.array(k_arr)
                cache.v_buf = mx.array(v_arr)
                sys_kv[idx] = cache
            except KeyError:
                log.warning("syscache: missing KV data for block %d", idx)
                return None

        sys_gdn = {}
        for gm in metadata.get("gdn_blocks", []):
            idx = gm["idx"]
            state = {}
            try:
                if gm.get("conv_size", 0) > 0:
                    state["conv_buf"] = mx.array(loaded[f"conv_{idx}"])
                if gm.get("has_state"):
                    state["state"] = mx.array(loaded[f"gdn_{idx}"])
                if state:
                    sys_gdn[idx] = state
            except KeyError:
                pass  # OK to skip

        elapsed = time.time() - start
        size_mb = path.stat().st_size / (1 << 20)
        log.info("syscache: LOADED  %s  blocks=%d  n_sys=%d  size=%.1fMB  "
                 "load=%.2fs", path.name, len(sys_kv), len(sys_ids),
                 size_mb, elapsed)
        result = (sys_ids, sys_kv, sys_gdn)
        _MEMO[memo_key] = result
        return result

    except Exception as exc:
        log.warning("syscache: load failed: %s", exc)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def try_inject(engine, model_id: str, sys_text: str, tools) -> bool:
    """If a disk cache exists for (model_id, hash_system(sys_text, tools)),
    load it and inject into engine._pcache so the next generate() skips the
    system prefill via the LCP path. Returns True on a cache hit."""
    if engine is None or not sys_text:
        return False
    sys_hash = hash_system(sys_text, tools)
    loaded = load_cache(model_id, sys_hash)
    if loaded is None:
        return False
    sys_ids, sys_kv, sys_gdn = loaded
    try:
        engine._pcache = (sys_ids, sys_kv, sys_gdn)
        log.info("syscache: INJECTED  n_sys=%d  blocks=%d  model=%s",
                 len(sys_ids), len(sys_kv), model_id)
        return True
    except Exception as exc:
        log.warning("syscache: inject failed: %s", exc)
        try:
            engine._pcache = None
        except Exception:
            pass
        return False


def save_from_engine(engine, model_id: str, sys_text: str, tools) -> bool:
    """After a successful generate() that included a system prompt, snapshot
    the system portion of engine._pcache and persist to disk.

    The engine's _pcache holds KV state for the ENTIRE prefilled prompt;
    we cap each cache to n_sys tokens and write the system portion to disk.
    """
    if engine is None or not sys_text:
        return False
    pcache = getattr(engine, "_pcache", None)
    if pcache is None:
        return False
    try:
        old_ids, old_kv, old_gdn = pcache
    except Exception:
        return False
    tok = getattr(engine, "tok", None)
    if tok is None:
        return False

    # Render the system block in the same shape the engine saw.
    if tools:
        from .toolcall import append_tools_to_system
        body = append_tools_to_system(sys_text, tools)
    else:
        body = sys_text
    sys_block = f"<|im_start|>system\n{body}<|im_end|>"
    try:
        sys_ids = tok.encode(sys_block)
    except Exception as exc:
        log.warning("syscache: encode failed: %s", exc)
        return False
    n_sys = len(sys_ids)
    if n_sys < 1 or n_sys > len(old_ids):
        return False

    # Cap each KVCache to n_sys. The underlying buffer is shared
    # (zero copy); we save a VIEW that is the system portion.
    sys_kv = {}
    for block_idx, cache in old_kv.items():
        try:
            if cache.n >= n_sys:
                # Take a sub-view of n_sys tokens (no copy of data)
                new_cache = type(cache)()
                new_cache.n_kv = cache.n_kv
                new_cache.head_dim = cache.head_dim
                new_cache.n = n_sys
                # Slicing MLX arrays is a view; data is shared
                new_cache.k_buf = cache.k_buf[:n_sys]
                new_cache.v_buf = cache.v_buf[:n_sys]
                sys_kv[block_idx] = new_cache
        except Exception:
            pass
    if not sys_kv:
        return False

    sys_gdn = dict(old_gdn) if old_gdn else {}
    sys_hash = hash_system(sys_text, tools)
    return save_cache(model_id, sys_hash, sys_ids, sys_kv, sys_gdn)


def list_caches() -> list[dict]:
    """List all cached system prompts (for debugging)."""
    if not CACHE_ROOT.exists():
        return []
    results = []
    for model_dir in CACHE_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        for cache_file in model_dir.glob("*.npz"):
            try:
                loaded = np.load(cache_file, allow_pickle=True)
                md = loaded.get("_metadata", None)
                if md is not None:
                    metadata = json.loads(str(md[0]))
                    results.append({
                        "model": model_dir.name,
                        "hash": cache_file.stem,
                        "n_sys": metadata.get("n_sys", 0),
                        "size_mb": round(cache_file.stat().st_size / (1<<20), 1),
                        "saved_at": metadata.get("saved_at", 0),
                    })
            except Exception:
                pass
    return results
