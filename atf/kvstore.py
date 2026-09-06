"""Paged SSD KV Cache (gamma/v7).

Two-tier page store under KVCache:
- Hot tier (GPU): mx.array fp16, bounded to kv_ssd_hot_pages.
- Cold tier (SSD): mmap over a preallocated per-session file.

Layout: Fixed-size pages of PAGE = 256 tokens.
Per page: [PAGE, n_kv, head_dim] in fp16.
"""
from __future__ import annotations

import os
import math
import mmap
import time
import shutil
import json
import hashlib
import tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx

PAGE = 256


# ─── gamma/v7 persistence (opt-in, ATF_KV_SSD_PERSIST=1) ──────────────────
# Stable keying + a small metadata sidecar so a paged KV session can be
# found and reused across process restarts, instead of always starting
# from an ephemeral time+pid name that nothing could ever look up again.
# See /Users/amgad/Desktop/Ai/Claude/atf-gamma-v7-ssd-cache-ui/notes.md for
# the correctness reasoning (GDN recurrent state cannot be trimmed to an
# arbitrary prefix -- reuse is only valid when the saved prefix's full
# length equals the new prompt's LCP with it, never a partial match).

def session_key(model_id: str, ids: list[int], prefix_len: int = 256) -> str:
    """Stable cache key: sha256(model_id + first `prefix_len` token ids)[:24].

    Deliberately keyed on a BOUNDED prefix (not the whole, ever-growing
    conversation) so every turn of the same conversation maps to the same
    key and reuses/overwrites the same on-disk session instead of forking
    a new one each turn.
    """
    h = hashlib.sha256()
    h.update((model_id or "unknown").encode("utf-8"))
    h.update(b"\x00")
    head = ids[:prefix_len]
    h.update(np.asarray(head, dtype=np.int64).tobytes())
    return h.hexdigest()[:24]


def manifest_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.meta.npz"


def save_manifest(cache_dir: Path, key: str, model_id: str,
                  fed_ids: list[int], gdn_states: dict) -> bool:
    """Persist fed_ids + per-block GDN recurrent state next to the paged
    KV files. The paged .kvmm files are already on disk (kv_ssd_path);
    this sidecar is what makes them SAFELY resumable -- without the GDN
    state, reusing the paged attention KV alone would still require
    replaying every prior token through the GDN blocks, which defeats the
    purpose and (worse) would silently desync GDN state from KV state.
    Best-effort: returns False (never raises) on any failure so a
    persistence bug can never break generation itself.
    """
    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = {"fed_ids": np.asarray(fed_ids, dtype=np.int64)}
        blocks_meta = []
        for b, st in (gdn_states or {}).items():
            if "state" in st:
                arrays[f"gdn_state_{b}"] = np.asarray(st["state"], dtype=np.float32)
            if "conv_buf" in st and st["conv_buf"].size > 0:
                arrays[f"gdn_conv_{b}"] = np.asarray(st["conv_buf"], dtype=np.float32)
            blocks_meta.append({
                "idx": int(b),
                "has_state": "state" in st,
                "conv_size": int(st["conv_buf"].size) if "conv_buf" in st else 0,
            })
        meta = {"model_id": model_id, "n_fed": len(fed_ids),
               "gdn_blocks": blocks_meta, "saved_at": time.time()}
        path = manifest_path(cache_dir, key)
        # np.savez_compressed silently appends ".npz" to the filename if it
        # doesn't already end with one -- naming the temp file "<key>.tmp"
        # made it actually write "<key>.tmp.npz" on disk, so the rename
        # below always failed (silently, into the except below) and this
        # never actually saved anything. Give the temp file a name that
        # ALREADY ends in ".npz" so numpy doesn't rewrite it out from under
        # the subsequent `.replace()`.
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays,
                            _meta=np.array([json.dumps(meta)], dtype=object))
        tmp.replace(path)
        return True
    except Exception as _exc:
        # Best-effort by design (a persistence bug must never break
        # generation), but silent failure here cost real debugging time
        # once already (the .npz-suffix bug below) -- always leave a
        # breadcrumb on stderr so a future "why didn't this save" doesn't
        # start from zero.
        import sys
        print(f"[atf.kvstore] save_manifest({key!r}) failed: {_exc!r}",
              file=sys.stderr)
        return False


def load_manifest(cache_dir: Path, key: str):
    """Returns {model_id, fed_ids, gdn_states} or None (missing/corrupt).
    Never raises -- a bad/partial sidecar just means "no reuse", not a
    crash."""
    try:
        path = manifest_path(cache_dir, key)
        if not path.exists():
            return None
        loaded = np.load(path, allow_pickle=True)
        meta_raw = loaded.get("_meta")
        if meta_raw is None:
            return None
        meta = json.loads(str(meta_raw[0]))
        fed_ids = loaded["fed_ids"].tolist()
        gdn_states = {}
        for bm in meta.get("gdn_blocks", []):
            idx = bm["idx"]
            st = {}
            if bm.get("conv_size", 0) > 0 and f"gdn_conv_{idx}" in loaded:
                st["conv_buf"] = mx.array(loaded[f"gdn_conv_{idx}"])
            if bm.get("has_state") and f"gdn_state_{idx}" in loaded:
                st["state"] = mx.array(loaded[f"gdn_state_{idx}"])
            if st:
                gdn_states[idx] = st
        return {"model_id": meta.get("model_id"), "fed_ids": fed_ids,
               "gdn_states": gdn_states}
    except Exception:
        return None


def find_reusable(cache_dir: Path, model_id: str, ids: list[int],
                  min_lcp: int = 64):
    """Look for a prior session (same key) whose saved fed_ids is an exact
    prefix of `ids` (LCP == the ENTIRE saved length -- see the module
    docstring above for why a partial match is not safe to reuse for GDN
    state). Returns {key, lcp, gdn_states} or None.
    """
    key = session_key(model_id, ids)
    man = load_manifest(cache_dir, key)
    if man is None or man.get("model_id") != model_id:
        return None
    old_ids = man["fed_ids"]
    if not old_ids:
        return None
    lcp = 0
    for a, b_ in zip(old_ids, ids):
        if a != b_:
            break
        lcp += 1
    if lcp != len(old_ids) or lcp < min_lcp or lcp >= len(ids):
        return None
    return {"key": key, "lcp": lcp, "gdn_states": man["gdn_states"]}


class PagedKV:
    """Page-oriented KV store with GPU hot tier and SSD mmap cold tier."""

    def __init__(self, *, n_kv: int, head_dim: int, capacity_pages: int,
                 hot_pages: int = 256, mmap_path: str | Path | None = None,
                 session_id: str | None = None, resume_n: int = 0,
                 persist: bool = False):
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.capacity_pages = max(1, capacity_pages)
        self.capacity = self.capacity_pages * PAGE
        self.hot_pages = max(1, min(hot_pages, self.capacity_pages))
        self.n = 0
        # gamma/v7 (ATF_KV_SSD_PERSIST): when True, close() leaves the
        # .kvmm file on disk instead of deleting it, so a later process can
        # resume it via `resume_n`. Default False keeps the original
        # ephemeral (session-scoped) behavior unchanged.
        self.persist = persist

        self.tokens_per_page = PAGE
        self.page_k_bytes = PAGE * n_kv * head_dim * 2  # fp16
        self.page_v_bytes = PAGE * n_kv * head_dim * 2
        self.page_total_bytes = self.page_k_bytes + self.page_v_bytes

        if mmap_path is None:
            sid = session_id or f"sess_{int(time.time() * 1000)}_{os.getpid()}_{id(self)}"
            kv_dir = Path(os.environ.get("ATF_KV_SSD_PATH", Path.home() / ".cache" / "atf" / "kvpages"))
            kv_dir.mkdir(parents=True, exist_ok=True)
            self.mmap_path = kv_dir / f"{sid}.kvmm"
        else:
            self.mmap_path = Path(mmap_path)
            self.mmap_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_size = self.capacity_pages * self.page_total_bytes
        self._f = open(self.mmap_path, "a+b")
        if self._f.tell() < self._file_size:
            self._f.truncate(self._file_size)
            self._f.flush()

        self._mm = mmap.mmap(self._f.fileno(), self._file_size, access=mmap.ACCESS_WRITE)

        # page_idx -> {"k": mx.array [PAGE, n_kv, head_dim], "v": mx.array, "lru_ts": float}
        self._hot: dict[int, dict] = {}
        self._cold_pages: set[int] = set()
        self._access_counter: int = 0

        # gamma/v7 (ATF_KV_SSD_PERSIST): resuming an existing .kvmm file.
        # The mmap already holds valid bytes for tokens [0, resume_n) from a
        # prior process (that's the whole point of not deleting it on
        # close()); mark those pages "cold" so _get_or_create_page() reads
        # them back from the mmap on first touch instead of assuming they're
        # unwritten and zero-initializing them (which would silently corrupt
        # every full-attention block's KV for the resumed prefix).
        if resume_n > 0:
            if resume_n > self.capacity:
                raise ValueError(
                    f"resume_n ({resume_n}) exceeds this session's capacity "
                    f"({self.capacity} tokens) -- refusing to resume into an "
                    f"undersized cache file")
            self.n = resume_n
            n_resumed_pages = math.ceil(resume_n / PAGE)
            self._cold_pages = set(range(n_resumed_pages))

    def _now(self) -> int:
        self._access_counter += 1
        return self._access_counter

    def _evict_one(self, avoid_page: int | None = None):
        """Evict least-recently-touched page from hot tier to cold tier mmap."""
        if len(self._hot) < self.hot_pages:
            return
        candidates = [p for p in self._hot.keys() if p != avoid_page]
        if not candidates:
            return
        # Prefer preserving page 0 (system prompt / initial context)
        if len(candidates) > 1 and 0 in candidates:
            candidates.remove(0)
        victim = min(candidates, key=lambda p: self._hot[p]["lru_ts"])

        # Write victim page to mmap
        k_data = np.array(self._hot[victim]["k"])
        v_data = np.array(self._hot[victim]["v"])
        base = victim * self.page_total_bytes
        k_view = np.frombuffer(self._mm, dtype=np.float16,
                               count=PAGE * self.n_kv * self.head_dim, offset=base)
        np.copyto(k_view, k_data.reshape(-1))
        v_view = np.frombuffer(self._mm, dtype=np.float16,
                               count=PAGE * self.n_kv * self.head_dim,
                               offset=base + self.page_k_bytes)
        np.copyto(v_view, v_data.reshape(-1))

        del self._hot[victim]
        self._cold_pages.add(victim)

    def _get_or_create_page(self, page_idx: int) -> dict:
        """Ensure page_idx is hot in GPU memory."""
        if page_idx in self._hot:
            self._hot[page_idx]["lru_ts"] = self._now()
            return self._hot[page_idx]

        self._evict_one(avoid_page=page_idx)

        if page_idx in self._cold_pages:
            # Read back from mmap
            base = page_idx * self.page_total_bytes
            k_raw = np.frombuffer(self._mm, dtype=np.float16,
                                  count=PAGE * self.n_kv * self.head_dim, offset=base).copy()
            v_raw = np.frombuffer(self._mm, dtype=np.float16,
                                  count=PAGE * self.n_kv * self.head_dim,
                                  offset=base + self.page_k_bytes).copy()
            k_page = mx.array(k_raw.reshape(PAGE, self.n_kv, self.head_dim))
            v_page = mx.array(v_raw.reshape(PAGE, self.n_kv, self.head_dim))
            self._cold_pages.remove(page_idx)
        else:
            k_page = mx.zeros((PAGE, self.n_kv, self.head_dim), dtype=mx.float16)
            v_page = mx.zeros((PAGE, self.n_kv, self.head_dim), dtype=mx.float16)

        entry = {"k": k_page, "v": v_page, "lru_ts": self._now()}
        self._hot[page_idx] = entry
        return entry

    def append(self, k: mx.array, v: mx.array) -> None:
        """Append k, v tensors of shape [T, n_kv, head_dim] into the paged store."""
        T = k.shape[0]
        if self.n + T > self.capacity:
            raise RuntimeError(
                f"KV cache overflow: {self.n}+{T} > {self.capacity} tokens "
                f"(raise GenConfig.max_context)")

        k = k.astype(mx.float16)
        v = v.astype(mx.float16)
        t_done = 0
        while t_done < T:
            cur_pos = self.n
            p_idx = cur_pos // PAGE
            p_off = cur_pos % PAGE
            chunk_len = min(T - t_done, PAGE - p_off)

            page = self._get_or_create_page(p_idx)
            page["k"][p_off:p_off + chunk_len] = k[t_done:t_done + chunk_len]
            page["v"][p_off:p_off + chunk_len] = v[t_done:t_done + chunk_len]
            page["lru_ts"] = self._now()

            self.n += chunk_len
            t_done += chunk_len

    def keys(self) -> mx.array:
        """Return full concatenated keys up to self.n in fp32."""
        if self.n == 0:
            return mx.zeros((0, self.n_kv, self.head_dim), dtype=mx.float32)

        num_pages = math.ceil(self.n / PAGE)
        # Fast path: single active page in hot tier
        if num_pages == 1 and 0 in self._hot:
            return self._hot[0]["k"][:self.n].astype(mx.float32)

        views = []
        for p in range(num_pages):
            n_in_p = min(self.n - p * PAGE, PAGE)
            if p in self._hot:
                self._hot[p]["lru_ts"] = self._now()
                views.append(self._hot[p]["k"][:n_in_p])
            else:
                # Read slice from cold mmap
                base = p * self.page_total_bytes
                k_raw = np.frombuffer(self._mm, dtype=np.float16,
                                      count=PAGE * self.n_kv * self.head_dim, offset=base)
                k_slice = k_raw[:n_in_p * self.n_kv * self.head_dim].copy()
                views.append(mx.array(k_slice.reshape(n_in_p, self.n_kv, self.head_dim)))

        return mx.concatenate(views, axis=0).astype(mx.float32)

    def values(self) -> mx.array:
        """Return full concatenated values up to self.n in fp32."""
        if self.n == 0:
            return mx.zeros((0, self.n_kv, self.head_dim), dtype=mx.float32)

        num_pages = math.ceil(self.n / PAGE)
        if num_pages == 1 and 0 in self._hot:
            return self._hot[0]["v"][:self.n].astype(mx.float32)

        views = []
        for p in range(num_pages):
            n_in_p = min(self.n - p * PAGE, PAGE)
            if p in self._hot:
                self._hot[p]["lru_ts"] = self._now()
                views.append(self._hot[p]["v"][:n_in_p])
            else:
                base = p * self.page_total_bytes
                v_raw = np.frombuffer(self._mm, dtype=np.float16,
                                      count=PAGE * self.n_kv * self.head_dim,
                                      offset=base + self.page_k_bytes)
                v_slice = v_raw[:n_in_p * self.n_kv * self.head_dim].copy()
                views.append(mx.array(v_slice.reshape(n_in_p, self.n_kv, self.head_dim)))

        return mx.concatenate(views, axis=0).astype(mx.float32)

    def __len__(self) -> int:
        return self.n

    def close(self):
        # gamma/v7: flush any hot pages to the mmap BEFORE closing it when
        # persisting, so a page that never got evicted during this session
        # (e.g. a short conversation that never exceeded hot_pages) is still
        # on disk for the next process to resume -- previously only
        # _evict_one() wrote pages back, so a persisted-but-never-evicted
        # page would resume as all-zeros and silently corrupt the reused KV.
        if self.persist and hasattr(self, "_mm") and self._mm is not None:
            for page_idx in list(self._hot.keys()):
                try:
                    self._evict_one_for_close(page_idx)
                except Exception:
                    pass
        if hasattr(self, "_mm") and self._mm is not None:
            try:
                self._mm.flush()
            except Exception:
                pass
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if hasattr(self, "_f") and self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None
        if not self.persist and hasattr(self, "mmap_path") and self.mmap_path.exists():
            try:
                self.mmap_path.unlink()
            except Exception:
                pass

    def _evict_one_for_close(self, page_idx: int) -> None:
        """Write one still-hot page to the mmap unconditionally (close()-time
        flush). Mirrors _evict_one()'s write path but skips the hot_pages
        threshold check and the LRU victim-selection -- at close() every
        remaining hot page must be written, not just the least-recent one."""
        if page_idx not in self._hot:
            return
        k_data = np.array(self._hot[page_idx]["k"])
        v_data = np.array(self._hot[page_idx]["v"])
        base = page_idx * self.page_total_bytes
        k_view = np.frombuffer(self._mm, dtype=np.float16,
                               count=PAGE * self.n_kv * self.head_dim, offset=base)
        np.copyto(k_view, k_data.reshape(-1))
        v_view = np.frombuffer(self._mm, dtype=np.float16,
                               count=PAGE * self.n_kv * self.head_dim,
                               offset=base + self.page_k_bytes)
        np.copyto(v_view, v_data.reshape(-1))

    def __del__(self):
        self.close()
