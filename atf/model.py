"""ATF model loading with MLX backend.

Memory model:
  - ALL weights resident as INT8 on GPU (+ small f32 scales)
  - Dequantization happens on the fly: q.float() * scales
  - LRU cache for hot dequantized f32 tensors
"""
from __future__ import annotations

import mmap
import os
import re
import struct
import time
import resource
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx

from .gguf_metal import gguf_matmul, gguf_gather, supported as gguf_supported
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from .format import Header, LayerManifest, ExpertEntry, LodLevel, HEADER_SIZE

console = Console()

LRU_CAP_BYTES = 3 * (1 << 30)


def gpu_mem_line() -> str:
    """One-line snapshot of MLX/Metal GPU memory: active (in use right now),
    peak (high-water mark since last reset_peak_memory), and cache (freed
    buffers MLX is holding for reuse rather than returning to the OS)."""
    active = mx.get_active_memory() / (1 << 30)
    peak = mx.get_peak_memory() / (1 << 30)
    cache = mx.get_cache_memory() / (1 << 30)
    return f"active {active:.2f} GB | peak {peak:.2f} GB | cache {cache:.2f} GB"


def host_mem_line() -> str:
    """Process resident set size (host RAM actually held by this process),
    via getrusage -- no extra dependency (psutil isn't installed).
    ru_maxrss is bytes on macOS, KB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return f"host RSS (peak) {rss / (1 << 30):.2f} GB"


def _parse_int8_blob(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = struct.unpack_from("<II", data, 0)
    off = 8
    scales = np.frombuffer(data, dtype=np.float32, count=rows, offset=off).copy().reshape(rows, 1)
    q = np.frombuffer(data, dtype=np.int8, count=rows * cols, offset=off + rows * 4).copy().reshape(rows, cols)
    return q, scales


def _parse_int8_blob_at(mm, base: int):
    """v8: zero-copy parse of an INT8 blob directly out of an mmap at absolute
    offset `base`. Returns (q, scales, blob_size). np.frombuffer over the mmap
    yields views -- no intermediate host buffer, no .copy()."""
    rows, cols = struct.unpack_from("<II", mm, base)
    off = base + 8
    scales = np.frombuffer(mm, dtype=np.float32, count=rows, offset=off).reshape(rows, 1)
    q = np.frombuffer(mm, dtype=np.int8, count=rows * cols, offset=off + rows * 4).reshape(rows, cols)
    return q, scales, 8 + rows * 4 + rows * cols


def _parse_fp4_blob_at(mm, base: int):
    """v8.1: parse an FP4 (E2M1 nibble, f32 group-64 scale) blob out of an mmap.
    Returns (f32 weights, blob_size); caller stores weights with unit scales."""
    from .quantize import dequantize_fp4
    rows, cols = struct.unpack_from("<II", mm, base)
    n_groups = rows * (cols // 64)
    off = base + 8 + n_groups * 4
    packed = np.frombuffer(mm, dtype=np.uint8,
                           count=rows * ((cols + 1) // 2), offset=off)
    w = dequantize_fp4(bytes(mm[base:off + packed.nbytes]))
    return (w.reshape(rows, cols), None,
            8 + n_groups * 4 + packed.nbytes)


def _read_header(f) -> Header:
    return Header.unpack(f.read(HEADER_SIZE))


def _read_manifest(f, offset: int) -> LayerManifest:
    f.seek(offset)
    nblocks, nexp_per_blk = struct.unpack("<HH", f.read(4))
    (ndomains,) = struct.unpack("<H", f.read(2))
    domain_names = []
    for _ in range(ndomains):
        (slen,) = struct.unpack("<H", f.read(2))
        domain_names.append(f.read(slen).decode("utf-8"))
    (nsubs,) = struct.unpack("<H", f.read(2))
    sub_domain_names = []
    for _ in range(nsubs):
        (slen,) = struct.unpack("<H", f.read(2))
        sub_domain_names.append(f.read(slen).decode("utf-8"))
    (nexperts,) = struct.unpack("<I", f.read(4))
    raw = f.read(nexperts * struct.calcsize("<QBBBBBQQQI"))
    experts = []
    for i in range(nexperts):
        (eid, bidx, did, sid, spec, lod, woff, wsize, coff, cdim) = \
            struct.unpack_from("<QBBBBBQQQI", raw, i * struct.calcsize("<QBBBBBQQQI"))
        experts.append(ExpertEntry(
            expert_id=eid, block_index=bidx, domain_id=did, sub_domain_id=sid,
            specificity=spec, lod_level=LodLevel(lod),
            weight_offset=woff, weight_size=wsize,
            centroid_offset=coff, centroid_dim=cdim,
        ))
    return LayerManifest(
        num_blocks=nblocks, num_experts_per_block=nexp_per_blk,
        experts=experts, domain_names=domain_names, sub_domain_names=sub_domain_names,
    )


class AtfModel:
    def __init__(self, header: Header, manifest: LayerManifest,
                 quant: dict, router, shapes: dict,
                 load_time: float, ram_bytes: int,
                 lru_cap_bytes: int = LRU_CAP_BYTES,
                 prebuilt_ffn: dict | None = None,
                 raw: dict | None = None):
        self.header = header
        # v8 format-v2: {(block:int, proj:str): (wq, scales, bits)} already in
        # MLX quantized_matmul layout -- merge_ffn uses these with zero work.
        self.prebuilt_ffn: dict = prebuilt_ffn or {}
        # v9 raw-GGUF mode: weights kept in the ORIGINAL GGUF quantization,
        # resident on the GPU as packed bytes; matmuls run through fused
        # dequant kernels (atf.gguf_metal). No int8/f32 conversion anywhere.
        self.raw = raw or None
        if raw:
            self._raw_gpu = raw["gpu"]          # name -> mx.array uint8
            self._raw_shapes = raw["shapes"]    # name -> logical [in, out]
            self._raw_dtypes = raw["dtypes"]    # name -> GGUF dtype code
            self._raw_path = raw.get("path")
            self._raw_off = raw.get("offs", {})
            self._raw_len = raw.get("lens", {})
            # v16: prefill GEMM fallback -- dequantized weight slices for
            # batched native matmul (see _qmm_gemm). The custom GGUF kernels
            # are GEMV-shaped (per-row dequant, no reuse), which made prefill
            # run at decode-class speed (~20 tok/s on the 9B) and stalled
            # long-prompt API requests.
            self._gemm_cache: "OrderedDict[tuple, Any]" = OrderedDict()
            self._gemm_cache_bytes = 0
            # name -> row offset of that tensor inside its shared pre-packed
            # fusion buffer (loader-built); members share one mx.array.
            self._raw_pack = raw.get("pack", {})
        self.manifest = manifest
        self.quant = quant            # name -> (q int8 mx.array [r,c], scales f32 mx.array [r,1])
        self.shapes = shapes          # name -> original shape
        self.router = router          # mx.array [num_blocks, hidden, num_experts] f32
        self.vocab_size = header.vocab_size
        self.load_time = load_time
        self.ram_bytes = ram_bytes
        self.device = "mlx"
        self._lru: "OrderedDict[str, Any]" = OrderedDict()
        self._lru_bytes = 0
        self._lru_cap_bytes = lru_cap_bytes
        self._qcache: dict[str, tuple] = {}
        # v10 perf fix: _raw_base() does a regex match to resolve FFN/dense
        # slice offsets; in raw-GGUF decode this runs ~350x/token (every
        # projection, every block) for names that never change meaning
        # across calls. Memoize per name so the regex only ever runs once.
        self._raw_base_cache: dict[str, tuple | None] = {}
        # v11 perf fix: cache for fused multi-tensor byte buffers (see
        # qmm_fuse below) -- built once per unique tensor-name-group, then
        # reused every subsequent call/token.
        self._fuse_cache: dict[tuple, Any] = {}
        # Diagnostics: how often w() actually has to re-dequantize vs reuse
        # a cached tensor. Cheap to track (no mx.eval(), doesn't touch the
        # lazy graph) -- use this to tell whether the cache is helping or
        # is just too small for the working set (thrashing every call).
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_bytes(self, value) -> int:
        return value.nbytes

    def cache_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total else 0.0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_bytes": self._lru_bytes,
            "cache_cap_bytes": self._lru_cap_bytes,
            "cached_tensors": len(self._lru),
        }

    def w(self, name: str):
        """Dequantized f32 MLX array for `name` (original shape), served from LRU."""
        if self.raw and name not in self.quant:
            return self._w_raw(name)
        if self.raw:
            return self._w_raw(name)
        hit = self._lru.get(name)
        if hit is not None:
            self._lru.move_to_end(name)
            self.cache_hits += 1
            return hit
        self.cache_misses += 1
        q, scales = self.quant[name]
        w = q.astype(mx.float32) * scales
        shape = self.shapes.get(name)
        if shape is not None and w.shape != tuple(shape):
            w = w.reshape(shape)
        self._lru[name] = w
        self._lru_bytes += w.nbytes
        while self._lru_bytes > self._lru_cap_bytes and len(self._lru) > 1:
            _, evicted = self._lru.popitem(last=False)
            self._lru_bytes -= evicted.nbytes
        return w

    def _w_raw(self, name: str):
        """CPU decode path for small / unsupported raw tensors (norms, biases,
        IQ1_M ...). Cached in the LRU.

        Refuses 3-D MoE tensors (`blk.*.ffn_*_exps.weight`): those must go
        through `_moe_fill_proj_raw` -> `_dequant_expert_slice` so we
        dequant only the requested expert, not the entire n_exp-wide
        fp32 record (which would blow past the LRU cap and thrash)."""
        if name.startswith("blk.") and ".ffn_" in name and name.endswith("_exps.weight"):
            raise ValueError(
                f"_w_raw called on 3-D MoE tensor {name!r}; use "
                f"moe_expert(b, e) -> _moe_fill_proj_raw -> _dequant_expert_slice")
        hit = self._lru.get(name)
        if hit is not None:
            self.cache_hits += 1
            self._lru.move_to_end(name)
            return hit
        self.cache_misses += 1
        from .gguf_io import dequantize as _dq
        with open(self._raw_path, "rb") as f:
            f.seek(self._raw_off[name])
            data = f.read(self._raw_len[name])
        arr = _dq(data, self._raw_dtypes[name], self._raw_shapes[name])
        w = mx.array(np.ascontiguousarray(arr, dtype=np.float32))
        self._lru[name] = w
        self._lru_bytes += w.nbytes
        return w

    def w_raw(self, name: str) -> tuple:
        """Raw (q, scales) for direct access without dequant."""
        return self.quant[name]

    def qmm(self, name: str, x, bits: int = 4, group_size: int = 64):
        if self.raw:
            base = self._raw_base(name)
            if base is None:
                return self.w(name) @ x.T
            rec, n_base, k_base, n_out, k_len = base
            shape = self._raw_shapes[rec]
            dt = self._raw_dtypes[rec]
            k_full = int(shape[0]); n_full = int(shape[1])
            if n_out is None: n_out = n_full
            if k_len is None: k_len = k_full - k_base
            if not gguf_supported(dt):
                # CPU decode fallback (e.g. the single IQ1_M tensor)
                W = self._w_raw(rec)          # logical [in, out]
                W = W[k_base:k_base + k_len, n_base:n_base + n_out]
                return x @ W
            if (self._gemm_min_rows() > 0 and x.ndim == 2
                    and x.shape[0] >= self._gemm_min_rows()):
                # v16.1: real batch (prefill) -> gather-once + native GEMM.
                # n_out/k_len arrive FILLED with full-range defaults (see
                # above); _qmm_gemm narrows slices by comparing them against
                # n_full/k_full. n_base is a packed-buffer ROW offset and is
                # only ever used as a buffer row index (gather ids).
                return self._qmm_gemm(x, rec, n_base, k_base, n_out, k_len,
                                      k_full=k_full, n_full=n_full)
            return gguf_matmul(x, self._raw_gpu[rec], n_out, k_full, dt,
                               n_base=n_base, k_base=k_base, k_len=k_len)
        # Non-raw path: x @ W via an MLX-native quantized matmul.
        # First call converts the stored INT8 tensor to a group-wise MLX
        # quantized form (packed along the out dim), caches it permanently,
        # and frees the INT8 original plus any f32 LRU entry.
        ent = self._qcache.get(name)
        if ent is None:
            w = self.w(name)
            wq, s, b = mx.quantize(w.T, group_size=group_size, bits=bits)
            ent = (wq, s, b)
            self._qcache[name] = ent
            self.quant.pop(name, None)
            self.shapes.pop(name, None)
            old = self._lru.pop(name, None)
            if old is not None:
                self._lru_bytes -= old.nbytes
        wq, s, b = ent
        return mx.quantized_matmul(x, wq, s, b, transpose=True)

    @staticmethod
    def _gemm_min_rows() -> int:
        """Rows above which the native-GEMM prefill path wins over the
        per-row GGUF kernels.

        The identity-matrix variant that wedded a Metal command buffer
        (machine-hang class) is gone -- this path only launches bounded
        gather kernels + native GEMM, validated bit-exact vs the decode
        kernels (see update.md v16 work log). ON by default (>=8 rows);
        set ATF_PREFILL_GEMM_MIN=0 to force the per-row GGUF kernels."""
        try:
            n = int(os.environ.get("ATF_PREFILL_GEMM_MIN", "8"))
        except ValueError:
            return 8
        return max(0, n)

    def _qmm_gemm(self, x, rec, n_base, k_base, n_out, k_len,
                  k_full: int = 0, n_full: int = 0):
        """Batched matmul via a dequantized weight slice + native matmul.

        The slice is cached on the GPU (LRU, ATF_GEMM_CACHE_GB cap) so
        repeated prefill passes across requests do not re-dequantize.
        Numerics: same exact f32 dequantized weights as the custom kernels,
        accumulated by MLX's tiled f32 GEMM.

        Slice resolution (mirrors what gguf_matmul computes in-buffer):
          - whole record   (n_out/k_len None)      -> full [in, out]
          - .eN down       (K-slice, all columns)  -> W[k_base:k_base+k_len, :]
          - .eN gate/up    (N-chunk, all rows)     -> W[:, col:col+n_out] where
             col strips the packed-buffer row offset from n_base.
        """
        # v16.1: materialize the slice with gguf_gather -- a dedicated
        # dequantize-selected-rows -> f32 kernel where each packed byte is
        # read EXACTLY once (reads scale with output size, NOT with batch
        # length). The earlier variant fed an identity-matrix selector
        # through gguf_matmul; the GEMV kernel re-read the whole packed
        # buffer per selector row (terabytes for a real chunk -> multi-
        # minute command buffers -> machine-hang class). Never again.
        if not k_full:
            k_full = int(self._raw_shapes[rec][0])
        if not n_full:
            n_full = int(self._raw_shapes[rec][1])
        pack_off = int(self._raw_pack.get(rec, 0))
        kb, kl = int(k_base), int(k_len) if k_len is not None else None
        no = int(n_out) if n_out is not None else None
        # rows: K-slice when k_base/k_len narrow the record's K dim
        # (.eN down experts).
        row_lo, row_hi = (kb, kb + kl) \
            if (kl is not None and (kb > 0 or kl != k_full)) else (0, k_full)
        # cols: gather ids are BUFFER ROW indices. n_base is the packed-buffer
        # row offset (whole record -> pack_off; .eN gate/up -> pack_off +
        # e*n_chunk), so ids ALWAYS start at n_base. Zeroing it for whole-
        # record calls silently shifted every packed fusion member (qkv,
        # gate/up) onto the wrong rows -- numerics looked plausible on one
        # probe but generated garbage end-to-end.
        col_lo = int(n_base)
        col_hi = int(n_base) + (no if no is not None else n_full)
        key = (rec, row_lo, row_hi, col_lo, col_hi)
        Wt = self._gemm_cache.get(key)
        if Wt is None:
            ids = mx.arange(col_lo, col_hi, dtype=mx.uint32)
            # gather returns [n_sel, k_sel], [j, i] = W[row_lo+i, col_lo+j].
            # Store/compute f16: exact f32 dequant -> nearest-f16 weight
            # (relative error ~5e-4, same order as GEMM accumulation noise),
            # but hits the GPU's fast f16 GEMM path instead of the very
            # slow f32 one -- measured ~3x on the 9B prefill.
            Wt = gguf_gather(self._raw_gpu[rec], ids, k_full,
                             self._raw_dtypes[rec],
                             k_base=row_lo, k_len=row_hi - row_lo)
            Wt = Wt.astype(mx.float16)
            self._gemm_cache[key] = Wt
            self._gemm_cache_bytes += int(Wt.nbytes)
            try:
                cap = int(float(os.environ.get("ATF_GEMM_CACHE_GB", "2")) * (1 << 30))
            except ValueError:
                cap = 2 << 30
            while self._gemm_cache_bytes > cap and len(self._gemm_cache) > 1:
                _, evicted = self._gemm_cache.popitem(last=False)
                self._gemm_cache_bytes -= int(evicted.nbytes)
        y = x.astype(mx.float16) @ Wt.T
        return y.astype(mx.float32)

    def qmm_fuse(self, names: list[str], x):
        """Fuse several raw-GGUF matmuls that share the same input `x` into as
        few GPU kernel dispatches as possible.

        v11 perf fix: profiling showed decode cost is spread almost evenly
        across every block (~11ms/block regardless of block type), consistent
        with per-dispatch overhead (~1.3ms) dominating over actual compute --
        each GDN block alone makes 8 separate weight-matmul calls (qkv, gate,
        alpha, beta, ffn gate/up/down) that all could be fewer dispatches.

        GGUF quant blocks are contiguous per-output-row bytes, and the kernel
        addresses a row as `w + n*ROWBYTES`. Tensors that share the same
        quant dtype (-> same ROWBYTES) and the same input (K) dimension are
        packed by the LOADER into one shared buffer at upload time (zero
        post-load copies, zero extra residency) and computed in a single
        dispatch -- numerically identical to calling them separately, just
        fewer GPU launches. The earlier v11 attempt concatenated buffers at
        first use instead, duplicating multi-GB weights past the wired limit
        and page-thrashing decode; that path is gone.

        SAFETY: only tensors resolving to a whole, supported-dtype record
        with matching (dtype, k_full) get grouped; anything else (mismatched
        dtype/K, an expert slice, an unsupported dtype, a non-raw model)
        falls straight back to the existing per-tensor qmm() call -- so a
        mismatch can only cost us the speedup, never produce wrong output.
        """
        if not self.raw:
            return [self.qmm(n, x) for n in names]
        # Escape hatch: set ATF_NO_FUSE=1 to fall back to per-tensor dispatches.
        if os.environ.get("ATF_NO_FUSE") == "1":
            return [self.qmm(n, x) for n in names]
        # v16: batched input (prefill) -> per-tensor GEMM fallback instead of
        # the fused GEMV kernels; see _qmm_gemm for rationale. Disabled unless
        # ATF_PREFILL_GEMM_MIN > 0 (see _qmm_gemm safeguard note).
        if (self._gemm_min_rows() > 0 and x.ndim == 2
                and x.shape[0] >= self._gemm_min_rows()):
            return [self.qmm(n, x) for n in names]
        # v11 memory-neutral fusion: the loader pre-packed each planned group
        # into ONE shared GPU buffer (see _load_atf_raw); every member record
        # points at that shared array with its row offset (_raw_pack). So a
        # fused call is just one gguf_matmul over the span the members cover,
        # sliced back apart -- zero copies at decode time.
        resolved: list[tuple | None] = []
        for n in names:
            base = self._raw_base(n)
            if base is None:
                resolved.append(None)
                continue
            rec, n_base, k_base, n_out, k_len = base
            # fuse whole records only: no K slicing, and the member must be
            # part of a pre-packed group (guaranteed same dtype + input dim)
            if k_base or n_out is not None or k_len is not None \
                    or rec not in self._raw_pack:
                resolved.append(None)
                continue
            dt = self._raw_dtypes[rec]
            if not gguf_supported(dt):
                resolved.append(None)
                continue
            k_full = int(self._raw_shapes[rec][0])
            n_full = int(self._raw_shapes[rec][1])
            resolved.append((rec, n_base, dt, k_full, n_full))
        outs: list = [None] * len(names)
        groups: dict[int, tuple[Any, list[int]]] = {}
        for i, r in enumerate(resolved):
            if r is None:
                outs[i] = self.qmm(names[i], x)
                continue
            arr = self._raw_gpu[r[0]]
            ent = groups.get(id(arr))
            if ent is None:
                ent = (arr, [])
                groups[id(arr)] = ent
            ent[1].append(i)
        for arr, idxs in groups.values():
            if len(idxs) == 1:              # lone member: plain path
                i = idxs[0]
                outs[i] = self.qmm(names[i], x)
                continue
            rec0, start0, dt, k_full, _ = resolved[idxs[0]]
            start = min(resolved[i][1] for i in idxs)
            end = max(resolved[i][1] + resolved[i][4] for i in idxs)
            out = gguf_matmul(x, arr, end - start, k_full, dt,
                              n_base=start, k_base=0, k_len=k_full)
            for i in idxs:
                o = resolved[i][1] - start
                outs[i] = out[..., o:o + resolved[i][4]]
        return outs

    def mm(self, name: str, x):
        if self.raw and name in self._raw_gpu:
            dt = self._raw_dtypes[name]
            if not gguf_supported(dt):
                # dtype without a Metal kernel (e.g. F32 LM head):
                # CPU-decoded f32 matmul via the LRU.
                return x @ self.w(name)
            shape = self._raw_shapes[name]
            return gguf_matmul(x, self._raw_gpu[name], int(shape[1]),
                               int(shape[0]), dt)
        """Dense bf16 matmul for tensors too precision-sensitive to quantize
        (currently the LM head). Converts once, frees the INT8 original."""
        w = self._qcache.get("bf16:" + name)
        if w is None:
            w = self.w(name).astype(mx.bfloat16)
            self._qcache["bf16:" + name] = w
            self.quant.pop(name, None)
            self.shapes.pop(name, None)
            old = self._lru.pop(name, None)
            if old is not None:
                self._lru_bytes -= old.nbytes
        return (x.astype(mx.bfloat16) @ w).astype(mx.float32)

    def _embed_fallback_ok(self) -> bool:
        """Raw mode: gather kernel supports this embedding dtype?"""
        dt = self._raw_dtypes.get("token_embd.weight")
        return dt is not None and gguf_supported(dt)

    def embed(self, token_id: int):
        if self.raw:
            if not self._embed_fallback_ok():
                # dtype the Metal gather kernel doesn't handle (e.g. F32
                # embeddings): CPU-decode path via w().
                return self.w("token_embd.weight")[:, token_id]
            shape = self._raw_shapes["token_embd.weight"]
            return gguf_gather(self._raw_gpu["token_embd.weight"],
                               mx.array([token_id]), int(shape[0]),
                               self._raw_dtypes["token_embd.weight"])[0]
        """Single embedding vector: [hidden] f32."""
        q, scales = self.quant["token_embd.weight"]
        return (q[:, token_id].astype(mx.float32) * scales[:, 0])

    def embed_batch(self, token_ids: list[int]):
        if self.raw:
            if not self._embed_fallback_ok():
                W = self.w("token_embd.weight")          # [hidden, vocab]
                return W[:, mx.array(list(token_ids))].T  # [T, hidden]
            shape = self._raw_shapes["token_embd.weight"]
            return gguf_gather(self._raw_gpu["token_embd.weight"],
                               mx.array(list(token_ids)), int(shape[0]),
                               self._raw_dtypes["token_embd.weight"])
        """Batched embedding lookup: [T, hidden] f32."""
        q, scales = self.quant["token_embd.weight"]
        idx = mx.array(token_ids)
        return (q[:, idx].astype(mx.float32) * scales[:, 0:1]).transpose(1, 0)

    def logits(self, x):
        """LM head: x [hidden] @ output_weight [hidden, vocab] -> [vocab].

        v8: delegates to mm() so both paths share one bf16-cached weight
        (the old version read self.quant directly and broke after any
         forward pass had converted + freed the INT8 original)."""
        return self.mm("output.weight", x)

    def expert_weights(self, b: int, e: int) -> tuple:
        return (self.w(f"blk.{b}.ffn_gate.weight.e{e}"),
                self.w(f"blk.{b}.ffn_up.weight.e{e}"),
                self.w(f"blk.{b}.ffn_down.weight.e{e}"))

    # ── true-MoE support (v19.1) ─────────────────────────────────────────
    def moe_expert(self, b: int, e: int) -> tuple:
        """(gate, up, down) f32 MLX arrays, each [in, out], for routed
        expert `e` of block `b`.

        Legacy-converted files store the slices as plain dense records
        (`blk.b.moe_{proj}.weight.e{e}`) resolved through w(). Raw-GGUF
        files keep the original 3-D `ffn_*_exps` tensors; we dequant and
        cache only the requested expert's per-proj slice on a miss (NOT
        the whole record) so the LRU doesn't churn on full-tensor spills.
        """
        _moe_debug = os.environ.get("ATF_MOE_DEBUG") == "1"
        out = []
        for proj in ("gate", "up", "down"):
            name = f"blk.{b}.moe_{proj}.weight.e{e}"
            hit = self._lru.get(name)
            if hit is not None:
                self._lru.move_to_end(name)
                self.cache_hits += 1
                out.append(hit)
                continue
            if _moe_debug:
                import sys as _sys
                print(f"[moe-debug] cache MISS for {name} (block {b} expert {e} "
                      f"{proj}) -- dequantizing one expert slice only",
                      file=_sys.stderr, flush=True)
            if self.raw:
                self._moe_fill_proj_raw(b, proj, e)
                hit = self._lru.get(name)
            if hit is None:
                # legacy quant path (w() also fills the LRU)
                hit = self.w(name)
            out.append(hit)
        return tuple(out)

    def moe_router(self, b: int):
        """Router weight [hidden, n_exp] for block b."""
        return self.w(f"blk.{b}.ffn_gate_inp.weight")

    def _dequant_expert_slice(self, src: str, e: int) -> "mx.array":
        """Read and dequantize a SINGLE expert `e` of the 3-D MoE record
        `src` (e.g. `blk.3.ffn_gate_exps.weight`).

        On-disk shape3 (verified against the real 35B-A3B ATF at
        /Volumes/SSD5/Ai/atf/models/v4/Qwen-AgentWorld-35B-A3B-UD-IQ2_M.atf):
            shape3 = (hidden, ffe, n_exp)             for gate/up
            shape3 = (ffe,    hidden, n_exp)          for down
        i.e. `n_exp` is the LAST axis on disk. Each per-expert 2-D slice
        is therefore shape3[0..1] = (hidden, ffe) for gate/up or
        (ffe, hidden) for down -- which is already the engine-facing
        [in, out] orientation (Xe @ Wg = [T, hidden] @ [hidden, ffe] for
        gate, [T, ffe] @ [ffe, hidden] for down). No transpose needed.

        The record is a flat concatenation of `n_exp` identical ggml-quant
        blocks, so we seek to e * (rec_len // n_exp) and feed that byte
        window to the existing per-type dequantizer in gguf_io, which
        returns an (in, out) array.

        Note: tmp/moe_test/build_synth.py writes the OPPOSITE layout
        (n_exp FIRST) and is wrong vs real GGUF; only the ATF produced by
        the project's convert.py matches this code path.
        """
        from .gguf_io import dequantize as _dq
        dtype = self._raw_dtypes[src]
        shape3 = self._raw_shapes[src]
        assert len(shape3) == 3, f"expected 3-D MoE tensor, got shape {shape3} for {src}"
        n_exp = int(shape3[2])                          # n_exp is LAST axis on disk
        if not (0 <= e < n_exp):
            raise IndexError(f"expert {e} out of range for {src} (n_exp={n_exp})")
        # Per-expert 2-D slice on disk is already (shape3[0], shape3[1])
        # = (in, out) for the engine. No transpose.
        per_expert_2d = (int(shape3[0]), int(shape3[1]))
        rec_len = int(self._raw_len[src])
        per_expert_bytes = rec_len // n_exp
        if rec_len != per_expert_bytes * n_exp:
            raise ValueError(
                f"MoE record {src} length {rec_len} not divisible by "
                f"n_exp={n_exp} (per_expert={per_expert_bytes})")
        off = int(self._raw_off[src]) + e * per_expert_bytes
        with open(self._raw_path, "rb") as f:
            f.seek(off)
            data = f.read(per_expert_bytes)
        if os.environ.get("ATF_MOE_DEBUG") == "1":
            console.print(
                f"[moe-debug] dequant_expert_slice {src} e={e}/{n_exp} "
                f"shape3={tuple(shape3)} per_expert_2d={per_expert_2d} "
                f"bytes={per_expert_bytes} dtype={dtype}")
        arr = _dq(data, dtype, per_expert_2d)
        if os.environ.get("ATF_MOE_DEBUG") == "1":
            ok = tuple(arr.shape) == per_expert_2d
            console.print(
                f"[moe-debug]   -> arr.shape={arr.shape} "
                f"expected={per_expert_2d} OK={ok}")
        return mx.array(np.ascontiguousarray(arr))
    def _moe_fill_proj_raw(self, b: int, proj: str, e: int):
        """Raw mode: dequantize ONE expert `e`'s per-proj slice and cache it.

        Two source layouts are supported:

        (A) 3-D packed `blk.{b}.ffn_{proj}_exps.weight` (real Qwen MoE
            GGUF, e.g. 35B-A3B). n_exp is the LAST axis on disk and each
            per-expert 2-D slice is shape3[0..1]. We dequant only the
            requested expert to bound the LRU working set -- a 35B
            full-record dequant would blow past the cap.

        (B) 2-D per-expert records `blk.{b}.moe_{proj}.weight.e{e}`
            (sparse-upcycled files written by atf.convert_moe.py). Each
            record is already one expert's quantized bytes; self.w(name)
            dequants and caches it.
        """
        nm = f"blk.{b}.moe_{proj}.weight.e{e}"
        # 3-D packed path
        src_3d = f"blk.{b}.ffn_{proj}_exps.weight"
        if src_3d in self._raw_shapes:
            shape3 = self._raw_shapes[src_3d]
            if len(shape3) != 3:
                raise ValueError(
                    f"expected 3-D MoE tensor for {src_3d}, got shape {shape3}")
            n_exp_on_disk = int(shape3[2])
            if not (0 <= e < n_exp_on_disk):
                raise IndexError(
                    f"expert {e} out of range for {src_3d} (n_exp={n_exp_on_disk})")
            from .gguf_io import dequantize as _dq
            dtype = self._raw_dtypes[src_3d]
            rec_len = int(self._raw_len[src_3d])
            per_expert_bytes = rec_len // n_exp_on_disk
            if rec_len != per_expert_bytes * n_exp_on_disk:
                raise ValueError(
                    f"MoE record {src_3d} length {rec_len} not divisible by "
                    f"n_exp={n_exp_on_disk}")
            off = int(self._raw_off[src_3d]) + e * per_expert_bytes
            with open(self._raw_path, "rb") as f:
                f.seek(off)
                data = f.read(per_expert_bytes)
            per_expert_2d = (int(shape3[0]), int(shape3[1]))
            arr = _dq(data, dtype, per_expert_2d)
            # Engine-facing orientation: moe_expert returns (Wg, Wu, Wd)
            # with each [in, out] for x @ W. For [hidden, ffe, n_exp] on
            # disk, [hidden, ffe] is already [in, out] (no transpose). The
            # v3.5 HF convention [n_exp, hidden, ffe] would need .T.
            if arr.ndim == 2 and arr.shape == per_expert_2d:
                w = mx.array(np.ascontiguousarray(arr, dtype=np.float32))
            else:
                w = mx.array(np.ascontiguousarray(arr.T, dtype=np.float32))
            self._lru[nm] = w
            self._lru_bytes += int(w.nbytes)
            while self._lru_bytes > self._lru_cap_bytes and len(self._lru) > 1:
                _, evicted = self._lru.popitem(last=False)
                self._lru_bytes -= int(evicted.nbytes)
            return
        # 2-D per-expert record path (sparse upcycling)
        if nm in self._raw_shapes:
            w = self.w(nm)   # self.w reads + dequants + caches
            return
        # Neither path has the tensor
        raise KeyError(
            f"raw MoE tensor missing: {src_3d} (3-D) and {nm} (2-D) both "
            f"not in this file. Block {b} is not MoE-routable.")





    def merge_ffn(self, b: int, bits: int = 4):
        if self.raw:
            return  # raw mode: merged FFN tensors already exist as records;
            # qmm() resolves the '.dense' names straight onto the kernels.
        """v8: fuse a block's 8 chunk-experts into 3 dense 4-bit matrices.

        convert.py splits gate/up along the intermediate dim ([hidden,
        inter/n] each) and down along the output dim ([inter/n, hidden]),
        so concatenation reproduces the original dense FFN exactly (the
        engine's old sum-of-expert-outputs was mathematically the same).
        Doing this once per block replaces 24 quantized matmuls with 3.

        NOTE: after merging, the per-expert tensors are freed; sparse FFN
        mode for this block is no longer available (all shipped router
        tiers use exact_ffn=True anyway).
        """
        dense_key = f"blk.{b}.ffn"
        if dense_key in self._qcache:
            return
        if bits != 4 and not self.quant:
            raise RuntimeError(
                f"expert_bits={bits} requested but this file has no INT8 "
                "expert data (lod1/lod1-all storage). Reconvert with "
                "--expert-storage dual for adaptive precision.")
        # v2 fast path: weights were packed at convert time -- zero work
        if bits == 4 and (b, "gate") in self.prebuilt_ffn:
            for proj in ("gate", "up", "down"):
                wq, sc, bi = self.prebuilt_ffn[(b, proj)]
                self._qcache[f"blk.{b}.ffn_{proj}.weight.dense"] = (wq, sc, bi)
            self._qcache[dense_key] = True
            return
        n = self.header.num_experts_per_block
        gs, us, ds = [], [], []
        for e in range(n):
            gs.append(self.w(f"blk.{b}.ffn_gate.weight.e{e}"))
            us.append(self.w(f"blk.{b}.ffn_up.weight.e{e}"))
            ds.append(self.w(f"blk.{b}.ffn_down.weight.e{e}"))
        merged = {
            "gate": mx.concatenate(gs, axis=1),   # [hidden, inter]
            "up": mx.concatenate(us, axis=1),     # [hidden, inter]
            "down": mx.concatenate(ds, axis=0),   # [inter, hidden]
        }
        del gs, us, ds
        for proj, W in merged.items():
            wq, s, bits = mx.quantize(W.T, group_size=64, bits=4)
            self._qcache[f"blk.{b}.ffn_{proj}.weight.dense"] = (wq, s, bits)
        mx.eval(*[t for ent in (
            self._qcache[f"blk.{b}.ffn_gate.weight.dense"],
            self._qcache[f"blk.{b}.ffn_up.weight.dense"],
            self._qcache[f"blk.{b}.ffn_down.weight.dense"]) for t in ent])
        # free the per-expert entries (raw INT8 + dequant LRU + qcache)
        for e in range(n):
            for proj in ("gate", "up", "down"):
                nm = f"blk.{b}.ffn_{proj}.weight.e{e}"
                self._qcache.pop(nm, None)
                self.quant.pop(nm, None)
                self.shapes.pop(nm, None)
                old = self._lru.pop(nm, None)
                if old is not None:
                    self._lru_bytes -= old.nbytes
        self._qcache[dense_key] = True

    def _raw_base(self, name: str):
        """Resolve an engine-facing weight name to a raw record + slice.

        Returns (record_name, n_base, k_base, n_out, k_len) or None.
        Handles FFN expert slicing ('.e{e}') and merged-FFN ('.dense').

        v10 perf fix: memoized per name -- this resolution is deterministic
        for the lifetime of a loaded model, but was previously re-run via
        regex on every single qmm()/mm() call (hundreds of times per decode
        token)."""
        if not self.raw:
            return None
        cached = self._raw_base_cache.get(name, "__miss__")
        if cached != "__miss__":
            return cached
        result = self._raw_base_uncached(name)
        self._raw_base_cache[name] = result
        return result

    def _raw_base_uncached(self, name: str):
        if name in self._raw_gpu:
            # n_base = row offset inside the shared pre-packed buffer (0 for
            # unpacked tensors) -- gguf_matmul addresses rows from it.
            return (name, int(self._raw_pack.get(name, 0)), 0, None, None)
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight\.e(\d+)", name)
        if m:
            b, proj, e = int(m.group(1)), m.group(2), int(m.group(3))
            rec = f"blk.{b}.ffn_{proj}.weight"
            if rec in self._raw_gpu:
                shape = self._raw_shapes[rec]
                n_exp = self.header.num_experts_per_block
                if proj == "down":
                    k_chunk = shape[0] // n_exp
                    return (rec, 0, e * k_chunk, shape[1], k_chunk)
                n_chunk = shape[1] // n_exp
                return (rec, e * n_chunk + int(self._raw_pack.get(rec, 0)),
                        0, n_chunk, shape[0])
            # gamma/v5 hotfix: engine.py's call sites for true-MoE experts
            # were renamed from 'blk.{b}.moe_{proj}.weight.e{e}' back to
            # this legacy 'ffn_{proj}.weight.e{e}' naming by a later edit
            # (see /Users/amgad/Desktop/Ai/Claude/atf-qwen3.6-35b-iq2m-load-
            # fix/notes.md for the incident) WITHOUT updating this resolver,
            # so the dense record above doesn't exist for true-MoE models
            # and this used to return None here -> qmm() fell back to
            # self.w(name) @ x.T -> _w_raw() -> KeyError (crash on real
            # weights). Falling through to the SAME true-MoE 3-D '_exps'
            # resolution as the 'moe_{proj}' case below (same math, just
            # reached via the legacy name) makes this resolver work no
            # matter which naming convention a future engine.py edit uses.
            rec3d = f"blk.{b}.ffn_{proj}_exps.weight"
            if rec3d in self._raw_gpu:
                shape3 = self._raw_shapes.get(rec3d)
                if shape3 and len(shape3) == 3:
                    k_full, n_per_expert, n_exp3d = (
                        int(shape3[0]), int(shape3[1]), int(shape3[2]))
                    if 0 <= e < n_exp3d:
                        return (rec3d, e * n_per_expert, 0, n_per_expert, k_full)
            return None
        # gamma/v5 fast-MoE fix: true-MoE per-expert slice of a 3-D
        # ffn_{proj}_exps.weight record (moe_expert()'s naming, distinct
        # from the legacy '.eN' pattern above -- 'moe_' vs 'ffn_'). Per
        # AtfModel._dequant_expert_slice's verified on-disk layout, the
        # record is n_exp full (in, out) = shape3[0:2] matrices
        # concatenated end to end (n_exp is the LAST logical axis, but
        # physically the SLOWEST-varying / outermost on disk) -- i.e.
        # exactly the same "N-chunk of one shared record" shape as the
        # legacy gate/up case just above, just with N_per_expert = the
        # WHOLE per-expert output width (shape3[1]) instead of a split of
        # one dense tensor's output width. That means the existing
        # gguf_matmul/gguf_gather addressing (row n = n_base..n_base+n_out,
        # k = 0..k_full) resolves a single expert's slice correctly with
        # NO new kernel: n_base = e * n_per_expert, k_base = 0,
        # k_len = k_full (shape3[0]) for gate/up/down uniformly -- down's
        # 3-D layout is NOT K-chunked like the legacy '.eN' down case
        # (that was a different, dense-FFN-split scheme); it's the same
        # per-expert-N-chunk shape as gate/up. This routes moe_{proj}.e{e}
        # through qmm()'s existing fast dispatch (_qmm_gemm batched-GEMM
        # for prefill / gguf_matmul for decode) instead of the CPU
        # reference dequantizer in _dequant_expert_slice -- see
        # Engine._ffn_moe, which now calls qmm() directly with this name
        # instead of AtfModel.moe_expert(). NOT yet run on real weights;
        # verify with ATF_VERBOSE=3 before trusting the speedup number.
        m = re.fullmatch(r"blk\.(\d+)\.moe_(gate|up|down)\.weight\.e(\d+)", name)
        if m:
            b, proj, e = int(m.group(1)), m.group(2), int(m.group(3))
            rec = f"blk.{b}.ffn_{proj}_exps.weight"
            if rec not in self._raw_gpu:
                return None
            shape3 = self._raw_shapes.get(rec)
            if not shape3 or len(shape3) != 3:
                return None
            k_full, n_per_expert, n_exp = (int(shape3[0]), int(shape3[1]),
                                           int(shape3[2]))
            if not (0 <= e < n_exp):
                return None
            return (rec, e * n_per_expert, 0, n_per_expert, k_full)
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight\.dense", name)
        if m:
            rec = f"blk.{m.group(1)}.ffn_{m.group(2)}.weight"
            if rec in self._raw_gpu:
                return (rec, int(self._raw_pack.get(rec, 0)), 0, None, None)
        return None

    def has(self, name: str) -> bool:
        if self.raw:
            if name in self._raw_gpu or self._raw_base(name):
                return True
        return name in self.quant or name in self._qcache

    # ─── gamma/v2: n-gram embedding lookup ─────────────────────────────
    # ─── gamma/v3: QSA (Qwen Sparse Attention) lookup ─────────────────
    def w_qsa_indexer(self, name: str):
        """Resolve a QSA indexer weight to a dequantized f32 mx.array.

        The QSA block carries two new tensor families (per the analysis
        doc section 3.3 + 6.6 of the converter-stub plan):

          - "blk.<b>.attn_indexer.q_proj.weight"  [hidden, indexer_heads * head_dim]
          - "blk.<b>.attn_indexer.k_proj.weight"  [hidden, kv_heads * head_dim]
          - "blk.<b>.attn_indexer.head_dim.weight" (if needed) or similar

        The main-attn q/k/v/o for the QSA block ride the existing
        attn_q/k/v/o weight names (the QSA spec uses the same
        projection family as full-attn, just with different head counts
        -- e.g. Qwen4: 24 Q heads, 2 KV heads, head_dim 256 for the
        main path; 4 Q heads + 1 K head for the indexer).

        For v3, this is a single dispatch to w() with a key that the
        loader puts in the dense section. A real converter integration
        would put the indexer weights in the dense section (tiny) and
        the main-attn q/k/v in the same slots as full-attn.
        """
        return self.w(name)

    def qsa_block_kind(self, b: int) -> str:
        """Return "qsa" if block b has QSA indexer tensors, else "full"
        if it has standard full-attn tensors, else "gdn" for Gated
        DeltaNet blocks. Used by Engine._forward_block to dispatch."""
        prefix = f"blk.{b}."
        if self.has(f"{prefix}attn_indexer.q_proj.weight"):
            return "qsa"
        if self.has(f"{prefix}attn_q.weight"):
            return "full"
        return "gdn"

    def w_ngram(self, name: str):
        """Resolve an n-gram tensor to a dequantized f32 mx.array.

        The three n-gram tensors (per docs/qwen38_flash_next_analysis.md
        §6.3) are:

          - "ngram_emb.weight"      [ngram_vocab, ngram_dim]   main table
                                    (LOD-1 int4 in the expert section
                                    for production Qwen4 weights;
                                    dense f16/f32 for the synthetic
                                    roundtrip tests in this tree)
          - "ngram_emb_proj.weight" [ngram_dim, hidden]        tiny projection
                                    (dense f16/f32, lives in the
                                    dense section)
          - "ngram_emb_norm.weight" [ngram_dim]                tiny RMSNorm
                                    (dense f16/f32, lives in the
                                    dense section; precision-critical)

        The big table routes through the existing _raw_base / dequant
        path so the LRU paging in this class already keeps the
        working set bounded. The small tensors route through w() (they
        are tiny so they stay resident in the LRU trivially).
        """
        if name in self.quant or name in self._qcache:
            return self.w(name)
        if self.raw and (name in self._raw_gpu
                         or self._raw_base(name) is not None):
            return self.w(name)
        raise KeyError(f"w_ngram: no such ngram tensor {name!r} "
                       f"(have quant keys: {sorted(self.quant.keys())[:5]}...)")




_RAW_STORAGE_CODES = {2, 4, 5}   # v8 int8 / qmm / fp4 -- anything else in a
                                  # flags&0x04 file is a GGUF dtype code

def _load_atf_raw(path: Path, f, mm, header, manifest, load_time: float,
                  lru_cap_bytes: int) -> AtfModel:
    """Load a raw-GGUF .atf: index tensor payloads and upload them to the
    GPU byte-for-byte."""
    from .gguf_io import dequantize as _dq
    console.print("  [green]Raw-GGUF mode: original quantization kept, "
                  "GPU-resident packed weights[/green]")

    gpu: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, int] = {}
    offs: dict[str, int] = {}
    lens: dict[str, int] = {}

    (count,) = struct.unpack_from("<I", mm, header.offset_dense)
    pos = header.offset_dense + 4
    entries = []
    for _ in range(count):
        (nlen,) = struct.unpack_from("<H", mm, pos); pos += 2
        name = mm[pos:pos + nlen].decode("utf-8"); pos += nlen
        dtype_code, ndim = struct.unpack_from("<BB", mm, pos); pos += 2
        shape = struct.unpack_from(f"<{ndim}I", mm, pos); pos += 4 * ndim
        blob_off, nbytes = struct.unpack_from("<QQ", mm, pos); pos += 16
        entries.append((name, shape, blob_off, nbytes, dtype_code))
    blob_base = pos

    # v11 memory-neutral fusion
    # v14 memory-pressure fix: mapped file pages stay resident once touched
    # and stacked on top of the growing GPU allocation during load (27B peak
    # RSS was 10.3 GB, system free down to 28%). Advise each payload range
    # MADV_DONTNEED as soon as its copy into the GPU-resident buffer exists;
    # the mapping is read-only so every page is clean and simply re-faults
    # from the file if ever touched again (it never is).
    _PAGE = 16384                   # macOS arm64 page size

    def _release(off: int, ln: int):
        if os.environ.get("ATF_NO_MADVISE"):
            return
        try:
            start = off - (off % _PAGE)
            end = min((off + ln + _PAGE - 1) // _PAGE * _PAGE, mm.size())
            if end > start:
                mm.madvise(mmap.MADV_DONTNEED, start, end - start)
        except Exception:
            pass

    # v11 memory-neutral fusion: pre-pack same-input projection groups into
    # ONE shared GPU buffer each, so qmm_fuse() runs one kernel dispatch per
    # group at decode time with ZERO extra residency. Members share the same
    # mx.array; their per-name row offsets land in `pack`.
    from .gguf_metal import supported as _metal_supported
    name2idx = {e[0]: i for i, e in enumerate(entries)}
    pack: dict[str, int] = {}
    groups_packed = 0
    for b in range(header.num_blocks):
        p = f"blk.{b}."
        templates = (
            [p + "attn_qkv.weight", p + "attn_gate.weight",
             p + "ssm_alpha.weight", p + "ssm_beta.weight"],   # GDN inputs
            [p + "attn_q.weight", p + "attn_k.weight",
             p + "attn_v.weight"],                             # full-attn Q/K/V
            [f"blk.{b}.ffn_gate.weight",                       # FFN gate+up
             f"blk.{b}.ffn_up.weight"],
        )
        for tmpl in templates:
            idxs = [name2idx[t] for t in tmpl if t in name2idx]
            if len(idxs) < 2:
                continue
            dts = {entries[i][4] for i in idxs}
            ks = {entries[i][1][0] for i in idxs}
            if len(dts) != 1 or len(ks) != 1 \
                    or not _metal_supported(next(iter(dts))):
                continue          # mismatched dtype/K: leave them separate
            combo = np.empty(sum(entries[i][3] for i in idxs), dtype=np.uint8)
            off_b = 0
            row_off = 0
            for i in idxs:
                nm, shp, blob_off, nbytes, dt_i = entries[i]
                combo[off_b:off_b + nbytes] = np.frombuffer(
                    mm, dtype=np.uint8, count=nbytes,
                    offset=blob_base + blob_off)
                shapes[nm] = tuple(int(v) for v in shp)
                dtypes[nm] = dt_i
                offs[nm] = blob_base + blob_off
                lens[nm] = nbytes
                pack[nm] = row_off
                row_off += int(shp[1])   # one row per output element
                off_b += nbytes
            arr = mx.array(combo)
            del combo                      # v14: staging buffer out ASAP
            for i in idxs:
                gpu[entries[i][0]] = arr
            # release all member ranges now (copy into GPU is done)
            for i in idxs:
                _nm, _shp, _boff, _nbytes, _dt = entries[i]
                _release(blob_base + _boff, _nbytes)
            groups_packed += 1

    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TimeElapsedColumn(),
                  console=console, transient=True) as progress:
        task = progress.add_task("Uploading packed weights to GPU", total=count)
        for name, shape, blob_off, nbytes, dtype_code in entries:
            progress.advance(task)
            if name in pack:            # already uploaded as a shared buffer
                continue                # (range released in the sweep below)
            payload = np.frombuffer(mm, dtype=np.uint8, count=nbytes,
                                    offset=blob_base + blob_off)
            try:
                gpu[name] = mx.array(payload)
                if int(os.environ.get("ATF_VERBOSE_LOAD", "0")):
                    console.print(f"    [green]Uploaded {name}: {nbytes/1e6:.1f} MB, shape {tuple(shape)}[/green]")
            except Exception as e:
                console.print(f"    [red]FAILED to upload {name} ({nbytes/1e6:.1f} MB, shape {tuple(shape)}): {e}[/red]")
                raise
            _release(blob_base + blob_off, nbytes)
            shapes[name] = tuple(int(v) for v in shape)
            dtypes[name] = dtype_code
            offs[name] = blob_base + blob_off
            lens[name] = nbytes
    # fusion-packed members were copied into shared staging buffers earlier;
    # sweep their ranges too so NO tensor page survives upload.
    for name, shape, blob_off, nbytes, dtype_code in entries:
        if name in pack:
            _release(blob_base + blob_off, nbytes)

    # router (same layout as v8 files)
    (rsize,) = struct.unpack_from("<I", mm, header.offset_router)
    router_np = np.frombuffer(mm, dtype=np.float32, count=rsize // 4,
                              offset=header.offset_router + 4).reshape(
        header.num_blocks, header.hidden_dim, header.num_experts_per_block).copy()
    router = mx.array(router_np)

    total_bytes = sum(lens.values())
    console.print(f"  Tensors: {len(entries)} | packed bytes: {total_bytes/1e9:.2f} GB | {gpu_mem_line()}")
    if groups_packed:
        console.print(f"  Pre-packed {groups_packed} fusion groups into shared "
                      f"buffers (one dispatch per group, no extra memory)")

    # v14: parsing/uploads are done -- drop the mapping entirely and return
    # the allocator's scratch to the OS before inference starts.
    try:
        mm.close()
    except Exception:
        pass
    try:
        import mlx.core as _mx
        _mx.clear_cache()
    except Exception:
        pass

    console.print(f"  [bold yellow]Model load complete: {len(entries)} tensors, "
                  f"{total_bytes/1e9:.2f} GB | memory = {gpu_mem_line()}[/bold yellow]")

    model = AtfModel(header, manifest, {}, router, shapes, load_time,
                     int(total_bytes), lru_cap_bytes=lru_cap_bytes,
                     raw={"gpu": gpu, "shapes": shapes, "dtypes": dtypes,
                          "path": path, "offs": offs, "lens": lens,
                          "pack": pack})
    return model


def load_atf(path: Path, use_gpu: bool = True, lru_cap_bytes: int | None = None) -> AtfModel:
    start = time.time()
    path = Path(path)
    console.rule("[bold green]ATF Model Loading")
    console.print(f"  File: {path.name} ({path.stat().st_size / 1e9:.3f} GB)")

    with open(path, "rb") as f:
        header = _read_header(f)
        manifest = _read_manifest(f, header.offset_manifest)

    console.print(f"  Blocks: {header.num_blocks}, Hidden: {header.hidden_dim}")
    console.print(f"  Experts/block: {header.num_experts_per_block}, top_k: {header.top_k}")
    console.print(f"  Vocab: {header.vocab_size}")
    console.print("  Storage: INT8 resident, on-the-fly dequant (LOD 0)")

    quant_np: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    total_bytes = 0

    # v8: keep the mmap open until the GPU transfer is done -- the np views
    # into it back the weight arrays until mx.array() copies them over.
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if getattr(header, "flags", 0) & 0x04:
        return _load_atf_raw(path, f, mm, header, manifest,
                             time.time() - start,
                             lru_cap_bytes=(lru_cap_bytes if lru_cap_bytes is not None
                                            else LRU_CAP_BYTES))

    # NOTE: mm is intentionally NOT closed -- np.frombuffer views into it back
    # the weight arrays until their mx.array() copies are taken, and an early
    # close raises BufferError. The mapping is file-backed (no RAM cost) and
    # dies with the process / when the views are collected.

    # v8 marker (kept open) the GPU transfer below (the np.frombuffer
    # views into it back the weight arrays until mx.array() copies them).
    if True:  # (body kept at its original indentation)
        (count,) = struct.unpack_from("<I", mm, header.offset_dense)
        pos = header.offset_dense + 4
        entries = []
        for _ in range(count):
            (nlen,) = struct.unpack_from("<H", mm, pos); pos += 2
            name = mm[pos:pos + nlen].decode("utf-8"); pos += nlen
            dtype_code, ndim = struct.unpack_from("<BB", mm, pos); pos += 2
            shape = struct.unpack_from(f"<{ndim}I", mm, pos); pos += 4 * ndim
            blob_off, nbytes = struct.unpack_from("<QQ", mm, pos); pos += 16
            entries.append((name, shape, blob_off, nbytes, dtype_code))
        blob_base = pos
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TimeElapsedColumn(),
                      console=console, transient=True) as progress:
            task = progress.add_task("Dense tensors", total=count)
            dense_prebuilt: dict = {}
            for name, shape, blob_off, nbytes, dcode in entries:
                progress.advance(task)
                if dcode == 4:
                    # v2: payload is already in quantized_matmul layout --
                    # wire it straight into the runtime cache, zero work.
                    rows, cols = shape[1], shape[0]     # stored as W.T
                    base = blob_base + blob_off
                    rq, rc = struct.unpack_from("<II", mm, base)
                    wq = mx.array(np.frombuffer(
                        mm, dtype=np.uint32,
                        count=rq * ((rc + 7) // 8),
                        offset=base + 8).reshape(rq, (rc + 7) // 8))
                    off2 = base + 8 + wq.nbytes
                    ns = rq * (rc // 64)
                    sc = mx.array(np.frombuffer(mm, dtype=np.float16,
                                                count=ns, offset=off2).reshape(rq, rc // 64))
                    bi = mx.array(np.frombuffer(mm, dtype=np.float16,
                                                count=ns, offset=off2 + ns * 2).reshape(rq, rc // 64))
                    # matmul weights served straight through qmm()
                    dense_prebuilt[name] = (wq, sc.astype(mx.float32),
                                            bi.astype(mx.float32))
                    shapes[name] = shape   # engine.__init__ probes these
                    total_bytes += int(wq.nbytes) + ns * 4
                    continue
                elif dcode == 1:
                    # v18 MLX-source files: raw f16 blob, full precision
                    n = int(np.prod(shape)) if shape else 1
                    arr = np.frombuffer(mm, dtype=np.float16, count=n,
                                        offset=blob_base + blob_off)
                    arr = arr.astype(np.float32).reshape(shape)
                    quant_np[name] = (arr, np.ones((1, 1), dtype=np.float32))
                    shapes[name] = shape
                    total_bytes += arr.nbytes
                    continue
                elif dcode == 5:
                    # v8.1: FP4 dense blob -- decode to f32 once at load;
                    # served through the same quant path with unit scales.
                    wf32, _, bsz = _parse_fp4_blob_at(mm, blob_base + blob_off)
                    quant_np[name] = (wf32.astype(np.float32),
                                      np.ones((1, 1), dtype=np.float32))
                    shapes[name] = shape
                    total_bytes += wf32.nbytes
                    continue
                q, scales = _parse_int8_blob_at(mm, blob_base + blob_off)[:2]
                quant_np[name] = (q, scales)
                shapes[name] = shape
                total_bytes += q.nbytes + scales.nbytes
        console.print(f"  Dense tensors read: {len(entries)} | {total_bytes/1e9:.2f} GB | {host_mem_line()}")

        # ── experts ────────────────────────────────────────────────────────
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TimeElapsedColumn(),
                      console=console, transient=True) as progress:
            task = progress.add_task("Expert INT8 weights", total=len(manifest.experts))
            for entry in manifest.experts:
                progress.advance(task)
                if entry.weight_size == 0:
                    continue        # v2 lod1 storage: no INT8 expert data
                cur = header.offset_experts + entry.weight_offset
                e = entry.expert_id % header.num_experts_per_block
                b = entry.block_index
                for proj in ("gate", "up", "down"):
                    q, scales, bsize = _parse_int8_blob_at(mm, cur)
                    cur += bsize
                    ekey = f"blk.{b}.ffn_{proj}.weight.e{e}"
                    quant_np[ekey] = (q, scales)
                    shapes[ekey] = q.shape
                    total_bytes += q.nbytes + scales.nbytes
        console.print(f"  Expert weights read: {len(manifest.experts)} | {total_bytes/1e9:.2f} GB total | {host_mem_line()}")

        # ── router ─────────────────────────────────────────────────────────
        (rsize,) = struct.unpack_from("<I", mm, header.offset_router)
        router_np = np.frombuffer(mm, dtype=np.float32, count=rsize // 4,
                                  offset=header.offset_router + 4).reshape(
            header.num_blocks, header.hidden_dim, header.num_experts_per_block
        ).copy()

    # ── v8 LOD1 section (immediately after router; legacy files have the
    # 32-byte footer there instead) ────────────────────────────────────────
    ffn_prebuilt: dict = {}
    pos_after_router = header.offset_router + 4 + router_np.nbytes
    if mm[pos_after_router:pos_after_router + 4] == b"LOD1":
        p2 = pos_after_router + 4
        nblocks_l1 = struct.unpack_from("<I", mm, p2)[0]
        p2 += 4
        console.print(f"  LOD1 section: {nblocks_l1} blocks of pre-packed FFN")
        import mlx.core as _mx
        for bidx in range(nblocks_l1):
            for proj in ("gate", "up", "down"):
                rq, rc = struct.unpack_from("<II", mm, p2); p2 += 8
                nq = rq * ((rc + 7) // 8)
                wq = _mx.array(np.frombuffer(mm, np.uint32, count=nq, offset=p2)
                               .reshape(rq, (rc + 7) // 8))
                p2 += nq * 4
                ns = rq * (rc // 64)
                sc = _mx.array(np.frombuffer(mm, np.float16, count=ns, offset=p2)
                               .reshape(rq, rc // 64).astype(np.float32))
                p2 += ns * 2
                bi = _mx.array(np.frombuffer(mm, np.float16, count=ns, offset=p2)
                               .reshape(rq, rc // 64).astype(np.float32))
                p2 += ns * 2
                ffn_prebuilt[(bidx, proj)] = (wq, sc, bi)

    load_time = time.time() - start
    ram = total_bytes + router_np.nbytes
    console.print(f"  Dense tensors: {len(entries)} | Experts: {len(manifest.experts)}")
    cap = lru_cap_bytes if lru_cap_bytes is not None else LRU_CAP_BYTES
    # Rough working-set estimate: int8 resident bytes dequantized to fp32 is
    # a 4x blowup. With exact_ffn (the default) EVERY forward pass -- prefill
    # AND every single decode step -- touches ALL experts in ALL blocks, not
    # just top-k, so this whole figure is what a cache needs to hold to avoid
    # re-dequantizing from scratch on every single token.
    dequant_fp32_estimate = ram * 4
    console.print(f"  [green]Loaded in {load_time:.1f}s | INT8 resident: {ram / 1e9:.2f} GB"
                  f" (+ f32 LRU cache cap {cap / (1 << 30):.1f} GB)[/green]")
    console.print(f"  [yellow]Full dequantized (fp32) working set if fully cached: "
                  f"~{dequant_fp32_estimate / 1e9:.1f} GB[/yellow]")
    if cap < dequant_fp32_estimate:
        console.print(f"  [yellow]Cache cap ({cap / (1 << 30):.1f} GB) is smaller than the full working set "
                       f"(~{dequant_fp32_estimate / 1e9:.1f} GB) -- with exact_ffn on, expect the cache to "
                       f"thrash and most weights to be re-dequantized on every single forward pass "
                       f"(prefill AND every decode token). Check the cache_hits/misses printed after "
                       f"generation; a low hit rate confirms this is the bottleneck.[/yellow]")

    # Convert to MLX arrays on GPU
    console.print(f"  Before GPU transfer: {host_mem_line()} | {gpu_mem_line()}")
    xfer_start = time.time()
    console.print("  Moving INT8 weights to GPU (MLX)...")
    names = list(quant_np.keys())
    quant = {}
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task("GPU transfer", total=len(names))
        for name in names:
            progress.advance(task)
            q, scales = quant_np.pop(name)
            quant[name] = (mx.array(q), mx.array(scales))
    router = mx.array(router_np)
    del quant_np, router_np  # release the mmap-backed views explicitly


    xfer_elapsed = time.time() - xfer_start
    console.print(f"  Transfer done in {xfer_elapsed:.1f}s | After: {host_mem_line()} | {gpu_mem_line()}")

    model = AtfModel(header, manifest, quant, router, shapes, load_time, int(ram),
                     lru_cap_bytes=cap, prebuilt_ffn=ffn_prebuilt)
    # v2: pre-packed dense matmul weights skip the INT8 path entirely
    if dense_prebuilt:
        model._qcache.update(dense_prebuilt)
    return model
