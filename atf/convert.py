"""GGUF -> ATF conversion pipeline (streaming, memory-bounded).

Design (v1, ~2x compression of a BF16 source):
  - EVERY tensor is stored INT8 (per-row scales): dense + expert FFN
  - File layout (offsets in header, written in this order):
      [header placeholder][dense][experts][manifest][index][router][footer sha256]
    The reader is offset-driven, so section order does not matter; writing
    manifest last lets us fill domain labels after classification, which
    happens in the same pass as the expert write (no second dequant sweep).
  - Memory: one FFN block (f32, ~600 MB) or one dense tensor at a time.

Passes:
  A. metadata only (tensor names/shapes, no data)
  B. stream dense tensors: dequant -> INT8 -> write
  C. stream per-block FFN: dequant -> split into experts -> INT8 -> write,
     collecting gate stats (classification) + router column means
  D. manifest + index + router, footer hash, header patch
"""
from __future__ import annotations
import os

import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from .format import (
    Header, LayerManifest, ExpertEntry, KnowledgeIndex,
    LodLevel, Specificity, ArchType,
    HEADER_SIZE, FOOTER_SIZE, MAGIC,
)
from .gguf_io import dequantize
from .quantize import quantize_int8, dequantize_int8, quantize_fp4, dequantize_fp4
from .taxonomy import classify_from_stats, chunk_stats, DOMAIN_NAMES, SUB_DOMAIN_NAMES


def _native_shape(t) -> tuple[int, ...]:
    """GGUF-native (logical) shape of a tensor -- NOT reversed, NOT byte-shaped.

    Pass this to `dequantize` so the dequantized array has the right row/col
    orientation (e.g. blk.0.attn_q.weight comes back as [4096, 8192], not
    [8192, 4096]).
    """
    return tuple(int(s) for s in t.shape)


def _dequantize_source_tensor(reader, t) -> np.ndarray:
    """Decode one GGUF tensor with validation before ATF quantization.

    This is the Q4_NL/mixed-GGUF conversion entry point.  GGUF tensors may
    use IQ4_NL, Q4_K/Q5_K/Q6_K, Q8_0, F32, or BF16 independently; conversion
    must dispatch by the tensor's actual type and must never allow NaN/Inf to
    reach the ATF quantizer.
    """
    arr = dequantize(bytes(t.data), int(t.tensor_type), _native_shape(t))
    if not np.isfinite(arr).all():
        raise ValueError(
            f"non-finite values while dequantizing {t.name!r} "
            f"(GGUF tensor type {int(t.tensor_type)}); source quantizer is "
            "not supported safely"
        )
    return arr


console = Console()


# gamma/v2: n-gram embedding (Qwen4) converter stub
# Per docs/qwen38_flash_next_analysis.md section 6.8, the n-gram
# converter is gated on a real Qwen4 GGUF landing in the model dir so
# we can probe the exact tensor names. Until then, this helper
# recognizes the predicted tensor family and returns the metadata
# (vocab, dim, insert_layer, context). If no ngram tensors are present
# the downstream code path produces a v2 header (no ngram).
#
# When a real Qwen4 GGUF arrives, set ATF_NGRAM_CONVERTER=1 to enable
# the rewrite. Without the flag this is a no-op.
_NGRAM_TENSOR_RE = re.compile(
    r"^(embedding|model)\.ngram_embedding(?:_table|_hash|_value)?\.weight$"
)


def _detect_ngram_section(reader):
    if os.environ.get("ATF_NGRAM_CONVERTER", "0") != "1":
        return (0, 0, 2, 2)
    for t in reader.tensors:
        if _NGRAM_TENSOR_RE.match(t.name):
            shape = tuple(int(s) for s in t.shape)
            if "ngram_embedding_table" in t.name or "ngram_embedding_value" in t.name:
                if len(shape) >= 2:
                    return (int(shape[0]), int(shape[1]), 2, 2)
            elif "ngram_embedding_hash" in t.name:
                if len(shape) >= 1:
                    return (int(shape[0]), 0, 2, 2)
    return (0, 0, 2, 2)


# ─── gamma/v3: QSA (Qwen Sparse Attention) — converter stub ────────────
# Per docs/qwen38_flash_next_analysis.md §3.3, the QSA converter is gated
# on a real Qwen4 GGUF landing in the model dir, so we can probe the
# exact tensor names. Until then, this stub recognizes the predicted
# tensor family and returns the metadata (indexer_heads, kv_heads,
# budget_blocks, block_size). If no QSA tensors are present, returns
# (0, 0, 0, 0) and the downstream code path produces a v3 header (no QSA).
#
# When a real Qwen4 GGUF arrives, set ATF_QSA_CONVERTER=1 to enable
# the rewrite. Without the flag this is a no-op.
_QSA_TENSOR_RE = re.compile(
    r"^blk\.(\d+)\.attn_indexer\.(q_proj|k_proj|q_norm|k_norm)\.weight$"
)


def _detect_qsa_section(reader):
    """Return (indexer_heads, kv_heads, budget_blocks, block_size) from a
    Qwen4 reader, or (0, 0, 0, 0) if no QSA tensors are found.

    The values are PINNED from the Qwen3.8-Flash-Next model card
    (analysis doc §1.1 + §3.3) -- not measured on ATF:
      indexer_heads = 4
      kv_heads      = 1 (shared)
      budget_blocks = 512
      block_size    = 16 tokens/block

    A real Qwen4 release may differ; the stub returns the pinned values
    and the engine honors them. The shape of the q_proj tensor
    ([hidden, indexer_heads * head_dim]) is the cross-check that the
    model actually has QSA weights; if it does, the engine's autodetect
    populates self.qsa_blocks.
    """
    if os.environ.get("ATF_QSA_CONVERTER", "0") != "1":
        return (0, 0, 0, 0)
    has_qsa = False
    for t in reader.tensors:
        if _QSA_TENSOR_RE.match(t.name):
            has_qsa = True
            break
    if not has_qsa:
        return (0, 0, 0, 0)
    # Pinned values per analysis doc §1.1 + §3.3. Real Qwen4 may
    # differ; update these constants when measured.
    return (4, 1, 512, 16)


@dataclass
class ConvertConfig:
    gguf_path: Path
    output_path: Path
    num_experts: int = 8
    top_k: int = 4
    ffn_pattern: str = r"blk\.(\d+)\.ffn_(gate|up|down)\.weight"
    # v8 format-v2 expert storage:
    #   "lod0"  legacy INT8-only (v1-compatible files)
    #   "dual"  INT8 experts + packed 4-bit merged FFN (adaptive tiers)
    #   "lod1"  ONLY the packed 4-bit merged FFN (smallest files; no 8-bit tier)
    #   "auto"  dual for unquantized sources, lod1 when source is pre-quantized GGUF
    expert_storage: str = "auto"
    dense_fp4: bool = False   # store remaining dense tensors as FP4 (E2M1, g64)
    # "legacy" = v1/v2 INT8/LOD pipeline; "raw" = v3 raw-GGUF preservation
    # (original quantization kept byte-for-byte; loaded by _load_atf_raw)
    mode: str = "legacy"


@dataclass
class ConvertResult:
    output_path: Path
    num_experts: int
    num_blocks: int
    total_size: int
    original_size: int
    duration: float


def _int8_blob_size(shape: tuple[int, ...]) -> int:
    """Size of quantize_int8 output for a given shape (1D treated as (1, n))."""
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    else:
        rows, cols = shape[0], shape[1]
    return 8 + rows * 4 + rows * cols


def convert(config: ConvertConfig) -> ConvertResult:
    if getattr(config, "mode", "legacy") == "raw":
        return convert_raw(config)
    return convert_legacy(config)


def convert_legacy(config: ConvertConfig) -> ConvertResult:
    start = time.time()
    gguf_path = Path(config.gguf_path)
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]ATF Conversion")
    console.print(f"  Input:   {gguf_path.name} ({gguf_path.stat().st_size / 1e9:.2f} GB)")
    console.print(f"  Output:  {out_path}")
    console.print(f"  Experts/block: {config.num_experts}, top_k: {config.top_k}")
    console.print(f"  Storage:   INT8 (dense + experts), on-the-fly dequant at load")
    console.print()

    from gguf import GGUFReader
    reader = GGUFReader(str(gguf_path))

    # ── Pass A: fields + metadata ──────────────────────────────────────────
    arch = _get_str(reader, "general.architecture")
    prefix = arch
    declared_blocks = int(_get_val(reader, f"{prefix}.block_count"))
    # Qwen3.5 IQ4_NL files commonly append one MTP block.  The base engine
    # executes only the trunk; blk.N with nextn.eh_proj is not a base block.
    base_blocks = [
        int(m.group(1)) for t in reader.tensors
        if (m := re.match(r"blk\.(\d+)\.nextn\.eh_proj\.weight$", t.name))
    ]
    num_blocks = min(declared_blocks, min(base_blocks, default=declared_blocks))
    hidden = int(_get_val(reader, f"{prefix}.embedding_length"))
    ffn_dim, moe_note = _resolve_moe_metadata(reader, prefix, config)
    num_heads = int(_get_val(reader, f"{prefix}.attention.head_count"))
    num_kv = int(_get_val(reader, f"{prefix}.attention.head_count_kv"))
    vocab, transpose_weights = _get_vocab(reader, hidden)
    moe_blocks = _moe_blocks_of(reader, num_blocks)

    # RoPE rotation dimension. Most models use full head_dim, but Qwen3 / Qwen3.5
    # only rotate the first `rope.dimension_count` dims (with mRoPE sections).
    # Default to head_dim when absent so older / non-RoPE models still work.
    head_dim = hidden // num_heads
    try:
        rope_dim = int(_get_val(reader, f"{prefix}.rope.dimension_count"))
    except (ValueError, KeyError):
        rope_dim = head_dim
    if rope_dim <= 0 or rope_dim > head_dim:
        rope_dim = head_dim

    console.print(f"  {arch}: {num_blocks} blocks, hidden={hidden}, ffn={ffn_dim}, "
                  f"heads={num_heads} (kv {num_kv}), vocab={vocab}, rope_dim={rope_dim}")
    if moe_note:
        console.print(f"  [cyan]MoE metadata: {moe_note}[/cyan]")
    if moe_blocks:
        console.print(f"  [cyan]True-MoE model: {len(moe_blocks)} routed-expert "
                      f"blocks (experts={config.num_experts}, top_k={config.top_k}); "
                      "experts stored as per-expert dense records[/cyan]")
    console.print()

    chunk = ffn_dim // max(1, config.num_experts)
    if chunk * config.num_experts != ffn_dim:
        raise ValueError(f"ffn_dim {ffn_dim} not divisible by num_experts {config.num_experts}")

    exps_re = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
    # name -> expert index for synthetic per-expert records carved out of a
    # 3-D `ffn_*_exps` tensor (stored in the dense section; the loader reads
    # them like any other dense tensor).
    moe_slices: dict[str, int] = {}
    ffn_re = re.compile(config.ffn_pattern)
    dense_meta: list[tuple[str, tuple[int, ...], int]] = []   # name, shape, tensor idx
    ffn_tensors: list[tuple[int, str, int]] = []              # block, proj, tensor idx
    # gamma/v1: collect MTP-block (blk.N.nextn.*) tensors and rename them
    # to the mtp.* namespace so the engine can find them via model.w().
    # Without this rename, the next two filters (the explicit "skip nextn"
    # that used to be here, and the block_match >= num_blocks cutoff that
    # shrinks num_blocks past the MTP block) would both drop the drafter
    # weights. See SPEC_DECODING_PROPOSAL.md §0 and
    # docs/qwen38_flash_next_analysis.md §3.2.
    mtp_re = re.compile(r"^blk\.(\d+)\.nextn\.(eh_proj|enorm|hnorm|shared_head_head|shared_head_norm)\.weight$")
    mtp_block_ids: list[int] = []
    for i, t in enumerate(reader.tensors):
        m_mtp = mtp_re.match(t.name)
        if m_mtp:
            mtp_block_ids.append(int(m_mtp.group(1)))
    mtp_block_ids = sorted(set(mtp_block_ids))
    # Map each MTP block's declared index (e.g. 32) to a 0-based MTP slot
    # (e.g. mtp.0). Convention: MTP blocks follow the trunk contiguously,
    # so block N+1 = mtp.0, N+2 = mtp.1, etc. (num_blocks = N + len(mtp_block_ids)).
    mtp_slot_of: dict[int, int] = {b: k for k, b in enumerate(mtp_block_ids)}
    for i, t in enumerate(reader.tensors):
        m_mtp = mtp_re.match(t.name)
        if m_mtp:
            b_mtp = int(m_mtp.group(1))
            slot = mtp_slot_of[b_mtp]
            field = m_mtp.group(2)
            # `shared_head_head` lives once per MTP block in the GGUF but the
            # engine will tie-break to whichever is the last MTP block; the
            # weights are identical across MTP slots in Qwen3.5+, so we
            # keep a single `mtp.shared_head_head.weight` from the first
            # MTP block and drop later copies with a warn-once message.
            if field in ("shared_head_head", "shared_head_norm"):
                if slot == 0:
                    new_name = "mtp.shared_head_head.weight"
                else:
                    console.print(f"  [yellow]MTP: dropping duplicate "
                                  f"{t.name} (kept mtp.shared_head_head.weight "
                                  f"from blk.{mtp_block_ids[0]})[/yellow]")
                    continue
            else:
                new_name = f"mtp.{slot}.{field}.weight"
            shape = tuple(int(s) for s in t.shape)
            if transpose_weights and len(shape) == 2:
                shape = shape[::-1]
            dense_meta.append((new_name, shape, i))
            continue
        block_match = re.match(r"blk\.(\d+)\.", t.name)
        if block_match and int(block_match.group(1)) >= num_blocks:
            continue
        # Logical (engine-facing) shape: GGUF-native dims; we apply the
        # transpose below for tensors that need it so the stored shape
        # matches what the engine expects (e.g. attn_q.weight = [hidden, 2*hidden]).
        shape = tuple(int(s) for s in t.shape)
        if transpose_weights and len(shape) == 2:
            shape = shape[::-1]
        m_exp = exps_re.match(t.name)
        if m_exp:
            # True-MoE block: carve the 3-D expert tensor into per-expert
            # 2-D records with engine-facing [in, out] shapes so they ride
            # the normal dense section (loader needs no changes).
            #
            # Verified against the real Qwen-AgentWorld-35B-A3B GGUF: the
            # on-disk GGUF shape of `ffn_{gate,up,down}_exps.weight` is
            # (hidden, ffe, n_exp) for gate/up and (ffe, hidden, n_exp)
            # for down -- i.e. n_exp is the LAST axis. The per-expert 2-D
            # slice is therefore native[0..1] = (hidden, ffe) for gate/up
            # (= engine-facing [in, out]) and (ffe, hidden) for down.
            # We do NOT transpose here; the dequant step below also uses
            # native[0..1] as the per-expert shape.
            b_exp = int(m_exp.group(1))
            native = tuple(int(s) for s in t.shape)
            for e in range(config.num_experts):
                nm = f"blk.{b_exp}.moe_{m_exp.group(2)}.weight.e{e}"
                dense_meta.append((nm, (native[0], native[1]), i))
                moe_slices[nm] = e
            continue
        m = ffn_re.match(t.name)
        if m:
            ffn_tensors.append((int(m.group(1)), m.group(2), i))
        else:
            dense_meta.append((t.name, shape, i))

    ffn_by_block: dict[int, dict[str, int]] = {}
    for b, proj, i in ffn_tensors:
        ffn_by_block.setdefault(b, {})[proj] = i
    missing = [b for b in range(num_blocks)
               if b not in moe_blocks
               and set(ffn_by_block.get(b, {})) != {"gate", "up", "down"}]
    if missing:
        raise ValueError(f"blocks missing FFN tensors: {missing}")

    total_dense_bytes = sum(_int8_blob_size(s) for _, s, _ in dense_meta)
    total_expert_bytes = num_blocks * config.num_experts * (
        _int8_blob_size((hidden, chunk)) * 2 + _int8_blob_size((chunk, hidden))
    )
    if config.dense_fp4:
        fp4_bytes = sum(
            8 + int(s[0]) * (int(s[1]) // 64) * 4 + int(s[0]) * ((int(s[1]) + 1) // 2)
            for n, s, _ in dense_meta
            if len(s) == 2 and int(s[1]) % 64 == 0
            and s not in ((1, 1),)
        )
        # embeddings/LM head excluded from FP4 only under qmm packing rules;
        # estimate: all g64-divisible 2-D tensors -> FP4, rest stays INT8
        console.print(f"  Dense: {len(dense_meta)} tensors -> ~{fp4_bytes / 1e9:.2f} GB (FP4)"
                      f" + ~{(total_dense_bytes - fp4_bytes) / 1e9:.2f} GB (INT8)")
    else:
        console.print(f"  Dense: {len(dense_meta)} tensors -> ~{total_dense_bytes / 1e9:.2f} GB (INT8)")
    console.print(f"  Experts: {num_blocks * config.num_experts} "
                  f"-> ~{total_expert_bytes / 1e9:.2f} GB raw INT8 "
                  f"(packed to 4-bit merged FFN in output)")
    console.print()

    # ── Open output, header placeholder ────────────────────────────────────
    out_f = open(out_path, "wb")

    def _write(b: bytes) -> None:
        out_f.write(b)

    _write(b"\x00" * HEADER_SIZE)

    try:
        # ── resolve expert storage mode (v8) ──────────────────────────────
        # "auto": quantized-source GGUFs -> lod1 (output SMALLER than source);
        #         unquantized (BF16/F16) sources -> dual (keeps INT8 tier).
        storage = config.expert_storage
        if storage == "auto":
            src_name = Path(gguf_path).name.upper()
            quantized_src = any(t in src_name for t in
                                ("Q2_", "Q3_", "Q4_", "Q5_", "Q6_", "IQ", "UD-"))
            storage = "lod1" if quantized_src else "dual"
            console.print(f"  Storage mode (auto): {storage}")
        import mlx.core as _mx

        def _pack_qmm(w) -> bytes:
            """Pack a dense weight [out,in] (x@W convention) into the exact MLX
            quantized_matmul layout the loader uses: quantize W.T, group 64,
            bits 4. Layout: <rows u32><cols u32><wq uint32 packed><scales f16>
            <biases f16>."""
            wt = _mx.array(np.ascontiguousarray(w.T, dtype=np.float32))
            wq, scales, biases = _mx.quantize(wt, group_size=64, bits=4)
            _mx.eval(wq, scales, biases)
            rows, cols = int(wt.shape[0]), int(wt.shape[1])
            hdr = struct.pack("<II", rows, cols)
            return (hdr + bytes(memoryview(np.ascontiguousarray(np.array(wq))).cast("B"))
                    + np.asarray(scales, dtype=np.float16).tobytes()
                    + np.asarray(biases, dtype=np.float16).tobytes())

        # ── Pass B: dense section [count][entries][blobs] ─────────────────
        offset_dense = out_f.tell()
        DENSE_QMM_CODE = 4   # v8: payload is a packed 4-bit quantized_matmul blob
        DENSE_FP4_CODE = 5   # v8.1: payload is FP4 (E2M1 nibbles, f32 scales, group 64)

        def _fp4_blob_size(shape) -> int:
            rows, cols = int(shape[0]), int(shape[1])
            return (8 + rows * (cols // 64) * 4
                    + rows * ((cols + 1) // 2))

        def _dense_is_packable(name: str, arr_shape) -> bool:
            """lod1-all packs the LARGE matmul weights; embeddings / LM head /
            norms / biases / conv stay INT8 (correctness + they are small)."""
            if storage != "lod1-all":
                return False
            if ".moe_" in name:
                return False   # per-expert MoE records stay INT8 LOD-0
            if name in ("token_embd.weight", "output.weight", "output_norm.weight"):
                return False
            if name.endswith(".bias") or name.endswith("conv1d.weight"):
                return False
            return len(arr_shape) == 2 and all(d % 64 == 0 for d in arr_shape)

        _write(struct.pack("<I", len(dense_meta)))
        dense_blob_sizes = [_int8_blob_size(s) for _, s, _ in dense_meta]
        dense_codes = [2] * len(dense_meta)
        # dense_fp4: any 2-D tensor whose cols split into g64 groups becomes FP4
        # (embeddings / LM head included -- qmm dequantizes to f32 at load either way)
        if config.dense_fp4:
            for i, (name, shape, _) in enumerate(dense_meta):
                if (dense_codes[i] == 2 and len(shape) == 2
                        and int(shape[1]) % 64 == 0):
                    dense_blob_sizes[i] = _fp4_blob_size(shape)
                    dense_codes[i] = DENSE_FP4_CODE
        blob_cursor = 0
        # pre-pass: decide codes/sizes (packed blobs get their true size)
        for i, (name, shape, _) in enumerate(dense_meta):
            if _dense_is_packable(name, shape):
                rows, cols = int(shape[1]), int(shape[0])   # packed layout is W.T
                bits_bytes = ((cols + 7) // 8) * 4          # uint32 packed, bits=4
                packed = (8 + rows * bits_bytes
                          + rows * (cols // 64) * 2 * 2)    # wq hdr + f16 scales+biases
                dense_blob_sizes[i] = packed
                dense_codes[i] = DENSE_QMM_CODE
        for (name, shape, _), bsize, code in zip(dense_meta, dense_blob_sizes, dense_codes):
            name_b = name.encode("utf-8")
            _write(struct.pack("<H", len(name_b)))
            _write(name_b)
            _write(struct.pack("<B", code))
            _write(struct.pack("<B", len(shape)))
            for d in shape:
                _write(struct.pack("<I", d))
            _write(struct.pack("<QQ", blob_cursor, bsize))
            blob_cursor += bsize
        # blobs
        dense_worst_name = None
        dense_worst_err = 0.0
        fp4_worst = [0.0, ""]
        moe_blob_count = 0
        di = 0
        while di < len(dense_meta):
            name, shape, ti = dense_meta[di][0], dense_meta[di][1], dense_meta[di][2]
            bsize = dense_blob_sizes[di]
            code = dense_codes[di]
            t = reader.tensors[ti]
            if code == 2 and name in moe_slices:
                # True-MoE experts: entries carved from one 3-D *_exps
                # tensor are contiguous -- dequantize the source ONCE and
                # emit every per-expert [in, out] slice from it.
                #
                # Verified against the real Qwen-AgentWorld-35B-A3B GGUF:
                # the on-disk shape is (hidden, ffe, n_exp) for gate/up
                # and (ffe, hidden, n_exp) for down -- n_exp is the LAST
                # axis. The per-expert 2-D slice is therefore taken with
                # [..., e_idx] (last-axis indexing), giving shape
                # (hidden, ffe) for gate/up (= engine-facing [in, out]) or
                # (ffe, hidden) for down. No transpose.
                j = di
                while (j < len(dense_meta)
                       and dense_meta[j][2] == ti
                       and dense_meta[j][0] in moe_slices):
                    j += 1
                arr3 = _dequantize_source_tensor(reader, t)
                if arr3.ndim != 3:
                    raise ValueError(f"expected 3-D expert tensor for {name}, "
                                     f"got shape {arr3.shape}")
                if arr3.shape[2] != config.num_experts:
                    raise ValueError(
                        f"MoE tensor {name} last axis {arr3.shape[2]} != "
                        f"num_experts {config.num_experts} (Qwen3-5 GGUF "
                        f"puts n_exp LAST)")
                for k2 in range(di, j):
                    nm = dense_meta[k2][0]
                    e_idx = moe_slices[nm]
                    mat = np.ascontiguousarray(arr3[..., e_idx])   # (hidden, ffe) for gate/up; (ffe, hidden) for down
                    blob = quantize_int8(mat)
                    assert len(blob) == dense_blob_sizes[k2],                         (nm, len(blob), dense_blob_sizes[k2])
                    _write(blob)
                    moe_blob_count += 1
                    del mat, blob
                del arr3
                di = j
                continue
            arr = _dequantize_source_tensor(reader, t)
            if transpose_weights and arr.ndim == 2:
                arr = arr.T
            if dense_codes[di] == DENSE_QMM_CODE:
                blob = _pack_qmm(arr)
                assert len(blob) == bsize, (name, len(blob), bsize)
                _write(blob)
                del arr, blob
                di += 1
                continue
            if dense_codes[di] == DENSE_FP4_CODE:
                blob = quantize_fp4(arr.reshape(arr.shape[0], -1))
                assert len(blob) == bsize, (name, len(blob), bsize)
                recon = dequantize_fp4(blob).reshape(arr.shape)
                src_norm = float(np.linalg.norm(arr))
                rel = (float(np.linalg.norm(recon - arr)) / src_norm
                       if src_norm > 1e-12 else 0.0)
                if rel > fp4_worst[0]:
                    fp4_worst[0], fp4_worst[1] = rel, name
                _write(blob)
                del arr, blob, recon
                di += 1
                continue
            q = arr.reshape(1, -1) if arr.ndim == 1 else arr
            blob = quantize_int8(q)
            assert len(blob) == bsize, (name, len(blob), bsize)
            # INT8 round-trip fidelity check: dequantize what we just wrote
            # and compare to the source array. Cheap (same tensor already in
            # memory), and directly answers "is INT8 quantization itself
            # introducing large errors" instead of assuming it isn't.
            recon = dequantize_int8(blob).reshape(arr.shape)
            src_norm = float(np.linalg.norm(arr))
            if src_norm > 1e-12:
                rel = float(np.linalg.norm(recon - arr)) / src_norm
                if rel > dense_worst_err:
                    dense_worst_err = rel
                    dense_worst_name = name
            _write(blob)
            del arr, q, blob, recon
            di += 1
        if config.dense_fp4:
            console.print(f"  Dense FP4 round-trip: worst-tensor rel. error "
                          f"{fp4_worst[0]*100:.3f}% ({fp4_worst[1]})")
        console.print(f"  Dense INT8 round-trip: worst-tensor rel. error "
                      f"{dense_worst_err*100:.3f}% ({dense_worst_name})")
        if dense_worst_err > 0.05:
            console.print(f"  [red]WARNING: dense tensor INT8 round-trip error is high -- "
                          f"quantize_int8/dequantize_int8 may be mismatched or a source tensor "
                          f"has extreme outliers breaking the per-row max-abs scale.[/red]")

        # ── Pass C: expert section + classification stats + router ─────────
        offset_experts = out_f.tell()

        lod1_tmp = out_path.with_suffix(".lod1.tmp")
        lod1_file = open(lod1_tmp, "w+b")
        gate_stats: list[np.ndarray] = []
        router = np.zeros((num_blocks, hidden, config.num_experts), dtype=np.float32)
        expert_entries: list[ExpertEntry] = []
        eid = 0
        offsets_cursor = 0

        # Fixed-seed probe vector, reused identically for every block, so
        # the reconstruction-fidelity numbers below are directly comparable
        # block-to-block (same x, only the weights differ).
        _probe_rng = np.random.default_rng(12345)
        _probe_x = _probe_rng.standard_normal(hidden).astype(np.float32)
        recon_report: list[tuple[int, float, float]] = []  # (block, rel_err, dense_norm)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), TimeElapsedColumn(),
            console=console, transient=True,
        ) as progress:
            task = progress.add_task("FFN blocks (dequant -> split -> INT8)",
                                     total=num_blocks - len(moe_blocks))
            lod1_blocks_written = 0
            for b in range(num_blocks):
                if b in moe_blocks:
                    # True-MoE block: routed experts already stored as
                    # per-expert dense records; nothing to chunk/merge here.
                    # BUT we still need to populate the MoE router weights,
                    # otherwise the engine routes every token to experts 0..7
                    # (softmax(zeros) is uniform; argpartition of a flat
                    # array always picks the lowest indices).
                    #
                    # The real MoE router is `blk.{b}.ffn_gate_inp.weight`.
                    # Verified against the real Qwen-AgentWorld-35B-A3B GGUF
                    # (Qwen3-5 family, llama.cpp convention): the on-disk
                    # GGUF shape is [hidden, n_exp] -- same as the engine's
                    # m.moe_router(b) layout -- so we copy it as-is. The
                    # HuggingFace Qwen3_5MoeTopKRouter stores the same
                    # parameter as [n_exp, hidden] (a convention swap) and
                    # consumes it with F.linear(x, W) = x @ W.T, but that is
                    # a HF-side convention only; the GGUF on disk and the
                    # engine both use [hidden, n_exp].
                    #
                    # The ATF router layout is [num_blocks, hidden, n_exp]
                    # (engine reads it at that stride at model.py:1069-1072).
                    _rg = reader.tensors.get(f"blk.{b}.ffn_gate_inp.weight")
                    if _rg is None:
                        raise RuntimeError(
                            f"MoE block {b} has ffn_*_exps but no "
                            f"ffn_gate_inp router (expected "
                            f"blk.{b}.ffn_gate_inp.weight)")
                    _rg_arr = _dequantize_source_tensor(reader, _rg)
                    if _rg_arr.shape != (hidden, config.num_experts):
                        raise RuntimeError(
                            f"unexpected router shape for blk.{b}: "
                            f"{_rg_arr.shape}, expected "
                            f"({hidden}, {config.num_experts})")
                    router[b] = _rg_arr.astype(np.float32)
                    continue
                progress.advance(task)
                lod1_blocks_written += 1
                gi = ffn_by_block[b]
                gate = _dequantize_source_tensor(reader, reader.tensors[gi["gate"]])
                up = _dequantize_source_tensor(reader, reader.tensors[gi["up"]])
                down = _dequantize_source_tensor(reader, reader.tensors[gi["down"]])
                if transpose_weights:
                    gate = gate.T
                    up = up.T
                    down = down.T

                # ── reconstruction-fidelity probe ──────────────────────────
                # Ground truth: dense FFN computed directly from the
                # un-chunked, un-quantized (full f32) source weights.
                x = _probe_x
                g_full = x @ gate
                u_full = x @ up
                inter_full = (g_full / (1.0 + np.exp(-g_full))) * u_full
                y_dense = inter_full @ down

                # Chunked reconstruction: sum of 8 unweighted expert outputs,
                # computed from the ACTUAL INT8-quantized-then-dequantized
                # chunk bytes -- i.e. exactly what the engine will do at
                # inference. This is the real end-to-end test, not a
                # theoretical argument about the math.
                y_chunked = np.zeros(hidden, dtype=np.float32)

                # v8 LOD1: store the MERGED dense FFN pre-packed at 4-bit
                # (quantized_matmul layout). One payload per projection --
                # this is exactly what AtfModel.merge_ffn would produce at
                # runtime, computed here once instead of every process start.
                for proj_w in (gate, up, down):
                    lod1_file.write(_pack_qmm(proj_w))

                skip_int8 = storage in ("lod1", "lod1-all")
                for e in range(config.num_experts):
                    g_chunk = gate[:, e * chunk:(e + 1) * chunk]
                    u_chunk = up[:, e * chunk:(e + 1) * chunk]
                    d_chunk = down[e * chunk:(e + 1) * chunk, :]
                    if skip_int8:
                        blob = b""
                        _write(b"")
                    else:
                        g = quantize_int8(g_chunk)
                        u = quantize_int8(u_chunk)
                        d = quantize_int8(d_chunk)
                        blob = g + u + d
                        _write(blob)
                    expert_entries.append(ExpertEntry(
                        expert_id=eid, block_index=b, domain_id=0, sub_domain_id=0,
                        specificity=Specificity.GENERAL, lod_level=LodLevel.LOD_0,
                        weight_offset=offsets_cursor, weight_size=len(blob),
                    ))
                    offsets_cursor += len(blob)
                    eid += 1

                    # dequantize back exactly like AtfModel.w() does, and
                    # accumulate this expert's contribution
                    if not skip_int8:
                        g_dq = dequantize_int8(g)
                        u_dq = dequantize_int8(u)
                        d_dq = dequantize_int8(d)
                        g_e = x @ g_dq
                        u_e = x @ u_dq
                        inter_e = (g_e / (1.0 + np.exp(-g_e))) * u_e
                        y_chunked += inter_e @ d_dq

                dense_norm = float(np.linalg.norm(y_dense))
                err_norm = float(np.linalg.norm(y_dense - y_chunked))
                rel_err = err_norm / dense_norm if dense_norm > 1e-12 else float("nan")
                recon_report.append((b, rel_err, dense_norm))

                gate_stats.extend(chunk_stats(gate[:, e * chunk:(e + 1) * chunk])
                                  for e in range(config.num_experts))
                for e in range(config.num_experts):
                    cm = gate[:, e * chunk:(e + 1) * chunk].mean(axis=1).astype(np.float32)
                    router[b, :, e] = cm / (np.linalg.norm(cm) + 1e-8)
                del gate, up, down

        # ── print the reconstruction-fidelity report ──────────────────────
        from rich.table import Table
        rt = Table(title="FFN reconstruction fidelity (sum-of-8-experts vs dense, probe vector)")
        rt.add_column("blk", justify="right")
        rt.add_column("dense ||y||", justify="right")
        rt.add_column("rel. error", justify="right")
        rt.add_column("flag")
        worst = 0.0
        for b, rel_err, dense_norm in recon_report:
            flag = ""
            if rel_err != rel_err:  # NaN
                flag = "[red]NaN[/red]"
            elif rel_err > 0.05:
                flag = "[red]>5%[/red]"
            elif rel_err > 0.01:
                flag = "[yellow]>1%[/yellow]"
            rt.add_row(str(b), f"{dense_norm:.3f}", f"{rel_err*100:.3f}%", flag)
            if rel_err == rel_err:
                worst = max(worst, rel_err)
        console.print(rt)
        if worst > 0.05:
            console.print(f"  [red]WARNING: worst-block reconstruction error {worst*100:.2f}% "
                          f"-- the sum-of-experts FFN reconstruction is NOT exact for this model. "
                          f"This alone can explain garbage output.[/red]")
        else:
            console.print(f"  [green]FFN reconstruction OK[/green] (worst-block error {worst*100:.4f}%) "
                          f"-- the sum-of-8-unweighted-experts trick is numerically exact here; "
                          f"the bug is NOT in FFN chunking/quantization.")

        # ── Pass D: classification + manifest + index + router ─────────────
        console.print("  Classifying experts...")
        if gate_stats:
            assignments = classify_from_stats(gate_stats, n_domains=len(DOMAIN_NAMES))
        else:
            # pure-MoE model: every FFN block is routed; there are no
            # chunk-expert statistics to bucket.
            assignments = []
        del gate_stats
        for e, (did, sid) in zip(expert_entries, assignments):
            e.domain_id, e.sub_domain_id = did, sid

        offset_manifest = out_f.tell()
        manifest = LayerManifest(
            num_blocks=num_blocks,
            num_experts_per_block=config.num_experts,
            experts=expert_entries,
            domain_names=list(DOMAIN_NAMES),
            sub_domain_names=list(SUB_DOMAIN_NAMES),
        )
        _write(_pack_manifest(manifest))

        offset_index = out_f.tell()
        index = KnowledgeIndex(domains={
            DOMAIN_NAMES[d]: {
                SUB_DOMAIN_NAMES[s]: [e.expert_id for e in expert_entries
                                     if e.domain_id == d and e.sub_domain_id == s]
                for s in range(len(SUB_DOMAIN_NAMES))
            }
            for d in range(len(DOMAIN_NAMES))
        })
        import json
        serialized = json.dumps({"domains": index.domains, "centroids": {}}).encode("utf-8")
        _write(struct.pack("<I", len(serialized)))
        _write(serialized)

        offset_router = out_f.tell()
        _write(struct.pack("<I", len(router.tobytes())))
        _write(router.tobytes())

        # ── v8 LOD1 section: pre-packed merged FFN per block ──────────────
        # Layout: magic "LOD1" | u32 num_blocks | per block: u32 len | 3
        # packed tensors (gate, up, down as merged dense, qmm layout).
        has_lod1 = storage in ("dual", "lod1", "lod1-all")
        if has_lod1:
            console.print("  Writing LOD1 packed-FFN section...")
            lod1_file.flush()
            import os as _os
            lod1_size = lod1_file.tell()
            lod1_file.seek(0)
            body = struct.pack("<4sI", b"LOD1", lod1_blocks_written)
            _write(body)
            remaining = lod1_size
            CH = 1 << 24
            while remaining:
                chunkb = lod1_file.read(min(CH, remaining))
                if not chunkb:
                    raise RuntimeError("LOD1 temp file truncated")
                _write(chunkb)
                remaining -= len(chunkb)
            lod1_file.close()
            lod1_tmp.unlink(missing_ok=True)
        else:
            lod1_tmp.unlink(missing_ok=True)
        out_f.flush()

        # ── Footer + header patch ──────────────────────────────────────────
        console.print("  Hashing file (SHA-256)...")
        out_f.close()
        import hashlib
        size_no_footer = out_path.stat().st_size
        header = Header(
            arch=ArchType.OTHER,
            version_major=2 if has_lod1 else 1,
            version_minor=2 if moe_blocks else 0,   # v19.2: true-MoE + wide expert field
            num_blocks=num_blocks,
            hidden_dim=hidden,
            rope_dim=rope_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv,
            vocab_size=vocab,
            num_experts_per_block=config.num_experts,
            top_k=config.top_k,
            num_lod_levels=2,
            flags=0x03,
            offset_manifest=offset_manifest,
            offset_dense=offset_dense,
            offset_experts=offset_experts,
            offset_index=offset_index,
            offset_router=offset_router,
            file_size=size_no_footer + FOOTER_SIZE,
        )
        with open(out_path, "r+b") as f:
            f.seek(0)
            f.write(header.pack())
            h = hashlib.sha256()
            f.seek(0)
            while chunk_bytes := f.read(1 << 24):
                h.update(chunk_bytes)
            f.seek(size_no_footer)
            f.write(h.digest())
    except BaseException:
        out_f.close()
        out_path.unlink(missing_ok=True)
        try:
            lod1_file.close()
        except Exception:
            pass
        Path(str(out_path)[:-4] + ".lod1.tmp").unlink(missing_ok=True)
        raise

    final_size = out_path.stat().st_size
    duration = time.time() - start
    console.rule()
    console.print(f"  [green]Done in {duration / 60:.1f} min[/green]")
    console.print(f"  Output: {final_size / 1e9:.3f} GB (original: {gguf_path.stat().st_size / 1e9:.3f} GB)")
    console.print(f"  Ratio: {final_size / gguf_path.stat().st_size:.2f}x")

    return ConvertResult(
        output_path=out_path,
        num_experts=len(expert_entries),
        num_blocks=num_blocks,
        total_size=final_size,
        original_size=gguf_path.stat().st_size,
        duration=duration,
    )


def _gguf_tensor_size(t) -> int:
    """Return the exact raw payload size exposed by GGUFReader."""
    data = getattr(t, "data", None)
    return len(bytes(data)) if data is not None else int(t.n_bytes)


def _pack_raw_records(records: list[tuple[str, tuple[int, ...], int, bytes]]) -> bytes:
    """Pack source tensors as metadata followed by their untouched payloads."""
    manifest = bytearray(struct.pack("<I", len(records)))
    blobs = bytearray()
    for name, shape, tensor_type, data in records:
        name_b = name.encode("utf-8")
        manifest.extend(struct.pack("<H", len(name_b)))
        manifest.extend(name_b)
        manifest.extend(struct.pack("<B", tensor_type))
        manifest.extend(struct.pack("<B", len(shape)))
        for dim in shape:
            manifest.extend(struct.pack("<I", dim))
        manifest.extend(struct.pack("<QQ", len(blobs), len(data)))
        blobs.extend(data)
    return bytes(manifest + blobs)


def _raw_record_header_size(name: str, shape: tuple[int, ...]) -> int:
    return 2 + len(name.encode("utf-8")) + 2 + 4 * len(shape) + 16


def _pack_raw_headers(records: list[tuple[str, tuple[int, ...], int, int, int]]) -> bytes:
    """Pack raw tensor metadata without retaining any tensor payloads."""
    manifest = bytearray(struct.pack("<I", len(records)))
    blob_cursor = 0
    for name, shape, tensor_type, nbytes, _ in records:
        name_b = name.encode("utf-8")
        manifest.extend(struct.pack("<H", len(name_b)))
        manifest.extend(name_b)
        manifest.extend(struct.pack("<B", tensor_type))
        manifest.extend(struct.pack("<B", len(shape)))
        for dim in shape:
            manifest.extend(struct.pack("<I", dim))
        manifest.extend(struct.pack("<QQ", blob_cursor, nbytes))
        blob_cursor += nbytes
    return bytes(manifest)


def convert_raw(config: ConvertConfig) -> ConvertResult:
    start = time.time()
    gguf_path = Path(config.gguf_path)
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]ATF Conversion")
    console.print(f"  Input:   {gguf_path.name} ({gguf_path.stat().st_size / 1e9:.2f} GB)")
    console.print(f"  Output:  {out_path}")
    console.print(f"  Experts/block: {config.num_experts}, top_k: {config.top_k}")
    console.print("  Storage:   original GGUF tensor types and raw bytes")
    console.print()

    from gguf import GGUFReader
    reader = GGUFReader(str(gguf_path))

    # ── Pass A: fields + metadata ──────────────────────────────────────────
    arch = _get_str(reader, "general.architecture")
    prefix = arch
    declared_blocks = int(_get_val(reader, f"{prefix}.block_count"))
    # Qwen3.5 IQ4_NL files commonly append one MTP block.  The base engine
    # executes only the trunk; blk.N with nextn.eh_proj is not a base block.
    base_blocks = [
        int(m.group(1)) for t in reader.tensors
        if (m := re.match(r"blk\.(\d+)\.nextn\.eh_proj\.weight$", t.name))
    ]
    num_blocks = min(declared_blocks, min(base_blocks, default=declared_blocks))
    hidden = int(_get_val(reader, f"{prefix}.embedding_length"))
    ffn_dim, moe_note = _resolve_moe_metadata(reader, prefix, config)
    num_heads = int(_get_val(reader, f"{prefix}.attention.head_count"))
    num_kv = int(_get_val(reader, f"{prefix}.attention.head_count_kv"))
    vocab, _ = _get_vocab(reader, hidden)
    moe_blocks = _moe_blocks_of(reader, num_blocks)

    # RoPE rotation dimension. Most models use full head_dim, but Qwen3 / Qwen3.5
    # only rotate the first `rope.dimension_count` dims (with mRoPE sections).
    # Default to head_dim when absent so older / non-RoPE models still work.
    head_dim = hidden // num_heads
    try:
        rope_dim = int(_get_val(reader, f"{prefix}.rope.dimension_count"))
    except (ValueError, KeyError):
        rope_dim = head_dim
    if rope_dim <= 0 or rope_dim > head_dim:
        rope_dim = head_dim

    console.print(f"  {arch}: {num_blocks} blocks, hidden={hidden}, ffn={ffn_dim}, "
                  f"heads={num_heads} (kv {num_kv}), vocab={vocab}, rope_dim={rope_dim}")
    if moe_note:
        console.print(f"  [cyan]MoE metadata: {moe_note}[/cyan]")
    if moe_blocks:
        console.print(f"  [cyan]True-MoE model: {len(moe_blocks)} routed-expert "
                      f"blocks (experts={config.num_experts}, top_k={config.top_k}); "
                      "raw mode keeps the original *_exps tensors[/cyan]")
    console.print()

    # gamma/v1: same MTP-rename as the legacy /v1 (dense) path above.
    # In raw mode, the engine looks up dense weights by name through
    # model.w(); we use the same mtp.* namespace so the engine's MTP
    # forward can find the drafter weights via the same key the
    # converter emitted. model.py's raw-mode loader just does
    # dict-lookup on the (renamed) name, so this works without any
    # further change in model.py.
    mtp_re_raw = re.compile(r"^blk\.(\d+)\.nextn\.(eh_proj|enorm|hnorm|shared_head_head|shared_head_norm)\.weight$")
    mtp_block_ids_raw: list[int] = []
    for i, t in enumerate(reader.tensors):
        m_mtp_r = mtp_re_raw.match(t.name)
        if m_mtp_r:
            mtp_block_ids_raw.append(int(m_mtp_r.group(1)))
    mtp_block_ids_raw = sorted(set(mtp_block_ids_raw))
    mtp_slot_of_raw: dict[int, int] = {b: k for k, b in enumerate(mtp_block_ids_raw)}
    raw_records: list[tuple[str, tuple[int, ...], int, int, int]] = []
    source_tensor_bytes = 0
    for i, t in enumerate(reader.tensors):
        m_mtp_r = mtp_re_raw.match(t.name)
        if m_mtp_r:
            b_mtp = int(m_mtp_r.group(1))
            slot = mtp_slot_of_raw[b_mtp]
            field = m_mtp_r.group(2)
            if field in ("shared_head_head", "shared_head_norm"):
                if slot == 0:
                    emit_name = "mtp.shared_head_head.weight"
                else:
                    console.print(f"  [yellow]MTP: dropping duplicate "
                                  f"{t.name} (kept mtp.shared_head_head.weight "
                                  f"from blk.{mtp_block_ids_raw[0]})[/yellow]")
                    continue
            else:
                emit_name = f"mtp.{slot}.{field}.weight"
            nbytes = _gguf_tensor_size(t)
            source_tensor_bytes += nbytes
            shape = tuple(int(s) for s in t.shape)
            raw_records.append((emit_name, shape, int(t.tensor_type), nbytes, i))
            continue
        block_match = re.match(r"blk\.(\d+)\.", t.name)
        if block_match and int(block_match.group(1)) >= num_blocks:
            continue
        nbytes = _gguf_tensor_size(t)
        source_tensor_bytes += nbytes
        shape = tuple(int(s) for s in t.shape)
        raw_records.append((t.name, shape, int(t.tensor_type), nbytes, i))

    raw_header_bytes = len(_pack_raw_headers(raw_records))
    total_atf_bytes = HEADER_SIZE + raw_header_bytes + source_tensor_bytes
    console.print(f"  Source tensors: ~{source_tensor_bytes / 1e9:.2f} GB "
                  f"(actual mixed GGUF quantization; file: "
                  f"{gguf_path.stat().st_size / 1e9:.2f} GB)")
    console.print(f"  Raw tensors: {len(raw_records)} -> ~{total_atf_bytes / 1e9:.2f} GB")
    console.print()

    # ── Open output, header placeholder ────────────────────────────────────
    out_f = open(out_path, "wb")

    def _write(b: bytes) -> None:
        out_f.write(b)

    _write(b"\x00" * HEADER_SIZE)

    try:
        # ── Pass B: raw tensor section [count][entries][blobs] ────────────
        offset_dense = out_f.tell()
        _write(_pack_raw_headers(raw_records))
        for _, _, _, _, tensor_index in raw_records:
            _write(bytes(reader.tensors[tensor_index].data))

        # ── Pass C: expert section + classification stats + router ─────────
        offset_experts = out_f.tell()
        router = np.zeros((num_blocks, hidden, config.num_experts), dtype=np.float32)
        # MoE-router discovery. Verified against the real Qwen-AgentWorld-
        # 35B-A3B GGUF: the on-disk shape of `blk.{b}.ffn_gate_inp.weight`
        # is [hidden, n_exp] (llama.cpp convention; Qwen3-5 family), which
        # matches the engine's m.moe_router(b) layout. We copy as-is. The
        # HuggingFace Qwen3_5MoeTopKRouter uses the opposite convention
        # [n_exp, hidden] internally and consumes it with F.linear(x, W) =
        # x @ W.T, but that is a HF-side convention only; the GGUF on
        # disk and the engine both use [hidden, n_exp].
        #
        # The ATF router layout is [num_blocks, hidden, n_exp] (engine
        # reads it at that stride at model.py:1069-1072). Without this,
        # the on-disk router is all zeros and the engine deterministically
        # routes every token to expert indices 0..top_k-1 (softmax(zeros)
        # is uniform; argpartition of a flat array picks the lowest
        # indices).
        for b in sorted(moe_blocks):
            _rg = reader.tensors.get(f"blk.{b}.ffn_gate_inp.weight")
            if _rg is None:
                raise RuntimeError(
                    f"MoE block {b} has ffn_*_exps but no ffn_gate_inp "
                    f"router (expected blk.{b}.ffn_gate_inp.weight)")
            _rg_arr = _dequantize_source_tensor(reader, _rg)
            if _rg_arr.shape != (hidden, config.num_experts):
                raise RuntimeError(
                    f"unexpected router shape for blk.{b}: {_rg_arr.shape}, "
                    f"expected ({hidden}, {config.num_experts})")
            router[b] = _rg_arr.astype(np.float32)
        expert_entries: list[ExpertEntry] = []
        console.print("  Expert section empty: exact raw quantized matrices cannot be split safely")

        # ── Pass D: classification + manifest + index + router ─────────────
        console.print("  Classifying experts...")
        offset_manifest = out_f.tell()
        manifest = LayerManifest(
            num_blocks=num_blocks,
            num_experts_per_block=config.num_experts,
            experts=expert_entries,
            domain_names=list(DOMAIN_NAMES),
            sub_domain_names=list(SUB_DOMAIN_NAMES),
        )
        _write(_pack_manifest(manifest))

        offset_index = out_f.tell()
        index = KnowledgeIndex(domains={
            DOMAIN_NAMES[d]: {
                SUB_DOMAIN_NAMES[s]: [e.expert_id for e in expert_entries
                                     if e.domain_id == d and e.sub_domain_id == s]
                for s in range(len(SUB_DOMAIN_NAMES))
            }
            for d in range(len(DOMAIN_NAMES))
        })
        import json
        serialized = json.dumps({"domains": index.domains, "centroids": {}}).encode("utf-8")
        _write(struct.pack("<I", len(serialized)))
        _write(serialized)

        offset_router = out_f.tell()
        _write(struct.pack("<I", len(router.tobytes())))
        _write(router.tobytes())
        out_f.flush()

        # ── Footer + header patch ──────────────────────────────────────────
        console.print("  Hashing file (SHA-256)...")
        out_f.close()
        import hashlib
        size_no_footer = out_path.stat().st_size
        header = Header(
            arch=ArchType.OTHER,
            version_minor=2 if moe_blocks else 0,   # v19.2: true-MoE + wide expert field
            num_blocks=num_blocks,
            hidden_dim=hidden,
            rope_dim=rope_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv,
            vocab_size=vocab,
            num_experts_per_block=config.num_experts,
            top_k=config.top_k,
            num_lod_levels=2,
            flags=0x07,  # router, index, and raw GGUF tensor records
            offset_manifest=offset_manifest,
            offset_dense=offset_dense,
            offset_experts=offset_experts,
            offset_index=offset_index,
            offset_router=offset_router,
            file_size=size_no_footer + FOOTER_SIZE,
        )
        with open(out_path, "r+b") as f:
            f.seek(0)
            f.write(header.pack())
            h = hashlib.sha256()
            f.seek(0)
            while chunk_bytes := f.read(1 << 24):
                h.update(chunk_bytes)
            f.seek(size_no_footer)
            f.write(h.digest())
    except BaseException:
        out_f.close()
        out_path.unlink(missing_ok=True)
        raise

    final_size = out_path.stat().st_size
    duration = time.time() - start
    console.rule()
    console.print(f"  [green]Done in {duration / 60:.1f} min[/green]")
    console.print(f"  Output: {final_size / 1e9:.3f} GB (original: {gguf_path.stat().st_size / 1e9:.3f} GB)")
    console.print(f"  Ratio: {final_size / gguf_path.stat().st_size:.2f}x")

    return ConvertResult(
        output_path=out_path,
        num_experts=len(expert_entries),
        num_blocks=num_blocks,
        total_size=final_size,
        original_size=gguf_path.stat().st_size,
        duration=duration,
    )


def _pack_manifest(m: LayerManifest) -> bytes:
    buf = bytearray()
    buf.extend(struct.pack("<HH", m.num_blocks, m.num_experts_per_block))
    buf.extend(struct.pack("<H", len(m.domain_names)))
    for name in m.domain_names:
        b = name.encode("utf-8")
        buf.extend(struct.pack("<H", len(b)))
        buf.extend(b)
    buf.extend(struct.pack("<H", len(m.sub_domain_names)))
    for name in m.sub_domain_names:
        b = name.encode("utf-8")
        buf.extend(struct.pack("<H", len(b)))
        buf.extend(b)
    buf.extend(struct.pack("<I", len(m.experts)))
    for e in m.experts:
        buf.extend(struct.pack(
            "<QBBBBBQQQI",
            e.expert_id, e.block_index, e.domain_id, e.sub_domain_id,
            int(e.specificity), int(e.lod_level),
            e.weight_offset, e.weight_size, e.centroid_offset, e.centroid_dim,
        ))
    return bytes(buf)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_val(reader, name: str):
    f = reader.get_field(name)
    if f is None:
        raise ValueError(f"Field not found: {name}")
    val = f.parts[-1]
    if isinstance(val, (list, np.ndarray)):
        return int(val[0])
    return int(val)


def _get_str(reader, name: str) -> str:
    f = reader.get_field(name)
    if f is None:
        raise ValueError(f"Field not found: {name}")
    raw = f.parts[-1]
    if isinstance(raw, (list, np.ndarray)):
        return bytes(np.array(raw, dtype=np.uint8)).decode("utf-8")
    return str(raw)


def _get_opt(reader, name: str, default=None):
    """Optional integer metadata field (returns `default` when absent)."""
    if reader.get_field(name) is None:
        return default
    try:
        return int(_get_val(reader, name))
    except (ValueError, KeyError):
        return default


def _moe_blocks_of(reader, num_blocks: int) -> set[int]:
    """Blocks that carry routed-expert tensors (`ffn_*_exps`)."""
    return {
        int(m.group(1)) for t in reader.tensors
        if (m := re.match(r"blk\.(\d+)\.ffn_gate_exps\.weight$", t.name))
        and int(m.group(1)) < num_blocks
    }


def _resolve_moe_metadata(reader, prefix: str, config: "ConvertConfig") -> tuple[int, str]:
    """Resolve ffn_dim with expert-FFN fallback + MoE metadata overrides.

    Returns (ffn_dim, note). Models like Qwen3.6-35B-A3B have NO plain
    `{arch}.feed_forward_length` key -- fall back to
    `{arch}.expert_feed_forward_length`. When expert_count/expert_used_count
    are present they OVERRIDE the config defaults for num_experts/top_k.
    """
    notes = []
    ffn_dim = _get_opt(reader, f"{prefix}.feed_forward_length")
    if ffn_dim is None:
        ffn_dim = _get_opt(
            reader, f"{prefix}.expert_feed_forward_length")
        if ffn_dim is None:
            raise ValueError(
                f"Field not found: {prefix}.feed_forward_length "
                f"(or {prefix}.expert_feed_forward_length)")
        notes.append("feed_forward_length absent -> using "
                     f"{prefix}.expert_feed_forward_length={ffn_dim}")
    ec = _get_opt(reader, f"{prefix}.expert_count")
    eu = _get_opt(reader, f"{prefix}.expert_used_count")
    if ec is not None:
        config.num_experts = ec
        notes.append(f"expert_count={ec}")
    if eu is not None:
        config.top_k = eu
        notes.append(f"expert_used_count={eu} (top_k)")
    return int(ffn_dim), "; ".join(notes)


def _get_vocab(reader, hidden: int) -> tuple[int, bool]:
    # token_embd last dim is the ground-truth vocab size (the GGUF tokens
    # field is a nested array that does not parse to a scalar reliably).
    for t in reader.tensors:
        if "token_embd" in t.name:
            native = tuple(int(s) for s in t.shape)
            if len(native) == 2 and native[0] == hidden:
                return native[1], False
            if len(native) == 2 and native[1] == hidden:
                return native[0], True
            break
    f = reader.get_field("tokenizer.ggml.tokens")
    if f is not None:
        try:
            return int(np.asarray(f.parts[0]).reshape(-1)[0]), False
        except Exception:
            pass
    return 32000, False
