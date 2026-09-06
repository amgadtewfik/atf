"""
ATF -> Ngram-augmented converter (Qwen3.8-Flash-Next lookup-table trick).

This is the FAST alternative to `convert_moe.py`. The motivation:

  * The current `convert_moe.py` adds a MoE scaffold (router + shared expert)
    that the engine routes through `_ffn_moe` -- which at decode time does
    `n_exp x top_k` separate per-expert matmuls + a Python-level grouping
    loop. On the 27B that costs ~230 ms / block (~9 s / token, see
    `versions/beta/v7/MOE_DECODE_SPEED_FIX.md`).
  * The Qwen3.8-Flash-Next insight from the model card is that an n-gram
    lookup table (51B params, indexed by token-pair / token-triple) is a
    strictly faster form of "knowledge scaling" than MoE:
      - one `mx.take` + one tiny `[ngram_dim, hidden]` matmul per token
      - no per-expert dispatch
      - table can be huge and live at LOD-1/int4 off-GPU
      - LRU-paged hot rows

  See docs/qwen38_flash_next_analysis.md sections 3.1, 6.5, 6.6.

What this converter does (recipe N -- "Fast MoE-via-Ngram"):

  1. Copy the source's dense records VERBATIM. The FFN stays dense
     (no per-expert slicing, no `blk.N.ffn_*_weight.e{e}` records).
  2. Skip the MoE scaffold entirely (no router, no `ffn_gate_inp*`,
     no shared expert). Engine routes the FFN through the existing
     dense SwiGLU path (3 matmuls per block, not 24+).
  3. Add 3 ngram records into the dense section:
       ngram_emb.weight          [ngram_vocab, ngram_dim]   F16
       ngram_emb_proj.weight     [ngram_dim, hidden]        F16
       ngram_emb_norm.weight     [ngram_dim]                F16
  4. Bump the header to version_minor=3 and populate the ngram_* fields
     so Engine.__init__ builds self.ngram and _forward_block dispatches
     through _ngram_insert at layer `ngram_insert_layer`.

Decode-step cost (per block):

  * Dense SwiGLU path: 3 matmuls per block (`qmm_fuse(gate|up)` +
    `qmm(down)`). Same as the non-MoE path -- unchanged.
  * At layer ngram_insert_layer (default: layer 2): 1 `mx.take` over the
    ngram table + 1 RMSNorm + 1 `[ngram_dim, hidden]` matmul. The
    `[ngram_dim, hidden]` matmul is tiny (ngram_dim ~ 64-256,
    hidden ~ 4096-5120) -- one GEMV at decode time.
  * Total per token: dense FFN cost (3 matmuls) + ngram cost (1 matmul).
  * No `mx.argpartition`, no per-expert Python loop, no shared-expert
    sigmoid gate.

File-size delta:

  * Source: ~10 GB (27B at Q2_K_XL)
  * + ngram table: ngram_vocab * ngram_dim * 2 bytes
        default (vocab=2_000_000, dim=64): 256 MB
        Qwen4 spec (vocab=20_000_000, dim=64): 2.5 GB
        (table is LRU-paged at load; only hot bigrams stay resident)
  * + ngram_emb_proj: ngram_dim * hidden * 2 bytes
        default (dim=64, hidden=5120): 640 KB
        (resident in LRU; trivial)
  * + ngram_emb_norm: ngram_dim * 2 bytes
        default: 128 bytes
  * + tiny scaffold: ~50 KB

Quality note (read this before deploying):

  This converter does NOT train the ngram table. The init is small
  Gaussian (proj) + small Gaussian (table) + ones (norm). The output of
  a model produced by this converter will be NEAR-random until the
  ngram weights are filled with real values. That is the same caveat
  the current `convert_moe.py` has about its router (recipe A caveat,
  top of convert_moe.py).

  The conversion is intended as an infrastructure PR:
    - proves the format path (raw pass-through + ngram header + ngram
      dense records) round-trips correctly
    - proves the engine routes through _ngram_insert
    - gives the bench a fast file to time against the MoE file
  Continued pre-training (or distillation of the source's logits into
  the ngram table) is a separate, larger work item.

  Idempotent-init contract: the ngram `proj` matrix is zero-init.
  The resulting model is therefore byte-identical to the source
  on the very first token (the ngram residual insertion is a
  perfect no-op), and only diverges once the user fills `proj`
  (and/or the table) with real values. This is the same
  convention the gamma/v2 zero-ngram parity test pins; see
  tests/test_ngram_parity.py::test_zero_ngram_tables_produce_no_change_to_x.
"""
from __future__ import annotations

import argparse, hashlib, json, os, struct, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from .format import (FOOTER_SIZE, HEADER_SIZE, HEADER_SIZE_V3, ArchType,
                     ExpertEntry, Header, LayerManifest, LodLevel,
                     Specificity)
from .taxonomy import DOMAIN_NAMES, SUB_DOMAIN_NAMES
from .model import _read_header, _read_manifest

console = Console()


# GGUF dtype codes (must match atf.gguf_io DEQUANTIZERS / GGMLQuantizationType).
# F16 is the standard GGUF 1; the raw loader dequantizes it via the gguf
# package fallback in gguf_io.dequantize.
GGUF_F16 = 1
GGUF_F32 = 0
GGUF_BF16 = 30


def _read_dense_table(mmap, offset: int):
    """Read the source's dense-table manifest. Same byte format as
    AtfReader.read_dense; only the manifest walk is needed for
    pass-through (we never dequantize the source records -- the
    output copies them byte-for-byte)."""
    pos = offset
    (count,) = struct.unpack_from("<I", mmap, pos); pos += 4
    entries = []
    for _ in range(count):
        (nlen,) = struct.unpack_from("<H", mmap, pos); pos += 2
        name = mmap[pos:pos + nlen].decode("utf-8"); pos += nlen
        dcode, ndim = struct.unpack_from("<BB", mmap, pos); pos += 2
        shape = struct.unpack_from(f"<{ndim}I", mmap, pos); pos += 4 * ndim
        blob_off, blob_sz = struct.unpack_from("<QQ", mmap, pos); pos += 16
        entries.append((name, tuple(shape), int(dcode), int(ndim),
                        int(blob_off), int(blob_sz)))
    return count, entries, pos


def _default_output_path(src: Path) -> Path:
    base = src.with_name(src.stem + "_ngram.atf")
    if not base.exists():
        return base
    n = 1
    while True:
        cand = src.with_name(f"{src.stem}_ngram_{n}.atf")
        if not cand.exists():
            return cand
        n += 1


def _pack_manifest(m: LayerManifest) -> bytes:
    """Same manifest layout as convert_moe._pack_manifest -- both
    converters share the v2 manifest format."""
    buf = bytearray()
    buf.extend(struct.pack("<HH", m.num_blocks, m.num_experts_per_block))
    buf.extend(struct.pack("<H", len(m.domain_names)))
    for name in m.domain_names:
        b = name.encode("utf-8")
        buf.extend(struct.pack("<H", len(b))); buf.extend(b)
    buf.extend(struct.pack("<H", len(m.sub_domain_names)))
    for name in m.sub_domain_names:
        b = name.encode("utf-8")
        buf.extend(struct.pack("<H", len(b))); buf.extend(b)
    buf.extend(struct.pack("<I", len(m.experts)))
    for e in m.experts:
        buf.extend(struct.pack(
            "<QBBBBBQQQI",
            e.expert_id, e.block_index, e.domain_id, e.sub_domain_id,
            int(e.specificity), int(e.lod_level),
            e.weight_offset, e.weight_size, e.centroid_offset, e.centroid_dim,
        ))
    return bytes(buf)


@dataclass
class NgramConfig:
    src_path: Path
    dst_path: Path
    # Qwen4 spec: vocab=20_000_000 bigrams, dim~64-256. Default to a
    # 2M x 64 table (256 MB at F16) -- enough rows to be a useful
    # lookup, small enough that the convert + load + bench stay fast
    # on a 16 GB mini. Override with --ngram-vocab and --ngram-dim.
    ngram_vocab: int = 2_000_000
    ngram_dim: int = 64
    ngram_insert_layer: int = 2
    ngram_context: int = 2        # 2 = bigram, 3 = trigram
    seed: int = 0
    dry_run: bool = False


def convert_ngram(cfg: NgramConfig) -> dict:
    """Pass-through a raw-GGUF .atf, appending 3 n-gram dense records
    and bumping the header to v3. See module docstring for the recipe."""
    start = time.time()
    src = Path(cfg.src_path)
    dst = Path(cfg.dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"output exists: {dst} (delete first)")

    console.rule("[bold blue]ATF Ngram Conversion (Qwen3.8-Flash-Next "
                 "lookup-table, fast MoE-via-Ngram)")
    console.print(f"  Source: {src.name} ({src.stat().st_size/1e9:.2f} GB)")
    console.print(f"  Output: {dst}")
    console.print(f"  Ngram vocab: {cfg.ngram_vocab:,}  dim: {cfg.ngram_dim}  "
                 f"insert_layer: {cfg.ngram_insert_layer}  "
                 f"context: {cfg.ngram_context}")
    if cfg.dry_run:
        console.print("  [yellow]Dry-run: not writing output[/yellow]")
    console.print()

    # Sanity-check the ngram params against the source so the table
    # we write cannot overflow or under-fill.
    import mmap
    src_f = open(src, "rb")
    src_mm = mmap.mmap(src_f.fileno(), 0, access=mmap.ACCESS_READ)

    header = _read_header(src_f)
    manifest = _read_manifest(src_f, header.offset_manifest)
    nblocks = header.num_blocks
    hidden = header.hidden_dim

    if cfg.ngram_insert_layer < 0 or cfg.ngram_insert_layer >= nblocks:
        raise ValueError(
            f"ngram_insert_layer={cfg.ngram_insert_layer} out of range "
            f"for num_blocks={nblocks}"
        )
    if cfg.ngram_context not in (2, 3):
        raise ValueError(f"ngram_context must be 2 (bigram) or 3 (trigram), "
                         f"got {cfg.ngram_context}")
    if cfg.ngram_vocab < 1024:
        raise ValueError(f"ngram_vocab={cfg.ngram_vocab} too small "
                         f"(minimum 1024 rows)")
    if cfg.ngram_dim < 8 or cfg.ngram_dim > 1024:
        raise ValueError(f"ngram_dim={cfg.ngram_dim} out of range "
                         f"(8-1024 recommended)")

    src_count, src_entries, src_blob_base = _read_dense_table(
        src_mm, header.offset_dense)
    has_ngram = any(name in ("ngram_emb.weight", "ngram_emb_proj.weight",
                             "ngram_emb_norm.weight")
                    for name, *_ in src_entries)
    if has_ngram:
        src_f.close(); src_mm.close()
        raise RuntimeError("source already has ngram tensors -- ngram file")

    ngram_table_bytes = cfg.ngram_vocab * cfg.ngram_dim * 2   # F16
    ngram_proj_bytes = cfg.ngram_dim * hidden * 2              # F16
    ngram_norm_bytes = cfg.ngram_dim * 2                       # F16
    ngram_total_bytes = ngram_table_bytes + ngram_proj_bytes + ngram_norm_bytes

    if cfg.dry_run:
        t = Table(title=f"DRY-RUN: {dst.name}")
        t.add_column("Field", style="cyan"); t.add_column("Value")
        t.add_row("Source", f"{src.name} ({src.stat().st_size/1e9:.2f} GB)")
        t.add_row("Output (planned)", dst.name)
        t.add_row("Source records (copied verbatim)", str(src_count))
        t.add_row("Ngram records (added)", "3")
        t.add_row("Total out records", str(src_count + 3))
        t.add_row("Blocks", str(nblocks))
        t.add_row("Hidden", str(hidden))
        t.add_row("Ngram vocab", f"{cfg.ngram_vocab:,}")
        t.add_row("Ngram dim", str(cfg.ngram_dim))
        t.add_row("Ngram context", f"{cfg.ngram_context}-gram")
        t.add_row("Ngram insert layer", str(cfg.ngram_insert_layer))
        t.add_row("ngram_emb.weight bytes",
                  f"{ngram_table_bytes/1e6:.1f} MB")
        t.add_row("ngram_emb_proj.weight bytes",
                  f"{ngram_proj_bytes/1e6:.1f} MB")
        t.add_row("ngram_emb_norm.weight bytes",
                  f"{ngram_norm_bytes/1e6:.1f} MB")
        t.add_row("Total ngram payload",
                  f"{ngram_total_bytes/1e6:.1f} MB")
        t.add_row("Estimated output size",
                  f"~{src.stat().st_size/1e9:.2f} GB + "
                  f"{ngram_total_bytes/1e6:.1f} MB")
        t.add_row("Decode-step ngram cost",
                  "1 take + 1 rmsnorm + 1 matmul [ngram_dim, hidden]")
        t.add_row("FFN path", "dense SwiGLU (3 matmuls/block) "
                  "-- no per-expert dispatch")
        console.print(t)
        src_f.close(); src_mm.close()
        return {"src": str(src), "dst": str(dst), "dry_run": True,
                "duration_s": time.time() - start,
                "ngram_vocab": cfg.ngram_vocab,
                "ngram_dim": cfg.ngram_dim}

    # Plan: copy every source dense record verbatim + append 3 ngram
    # records. The dense table layout is (count u32) + per-record
    # (name u16-bytes, dcode u8, ndim u8, shape u32*ndim, blob_off u64,
    # blob_sz u64) followed by the raw blob bytes.
    out_entries = []
    in_mem = []
    slice_refs = []

    def _add(name, shape, dcode, payload, src_ref=None):
        if payload is not None:
            out_entries.append((name, shape, dcode, -1, len(payload)))
            in_mem.append(payload)
        else:
            assert src_ref is not None
            out_entries.append((name, shape, dcode, -1, src_ref[1]))
            in_mem.append(None)
            slice_refs.append((len(out_entries) - 1, src_ref[0], src_ref[1]))

    # 1. Copy source dense records verbatim
    for name, shape, dcode, ndim, blob_off, blob_sz in src_entries:
        out_entries.append((name, shape, dcode, -1, blob_sz))
        in_mem.append(None)
        slice_refs.append((len(out_entries) - 1,
                            src_blob_base + blob_off, blob_sz))

    # 2. Ngram records. The init here is chosen so the resulting
    # model is BYTE-IDENTICAL to the source until the user fills
    # the ngram table with real values (continued pre-training).
    #
    # Math: at every layer the engine computes
    #     ngram_act = rmsnorm(table[pair_id]) @ proj
    # and adds it to x. If `proj` is all zeros, the contribution
    # is zero regardless of the table content -- the ngram path
    # is a perfect no-op. This is the same convention the
    # gamma/v2 zero-ngram parity test pins:
    # tests/test_ngram_parity.py::test_zero_ngram_tables_produce_no_change_to_x.
    #
    # The `table` is small Gaussian so that once the user fills
    # `proj` with real values, the path is immediately usable
    # (the table has non-zero gradient signal). `norm` is ones
    # (default RMSNorm init).
    rng = np.random.default_rng(cfg.seed)
    table = (rng.standard_normal(
                (cfg.ngram_vocab, cfg.ngram_dim)).astype(np.float32) * 0.02)
    proj = np.zeros((cfg.ngram_dim, hidden), dtype=np.float32)
    norm = np.ones((cfg.ngram_dim,), dtype=np.float32)

    _add("ngram_emb.weight",
         (cfg.ngram_vocab, cfg.ngram_dim), GGUF_F16,
         table.astype(np.float16).tobytes())
    _add("ngram_emb_proj.weight",
         (cfg.ngram_dim, hidden), GGUF_F16,
         proj.astype(np.float16).tobytes())
    _add("ngram_emb_norm.weight",
         (cfg.ngram_dim,), GGUF_F16,
         norm.astype(np.float16).tobytes())

    # Build dense table bytes (manifest only -- blob bytes are
    # appended below, one per entry, in order).
    table_buf = bytearray()
    table_buf.extend(struct.pack("<I", len(out_entries)))
    for name, shape, dcode, _, blob_sz in out_entries:
        nb = name.encode("utf-8")
        table_buf.extend(struct.pack("<H", len(nb))); table_buf.extend(nb)
        table_buf.extend(struct.pack("<B", dcode))
        table_buf.extend(struct.pack("<B", len(shape)))
        for d in shape:
            table_buf.extend(struct.pack("<I", d))
        table_buf.extend(struct.pack("<QQ", 0, blob_sz))
    # Patch in the per-entry blob offsets (the manifest walker in
    # AtfReader expects blob_off to be the cumulative byte offset of
    # this entry's payload within the blob, NOT the absolute file
    # offset -- see atf/format.py::AtfReader.read_dense).
    cur = 0
    patch_pos = 4
    for i, (name, shape, dcode, _, blob_sz) in enumerate(out_entries):
        nb = name.encode("utf-8")
        patch_pos += 2 + len(nb) + 2 + 4 * len(shape)
        struct.pack_into("<Q", table_buf, patch_pos, cur)
        cur += blob_sz
        patch_pos += 16

    console.print(f"  Writing {dst} ...")
    src_ref_by_idx = {idx: (off, ln) for idx, off, ln in slice_refs}
    with open(dst, "wb") as out_f:
        out_f.write(b"\x00" * HEADER_SIZE_V3)
        offset_dense = out_f.tell()
        out_f.write(bytes(table_buf))
        for i, payload in enumerate(in_mem):
            if payload is None:
                src_off, src_len = src_ref_by_idx[i]
                src_mm.seek(src_off)
                chunk_bytes = src_mm.read(src_len)
                if len(chunk_bytes) != src_len:
                    raise RuntimeError(
                        f"short read: {len(chunk_bytes)} vs {src_len}")
                out_f.write(chunk_bytes)
            else:
                out_f.write(payload)

        offset_experts = out_f.tell()
        out_f.write(b"")
        offset_manifest = out_f.tell()
        manifest_out = LayerManifest(
            num_blocks=nblocks,
            num_experts_per_block=0,           # ngram path, no MoE
            experts=[],
            domain_names=list(DOMAIN_NAMES),
            sub_domain_names=list(SUB_DOMAIN_NAMES),
        )
        out_f.write(_pack_manifest(manifest_out))
        offset_index = out_f.tell()
        idx_payload = json.dumps({
            "domains": {d: {s: [] for s in SUB_DOMAIN_NAMES} for d in DOMAIN_NAMES},
            "centroids": {},
        }).encode("utf-8")
        out_f.write(struct.pack("<I", len(idx_payload)))
        out_f.write(idx_payload)
        offset_router = out_f.tell()
        # Router section is zero-bytes: the ngram path doesn't use a
        # router (the per-token "which expert" choice is replaced by
        # the bigram lookup -- no learned routing). The engine's
        # self.num_experts_per_block / self.top_k remain 0, so
        # Engine.__init__ does not register any moe_blocks and
        # _ffn() falls through to the dense SwiGLU path.
        out_f.write(struct.pack("<I", 0))
        size_no_footer = out_f.tell()
        new_header = Header(
            arch=header.arch, version_major=2, version_minor=3,
            num_blocks=nblocks, hidden_dim=hidden,
            rope_dim=header.rope_dim or hidden // max(header.num_heads, 1),
            num_heads=header.num_heads, num_kv_heads=header.num_kv_heads,
            vocab_size=header.vocab_size,
            num_experts_per_block=0, top_k=0,
            num_lod_levels=header.num_lod_levels, flags=0x07,
            ngram_vocab=cfg.ngram_vocab,
            ngram_dim=cfg.ngram_dim,
            ngram_insert_layer=cfg.ngram_insert_layer,
            ngram_context=cfg.ngram_context,
            offset_manifest=offset_manifest, offset_dense=offset_dense,
            offset_experts=offset_experts, offset_index=offset_index,
            offset_router=offset_router,
            file_size=size_no_footer + FOOTER_SIZE,
        )
        out_f.seek(0)
        out_f.write(new_header.pack())
        out_f.flush()
        h = hashlib.sha256()
        with open(dst, "rb") as rf:
            rf.seek(0)
            while True:
                chunk_bytes = rf.read(1 << 24)
                if not chunk_bytes:
                    break
                h.update(chunk_bytes)
        out_f.seek(0, 2)
        out_f.write(h.digest())

    src_f.close()
    src_mm.close()
    duration = time.time() - start
    final_size = dst.stat().st_size
    console.rule()
    console.print(f"  [green]Done in {duration/60:.1f} min[/green]")
    console.print(f"  Output: {final_size/1e9:.3f} GB "
                 f"(source: {src.stat().st_size/1e9:.3f} GB)")
    return {"src": str(src), "dst": str(dst),
            "num_blocks": nblocks,
            "ngram_vocab": cfg.ngram_vocab,
            "ngram_dim": cfg.ngram_dim,
            "ngram_insert_layer": cfg.ngram_insert_layer,
            "ngram_context": cfg.ngram_context,
            "src_bytes": src.stat().st_size,
            "dst_bytes": final_size,
            "duration_s": duration}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atf-convert-ngram",
        description=("Convert a raw-GGUF .atf to a ngram-augmented .atf "
                     "using the Qwen3.8-Flash-Next lookup-table trick. "
                     "Replaces MoE per-expert dispatch with one bigram "
                     "gather + one tiny matmul per token at layer "
                     "ngram_insert_layer."))
    p.add_argument("src", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--ngram-vocab", type=int, default=2_000_000,
                   help="Number of n-gram table rows (default: 2,000,000; "
                        "Qwen4 spec: 20,000,000).")
    p.add_argument("--ngram-dim", type=int, default=64,
                   help="Embedding dim per n-gram entry (default: 64; "
                        "Qwen4 spec: 64-256).")
    p.add_argument("--ngram-insert-layer", type=int, default=2,
                   help="Layer index where the n-gram residual is added "
                        "(default: 2, matching Qwen4).")
    p.add_argument("--ngram-context", type=int, default=2, choices=(2, 3),
                   help="Bigram (2) or trigram (3) -- default: 2.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    src = Path(args.src)
    dst = Path(args.output) if args.output else _default_output_path(src)
    cfg = NgramConfig(src_path=src, dst_path=dst,
                      ngram_vocab=args.ngram_vocab,
                      ngram_dim=args.ngram_dim,
                      ngram_insert_layer=args.ngram_insert_layer,
                      ngram_context=args.ngram_context,
                      seed=args.seed, dry_run=args.dry_run)
    try:
        result = convert_ngram(cfg)
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        return 1
    console.print(f"  Result: {json.dumps(result, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
