"""
ATF -> MoE-ATF converter (v2: aliasing-shared + sliced-shared recipes).

Reads a raw-GGUF .atf file (already quantized) and writes a NEW raw-GGUF .atf
file that the engine routes through `_ffn_moe`. Two recipes are supported:

1. **Aliasing-shared recipe** (`--shexp-alias`): the dense FFN records are
   copied verbatim into the output; the shexp records are zero-cost aliases
   of those bytes. Engine reads them as full-size dense FFN matrices and
   the shared expert computation equals the full dense FFN. Routed experts
   contribute only `(top_k / n_exp)` of a slice (small noise) on top.
   The router is initialised to ZERO so softmax is uniform and argpartition
   is deterministic.

   ⚠ With this recipe the engine calls `m.w(shexp)` which dequantises the
   full dense FFN to f32 (e.g. 357 MB per tensor on a 27B). The 3 GB
   LRU cap can hold only ~8 of the 192 shexp tensors, so each token
   thrashes the cache and the model runs at ~1 token / 20 s. The recipe
   is correct, just very slow.

2. **Sliced-shared recipe** (default, `--shexp-slice`): for each block, the
   dense FFN is dequantised to f32 ONCE during conversion, columns
   [0:ffe_shex] (gate/up) or rows [0:ffe_shex] (down) are sliced, and the
   result is stored as a NEW shexp record (BF16 by default — half the size
   of F32, ~0.7 ms dequant per tensor). Shexp tensors fit in the LRU so
   the model runs at near-dense speed. Routed contribution is the same as
   the aliasing recipe (slices of the dense FFN via the legacy `.e{e}`
   addressing). Output behaviour: small shexp + sparse routed = structured
   approximation of the dense FFN.

Output file size: source + ~6-300 MB scaffold depending on recipe + dtype.

This recipe produces "sensible output without training": the shared expert
carries a real (narrow) FFN computation sampled from the dense FFN's first
`ffe_shex` columns, and the routed experts add structured contribution from
the rest. For a model that was originally dense, this preserves enough of
the dense behavior under the MoE-shaped file layout to give coherent
answers. For full dense-fidelity, use the aliasing recipe and accept the
inference slowdown.
"""
from __future__ import annotations

import argparse, hashlib, json, os, struct, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from .format import (FOOTER_SIZE, HEADER_SIZE, ArchType, ExpertEntry, Header,
                     LayerManifest, LodLevel, Specificity)
from .taxonomy import DOMAIN_NAMES, SUB_DOMAIN_NAMES
from .model import _read_header, _read_manifest

console = Console()


# GGUF dtype codes (must match atf.gguf_io / GGMLQuantizationType).
GGUF_F32 = 0
GGUF_F16 = 1
GGUF_BF16 = 30


def _read_dense_table(mmap, offset: int):
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
    base = src.with_name(src.stem + "_A2B.atf")
    if not base.exists():
        return base
    n = 1
    while True:
        cand = src.with_name(f"{src.stem}_A2B_{n}.atf")
        if not cand.exists():
            return cand
        n += 1


def _pack_manifest(m: LayerManifest) -> bytes:
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


def _f32_to_bf16_bytes(arr_f32: np.ndarray) -> bytes:
    """Pack an f32 ndarray to GGUF BF16 bytes (top 16 bits of each f32,
    little-endian uint16). BF16 has 7-bit mantissa (~3% relative error).
    """
    u32 = arr_f32.view(np.uint32)
    bf16 = (u32 >> 16).astype("<u2")
    return bf16.tobytes()


def _f32_to_f32_bytes(arr_f32: np.ndarray) -> bytes:
    """Pack an f32 ndarray as-is (GGUF F32)."""
    return np.ascontiguousarray(arr_f32, dtype=np.float32).tobytes()


def _table_entry_bytes(name: str, ndim: int) -> int:
    """Byte cost of one dense-table directory entry (must match the
    packing loop in convert_moe()'s table_buf construction)."""
    return 2 + len(name.encode("utf-8")) + 1 + 1 + 4 * ndim + 16


def _fixed_overhead_bytes(cfg: "MoeConfig", nblocks: int, hidden: int,
                          src_entries: list) -> int:
    """Bytes the real output file spends that do NOT scale with ffe_shex:
    header/footer, the dense-table directory (source entries + new
    scaffold entries -- dim COUNTS are fixed regardless of ffe_shex's
    actual value, so their directory-entry byte cost is fixed too), the
    manifest, the index-json blob, and the separate full router_section
    written at offset_router. Must be subtracted from the size budget
    before solving ffe_shex, or the real file silently exceeds the
    dry-run estimate.

    v3.1 fix (2026-09-03): found when a real conversion landed at
    11.645 GB against an 11.636 GB ceiling -- the ~84 MB router_section
    plus ~60 KB of directory overhead were missing from the original
    budget solve. See notes.md.
    """
    total = HEADER_SIZE + FOOTER_SIZE
    total += 4  # out_entries count prefix
    for name, shape, dcode, ndim, _off, _sz in src_entries:
        total += _table_entry_bytes(name, ndim)
    for b in range(nblocks):
        total += _table_entry_bytes(f"blk.{b}.ffn_gate_inp.weight", 2)
        total += _table_entry_bytes(f"blk.{b}.ffn_gate_inp_shexp.weight", 1)
        total += _table_entry_bytes(f"blk.{b}.ffn_gate_shexp.weight", 2)
        total += _table_entry_bytes(f"blk.{b}.ffn_up_shexp.weight", 2)
        total += _table_entry_bytes(f"blk.{b}.ffn_down_shexp.weight", 2)
    manifest_out = LayerManifest(
        num_blocks=nblocks, num_experts_per_block=cfg.num_experts,
        experts=[], domain_names=list(DOMAIN_NAMES),
        sub_domain_names=list(SUB_DOMAIN_NAMES))
    total += len(_pack_manifest(manifest_out))
    idx_payload = json.dumps({
        "domains": {d: {s: [] for s in SUB_DOMAIN_NAMES} for d in DOMAIN_NAMES},
        "centroids": {},
    }).encode("utf-8")
    total += 4 + len(idx_payload)
    total += 4 + (nblocks * hidden * cfg.num_experts * 4)  # router_section (+len prefix)
    return total


@dataclass
class MoeConfig:
    src_path: Path
    dst_path: Path
    num_experts: int = 64       # A2B recipe default (was 256 for A3B)
    top_k: int = 5               # A2B recipe default (was 8 for A3B)
    ffe_shex: int | None = None  # None = auto-solve from size ceiling (see
                                  # ceiling_file/max_output_bytes below); falls
                                  # back to ffe // num_experts if no ceiling known
    router_dtype: int = GGUF_F32 # 0=F32, 30=BF16 -- F16(1) is NOT supported,
                                  # gguf_io has no dequantizer for tensor_type=1
                                  # (found 2026-09-03: real load crashed with
                                  # "No dequantizer for tensor_type=1")
    shexp_gate_bias: float = 10.0  # sigmoid(x @ W) saturation; +10 -> ~1.0 for positive x@W
    shexp_storage: str = "reuse"  # v4 default: shared expert computed from the
                                  # block's own dense FFN tensors at inference
                                  # time (engine.py qmm/merge_ffn fallback) --
                                  # ZERO extra file bytes, output size stays
                                  # ~equal to source. "slice" and "alias" are
                                  # legacy: both confirmed NOT to give coherent
                                  # output at a size the ceiling allows (see
                                  # notes.md 2026-09-03).
    shexp_dtype: int = GGUF_BF16  # 0=F32, 30=BF16 -- BF16 is half the size with ~3% error
    seed: int = 0
    dry_run: bool = False
    # v3 (2026-09-03): size-ceiling-aware ffe_shex auto-solve -- see the
    # long comment in convert_moe() for why the old fixed ffe//num_experts
    # default produced garbage output.
    ceiling_file: Path | None = None     # output size must not exceed this file's size
    max_output_bytes: int | None = None  # explicit override of ceiling_file
    budget_margin: float = 0.97          # fraction of headroom actually used


def convert_moe(cfg: MoeConfig) -> dict:
    start = time.time()
    src = Path(cfg.src_path)
    dst = Path(cfg.dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"output exists: {dst} (delete first)")

    console.rule("[bold blue]ATF MoE Conversion (v2: alias+slice recipes)")
    console.print(f"  Source: {src.name} ({src.stat().st_size/1e9:.3f} GB)")
    console.print(f"  Output: {dst}")
    console.print(f"  Recipe: {cfg.shexp_storage}, "
                  f"experts: {cfg.num_experts}, top_k: {cfg.top_k}, "
                  f"shexp_dtype: {['F32','F16','?','?'][cfg.shexp_dtype] if cfg.shexp_dtype in (0,1,2,3) else f'code={cfg.shexp_dtype}'}")
    console.print(f"  router_dtype: {'F32' if cfg.router_dtype == GGUF_F32 else 'BF16'}")
    if cfg.dry_run:
        console.print("  [yellow]Dry-run: not writing output[/yellow]")
    console.print()

    import mmap as _mmap
    src_f = open(src, "rb")
    src_mm = _mmap.mmap(src_f.fileno(), 0, access=_mmap.ACCESS_READ)

    header = _read_header(src_f)
    manifest = _read_manifest(src_f, header.offset_manifest)
    nblocks = header.num_blocks
    hidden = header.hidden_dim

    src_count, src_entries, src_blob_base = _read_dense_table(src_mm, header.offset_dense)
    has_router = any(name.endswith("ffn_gate_inp.weight") and "blk." in name
                     for name, *_ in src_entries)
    if has_router:
        src_f.close(); src_mm.close()
        raise RuntimeError("source already has ffn_gate_inp weights -- MoE file")

    # Locate per-block dense FFN records
    block_ffn = {}
    for b in range(nblocks):
        p = f"blk.{b}."
        block_ffn[b] = {}
        for name, shape, dcode, ndim, blob_off, blob_sz in src_entries:
            if not name.startswith(p):
                continue
            for proj in ("gate", "up", "down"):
                if name.endswith(f".ffn_{proj}.weight"):
                    block_ffn[b][proj] = (shape, dcode, blob_off, blob_sz)
                    break
        if set(block_ffn[b]) != {"gate", "up", "down"}:
            src_f.close(); src_mm.close()
            raise ValueError(f"block {b}: no gate/up/down in source")

    ffe = block_ffn[0]["gate"][0][1]
    if ffe % cfg.num_experts != 0:
        src_f.close(); src_mm.close()
        raise ValueError(
            f"ffe ({ffe}) must be divisible by num_experts ({cfg.num_experts}); "
            f"rem={ffe % cfg.num_experts}")

    # Validate dense record uniformity
    for b in range(1, nblocks):
        if (block_ffn[b]["gate"][0] != block_ffn[0]["gate"][0] or
            block_ffn[b]["up"][0]   != block_ffn[0]["up"][0]   or
            block_ffn[b]["down"][0] != block_ffn[0]["down"][0]):
            src_f.close(); src_mm.close()
            raise ValueError(
                f"block {b} FFN shape differs from block 0")

    gate_shape = block_ffn[0]["gate"][0]
    up_shape = block_ffn[0]["up"][0]
    down_shape = block_ffn[0]["down"][0]

    router_dtype_size = 4 if cfg.router_dtype == GGUF_F32 else 2
    router_bytes = hidden * cfg.num_experts * router_dtype_size
    sgate_bytes = hidden * 4  # F32 shared-expert gate
    per_block_scaffold = router_bytes + sgate_bytes

    # Resolve the output-size ceiling (rule: must not exceed the real
    # 35B-A3B MoE model's size). cfg.max_output_bytes wins if set;
    # otherwise cfg.ceiling_file's size is used if that file exists.
    max_out_bytes = cfg.max_output_bytes
    ceiling_desc = None
    if max_out_bytes is None and cfg.ceiling_file is not None:
        cf = Path(cfg.ceiling_file)
        if cf.exists():
            max_out_bytes = cf.stat().st_size
            ceiling_desc = f"{cf.name} ({max_out_bytes/1e9:.3f} GB)"
    if max_out_bytes is not None:
        console.print(f"  Size ceiling: "
                      f"{ceiling_desc or f'{max_out_bytes/1e9:.3f} GB (explicit)'}")

    if cfg.shexp_dtype == GGUF_BF16:
        shexp_dtype_size = 2
    elif cfg.shexp_dtype == GGUF_F32:
        shexp_dtype_size = 4
    else:
        src_f.close(); src_mm.close()
        raise ValueError(f"unsupported shexp_dtype {cfg.shexp_dtype} "
                         "(use GGUF_F32=0 or GGUF_BF16=30)")

    if cfg.shexp_storage == "reuse":
        # v4 (2026-09-03): shared expert reuses the block's own dense FFN
        # tensors directly at inference time (engine.py fallback) -- no
        # shexp records are written at all, so there is no ffe_shex/size
        # question for this mode. Requires the v4 engine.py fix (falls
        # back automatically when shexp records are absent from the file).
        ffe_shex = 0
        console.print("  [dim]shexp_storage=reuse: shared expert computed "
                      "from the block's own dense FFN tensors at inference "
                      "time -- no new shexp records written (0 extra bytes "
                      "per block beyond the router).[/dim]")
    else:
        # v3 fix (2026-09-03): the old fixed default (ffe // num_experts, i.e.
        # ONE expert's slice -- 272 of 17408 on the 27B, ~1.6% of the dense
        # FFN's intermediate width) is why "slice" recipe output was garbage:
        # combined with top_k equally-narrow routed slices, only ~9% of the
        # original FFN contributed per block, and a dense-trained FFN's
        # weights assume ALL intermediate units contribute jointly -- dropping
        # ~90% of them is not a small perturbation. Fix: make ffe_shex as WIDE
        # as the size ceiling allows, auto-solved from real file sizes, instead
        # of a fixed 1/num_experts fraction. Wider shexp is closer to the
        # "alias" recipe's correctness (shared expert ~= full dense FFN) while
        # staying in the fast, LRU-cacheable "slice" storage format (pre-baked
        # once at convert time, unlike alias which forces the engine to
        # re-dequantize the FULL dense tensor from scratch on every cache
        # miss). Explicit --ffe-shex always overrides this.
        floor_default = ffe // cfg.num_experts
        if cfg.ffe_shex is not None:
            ffe_shex = cfg.ffe_shex
            if max_out_bytes is not None:
                console.print(f"  [dim]--ffe-shex explicit ({ffe_shex}); "
                              f"not auto-solving against the size ceiling[/dim]")
        elif cfg.shexp_storage == "slice" and max_out_bytes is not None:
            src_size = src.stat().st_size
            fixed_overhead = _fixed_overhead_bytes(cfg, nblocks, hidden, src_entries)
            budget_bytes = int((max_out_bytes - src_size - fixed_overhead) * cfg.budget_margin)
            per_block_budget = budget_bytes // nblocks - per_block_scaffold
            # 3 shexp records per block (gate, up, down), each hidden*ffe_shex
            # elements. (v3 fix: this used to be computed as 2x, undercounting
            # the real per-block cost by 1/3 -- see notes.md 2026-09-03.)
            denom = 3 * hidden * shexp_dtype_size
            solved = max(0, per_block_budget) // denom if denom > 0 else 0
            ffe_shex = max(floor_default, min(int(solved), ffe))
            console.print(f"  Auto-solved ffe_shex={ffe_shex} from size budget "
                          f"(ceiling {max_out_bytes/1e9:.3f} GB, src "
                          f"{src_size/1e9:.3f} GB, fixed overhead "
                          f"{fixed_overhead/1e6:.1f} MB, margin {cfg.budget_margin:.0%}) "
                          f"-- {ffe_shex/ffe*100:.1f}% of ffe={ffe} "
                          f"(old fixed default was {floor_default}, "
                          f"{floor_default/ffe*100:.1f}%)")
            if ffe_shex <= floor_default:
                console.print(
                    f"  [yellow]WARNING: size budget too tight to widen shexp "
                    f"beyond the old default -- this conversion will likely "
                    f"still produce poor output. Consider a smaller "
                    f"num_experts (wider routed slices too) or accept a "
                    f"smaller model.[/yellow]")
        else:
            ffe_shex = floor_default
            if cfg.shexp_storage == "slice":
                console.print(
                    f"  [yellow]No size ceiling known (pass --ceiling-file or "
                    f"--max-output-bytes) -- using old fixed default "
                    f"ffe_shex={ffe_shex} ({ffe_shex/ffe*100:.1f}% of ffe). "
                    f"This is the setting that produced garbage output before "
                    f"-- pass a ceiling to auto-widen it.[/yellow]")
    if ffe_shex > ffe:
        src_f.close(); src_mm.close()
        raise ValueError(f"ffe_shex {ffe_shex} > ffe {ffe}")

    # Shexp slice storage cost (BF16 or F32). For the alias recipe the
    # shexp records alias the dense records, so no extra bytes per block.
    if cfg.shexp_storage == "reuse":
        shexp_extra_bytes_per_block = 0
    elif cfg.shexp_storage == "alias":
        shexp_extra_bytes_per_block = 0
    elif cfg.shexp_storage == "slice":
        # 3 shexp records per block: gate (hidden, ffe_shex), up (hidden,
        # ffe_shex), down (ffe_shex, hidden). (v3 fix: was computed as 2x
        # before -- see above.)
        per_block_shexp = 3 * hidden * ffe_shex * shexp_dtype_size
        shexp_extra_bytes_per_block = per_block_shexp
    else:
        raise ValueError(f"unknown shexp_storage {cfg.shexp_storage!r} "
                         "(use 'alias' or 'slice')")

    total_scaffold_bytes = (per_block_scaffold + shexp_extra_bytes_per_block) * nblocks

    if cfg.dry_run:
        t = Table(title=f"DRY-RUN: {dst.name}")
        t.add_column("Field", style="cyan"); t.add_column("Value")
        t.add_row("Source", f"{src.name} ({src.stat().st_size/1e9:.3f} GB)")
        t.add_row("Output (planned)", dst.name)
        t.add_row("Source records (copied verbatim)", str(src_count))
        t.add_row("Scaffold records per block", "5 (router + shexp_gate + 3 shexp)")
        t.add_row("Total out records", str(src_count + 5 * nblocks))
        t.add_row("Blocks", str(nblocks))
        t.add_row("Hidden", str(hidden))
        t.add_row("FFE", str(ffe))
        t.add_row("FFE / n_exp (per-expert width)", str(ffe // cfg.num_experts))
        t.add_row("Shared-expert intermediate (ffe_shex)", str(ffe_shex))
        t.add_row("Shexp storage", f"{cfg.shexp_storage} "
                                     f"({['alias','BF16','F16','?'][1] if cfg.shexp_dtype==30 else 'F32'})")
        if cfg.shexp_storage == "alias":
            t.add_row("Note: shexp aliases the full dense FFN -- preserves dense behavior",
                       "yes (but inference is slow: ~20s/token on 27B)")
        elif cfg.shexp_storage == "reuse":
            t.add_row("Note: shared expert reuses the block's own dense FFN "
                       "at inference time -- no shexp records stored",
                       "yes (0 extra bytes; requires v4 engine.py)")
        else:
            t.add_row("Note: shexp is a real slice of the dense FFN -- inference is fast",
                       "yes")
        t.add_row("Per-block router bytes", f"{router_bytes:,}")
        t.add_row("Per-block shared-expert gate bytes", f"{sgate_bytes:,}")
        t.add_row("Per-block shexp storage bytes (slice recipe)",
                   f"{shexp_extra_bytes_per_block:,}")
        t.add_row("Total scaffold bytes", f"{total_scaffold_bytes:,}")
        t.add_row("Total scaffold MB", f"{total_scaffold_bytes / 1e6:.1f} MB")
        est_out = src.stat().st_size + total_scaffold_bytes
        t.add_row("Estimated output size",
                   f"{est_out/1e9:.3f} GB (src {src.stat().st_size/1e9:.3f} + "
                   f"scaffold {total_scaffold_bytes/1e9:.3f})")
        console.print(t)
        src_f.close(); src_mm.close()
        return {"src": str(src), "dst": str(dst), "dry_run": True,
                "num_blocks": nblocks, "num_experts": cfg.num_experts,
                "top_k": cfg.top_k, "ffe": ffe, "ffe_shex": ffe_shex,
                "shexp_storage": cfg.shexp_storage,
                "duration_s": time.time() - start}

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
            slice_refs.append((len(out_entries)-1, src_ref[0], src_ref[1]))

    # 1. Copy source dense records verbatim
    for name, shape, dcode, ndim, blob_off, blob_sz in src_entries:
        out_entries.append((name, shape, dcode, -1, blob_sz))
        in_mem.append(None)
        slice_refs.append((len(out_entries)-1,
                            src_blob_base + blob_off, blob_sz))

    # 2. Per-block scaffold records
    rng = np.random.default_rng(cfg.seed)
    for b in range(nblocks):
        # Router: zero init (uniform softmax, deterministic argpartition)
        if cfg.router_dtype == GGUF_F32:
            rtr_bytes = np.zeros((hidden, cfg.num_experts), dtype=np.float32).tobytes()
            _add(f"blk.{b}.ffn_gate_inp.weight", (hidden, cfg.num_experts), GGUF_F32,
                 rtr_bytes)
        else:
            # BF16 zeros: the all-zero bit pattern is identical across
            # float formats, so this is just a zero uint16 buffer (not
            # F16 -- gguf_io has no dequantizer for tensor_type=1/F16,
            # see MoeConfig.router_dtype comment).
            rtr_bytes = np.zeros(hidden * cfg.num_experts, dtype="<u2").tobytes()
            _add(f"blk.{b}.ffn_gate_inp.weight", (hidden, cfg.num_experts), GGUF_BF16,
                 rtr_bytes)

        # Shared-expert per-token scalar gate (F32, large positive bias so sigmoid ~ 1.0)
        sgate = np.full((hidden,), cfg.shexp_gate_bias, dtype=np.float32)
        _add(f"blk.{b}.ffn_gate_inp_shexp.weight", (hidden,), GGUF_F32, sgate.tobytes())

        if cfg.shexp_storage == "reuse":
            # v4: no shexp gate/up/down records -- engine.py computes the
            # shared expert directly from this block's existing dense FFN
            # tensors (blk.{b}.ffn_{gate,up,down}.weight, already copied
            # verbatim in step 1 above). Zero extra bytes.
            continue

        gate_shape_b, gate_dcode_b, gate_off_b, gate_sz_b = block_ffn[b]["gate"]
        up_shape_b, up_dcode_b, up_off_b, up_sz_b = block_ffn[b]["up"]
        down_shape_b, down_dcode_b, down_off_b, down_sz_b = block_ffn[b]["down"]

        if cfg.shexp_storage == "alias":
            # Alias the FULL dense FFN bytes for shexp. Engine reads them
            # as (gate/up: [hidden, ffe_shex=ffe], down: [ffe_shex=ffe, hidden])
            # which happens to match the dense record's shape when
            # ffe_shex == ffe (the aliasing recipe).
            if ffe_shex != ffe:
                src_f.close(); src_mm.close()
                raise ValueError(
                    "alias storage requires ffe_shex == ffe "
                    f"(got ffe_shex={ffe_shex}, ffe={ffe}). Use --shexp-slice "
                    "or set --ffe-shex to the dense FFN's intermediate dim.")
            _add(f"blk.{b}.ffn_gate_shexp.weight", (gate_shape_b[0], ffe_shex),
                 gate_dcode_b, None,
                 src_ref=(src_blob_base + gate_off_b, gate_sz_b))
            _add(f"blk.{b}.ffn_up_shexp.weight", (up_shape_b[0], ffe_shex),
                 up_dcode_b, None,
                 src_ref=(src_blob_base + up_off_b, up_sz_b))
            _add(f"blk.{b}.ffn_down_shexp.weight", (ffe_shex, down_shape_b[1]),
                 down_dcode_b, None,
                 src_ref=(src_blob_base + down_off_b, down_sz_b))
        else:
            # Sliced recipe: dequant the dense FFN to f32, slice the first
            # ffe_shex columns (gate/up) or rows (down), re-pack in the
            # chosen shexp dtype, and store as a NEW record. The slice
            # aligns with one expert's slice (since ffe_shex == ffe/n_exp
            # by default).
            from .gguf_io import dequantize as _dq

            def _read_dequant(off: int, ln: int, dcode: int,
                              shape: tuple[int, ...]) -> np.ndarray:
                src_mm.seek(off)
                data = src_mm.read(ln)
                if len(data) != ln:
                    raise RuntimeError(f"short read: {len(data)} vs {ln}")
                return _dq(data, dcode, shape).astype(np.float32, copy=False)

            gate_f32 = _read_dequant(src_blob_base + gate_off_b, gate_sz_b,
                                     gate_dcode_b, gate_shape_b)
            # gate_f32 shape is (hidden, ffe) -- slice cols [0:ffe_shex]
            if ffe_shex < ffe:
                gate_shex_f32 = gate_f32[:, :ffe_shex].copy()
            else:
                gate_shex_f32 = gate_f32
            del gate_f32

            up_f32 = _read_dequant(src_blob_base + up_off_b, up_sz_b,
                                   up_dcode_b, up_shape_b)
            if ffe_shex < ffe:
                up_shex_f32 = up_f32[:, :ffe_shex].copy()
            else:
                up_shex_f32 = up_f32
            del up_f32

            down_f32 = _read_dequant(src_blob_base + down_off_b, down_sz_b,
                                     down_dcode_b, down_shape_b)
            # down_f32 shape is (ffe, hidden) -- slice rows [0:ffe_shex]
            if ffe_shex < ffe:
                down_shex_f32 = down_f32[:ffe_shex, :].copy()
            else:
                down_shex_f32 = down_f32
            del down_f32

            # Pack as shexp dtype
            if cfg.shexp_dtype == GGUF_BF16:
                gate_bytes = _f32_to_bf16_bytes(gate_shex_f32)
                up_bytes = _f32_to_bf16_bytes(up_shex_f32)
                down_bytes = _f32_to_bf16_bytes(down_shex_f32)
                shexp_dcode = GGUF_BF16
            else:  # F32
                gate_bytes = _f32_to_f32_bytes(gate_shex_f32)
                up_bytes = _f32_to_f32_bytes(up_shex_f32)
                down_bytes = _f32_to_f32_bytes(down_shex_f32)
                shexp_dcode = GGUF_F32
            del gate_shex_f32, up_shex_f32, down_shex_f32

            _add(f"blk.{b}.ffn_gate_shexp.weight", (hidden, ffe_shex),
                 shexp_dcode, gate_bytes)
            _add(f"blk.{b}.ffn_up_shexp.weight", (hidden, ffe_shex),
                 shexp_dcode, up_bytes)
            _add(f"blk.{b}.ffn_down_shexp.weight", (ffe_shex, hidden),
                 shexp_dcode, down_bytes)

    # 3. Build the output dense table manifest
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
        out_f.write(b"\x00" * HEADER_SIZE)
        offset_dense = out_f.tell()
        out_f.write(bytes(table_buf))
        for i, payload in enumerate(in_mem):
            if payload is None:
                src_off, src_len = src_ref_by_idx[i]
                src_mm.seek(src_off)
                chunk_bytes = src_mm.read(src_len)
                if len(chunk_bytes) != src_len:
                    raise RuntimeError(f"short read: {len(chunk_bytes)} vs {src_len}")
                out_f.write(chunk_bytes)
            else:
                out_f.write(payload)

        offset_experts = out_f.tell()
        out_f.write(b"")

        offset_manifest = out_f.tell()
        manifest_out = LayerManifest(
            num_blocks=nblocks,
            num_experts_per_block=cfg.num_experts,
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
        router_section = np.zeros((nblocks, hidden, cfg.num_experts), dtype=np.float32)
        out_f.write(struct.pack("<I", router_section.nbytes))
        out_f.write(router_section.tobytes())

        size_no_footer = out_f.tell()
        new_header = Header(
            arch=header.arch, version_major=2, version_minor=2,
            num_blocks=nblocks, hidden_dim=hidden,
            rope_dim=header.rope_dim or (hidden // max(header.num_heads, 1)),
            num_heads=header.num_heads, num_kv_heads=header.num_kv_heads,
            vocab_size=header.vocab_size,
            num_experts_per_block=cfg.num_experts, top_k=cfg.top_k,
            num_lod_levels=header.num_lod_levels, flags=0x07,
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
    console.print(f"  Output: {final_size/1e9:.3f} GB (source: {src.stat().st_size/1e9:.3f} GB)")
    return {"src": str(src), "dst": str(dst), "num_blocks": nblocks,
            "num_experts": cfg.num_experts, "top_k": cfg.top_k,
            "ffe": ffe, "ffe_shex": ffe_shex,
            "shexp_storage": cfg.shexp_storage,
            "src_bytes": src.stat().st_size, "dst_bytes": final_size,
            "duration_s": duration}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atf-convert-moe",
        description=("Add MoE scaffold (router + shared expert) to a "
                    "raw-GGUF ATF without re-quantizing. v2: alias and "
                    "slice recipes; slice is the new default."))
    p.add_argument("src", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--num-experts", type=int, default=64,
                   help="Number of experts per block (must divide ffe). "
                        "Default 64 (A2B); 256 was the A3B recipe.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Active experts per token. Default 5 (A2B); "
                        "8 was the A3B recipe.")
    p.add_argument("--ffe-shex", type=int, default=None,
                   help="Shared-expert intermediate dim. Default: auto-solved "
                        "from the size ceiling (--ceiling-file / "
                        "--max-output-bytes) so shexp is as wide as the "
                        "budget allows; falls back to ffe // num_experts "
                        "(one expert's slice) if no ceiling is known.")
    p.add_argument("--ceiling-file", type=Path, default=None,
                   help="Output size must not exceed this file's size (e.g. "
                        "the real 35B-A3B MoE model this A2B conversion is "
                        "bounded by). Auto-detected as "
                        "'Qwen-AgentWorld-35B-A3B-UD-IQ2_M.atf' next to the "
                        "source file if present and this flag is omitted.")
    p.add_argument("--max-output-bytes", type=int, default=None,
                   help="Explicit output size cap in bytes; overrides "
                        "--ceiling-file.")
    p.add_argument("--budget-margin", type=float, default=0.97,
                   help="Fraction of the size headroom under the ceiling to "
                        "actually use when auto-solving --ffe-shex (default "
                        "0.97 = 3%% safety slack).")
    p.add_argument("--router-dtype", choices=["f32", "bf16"], default="f32",
                   help="Router weight dtype (all-zero, untrained). f32 "
                        "(default) is safest; bf16 saves ~21MB total. NOTE: "
                        "f16 is NOT offered -- gguf_io has no dequantizer for "
                        "tensor_type=1, it crashes the engine on load.")
    p.add_argument("--shexp-gate-bias", type=float, default=10.0,
                   help="Constant for the shared-expert per-channel gate "
                        "(sigmoid(bias) ~ 1.0)")
    p.add_argument("--shexp-reuse", dest="shexp_storage",
                   action="store_const", const="reuse", default=None,
                   help="(default, v4) Compute the shared expert directly "
                        "from the block's own dense FFN tensors at inference "
                        "time -- zero extra file bytes, output size stays "
                        "~equal to the source model. Requires the v4 "
                        "engine.py fallback (auto-activates when shexp "
                        "records are absent from the file).")
    p.add_argument("--shexp-alias", dest="shexp_storage",
                   action="store_const", const="alias", default=None,
                   help="Legacy: physically duplicate the FULL dense FFN as "
                        "new shexp records (large -- roughly doubles FFN "
                        "storage) and slow at inference (CPU dequant, no "
                        "Metal path).")
    p.add_argument("--shexp-slice", dest="shexp_storage",
                   action="store_const", const="slice", default=None,
                   help="Legacy: dequant + slice + re-store the first "
                        "ffe_shex columns of the dense FFN as new shexp "
                        "records. Confirmed NOT coherent at the sizes the "
                        "size ceiling allows -- see notes.md 2026-09-03. "
                        "Prefer --shexp-reuse.")
    p.add_argument("--shexp-dtype", choices=["f32", "bf16"], default="bf16",
                   help="Dtype for sliced shexp records (default bf16, "
                        "half the size of f32 with ~3%% mantissa loss)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    src = Path(args.src)
    dst = Path(args.output) if args.output else _default_output_path(src)
    ceiling_file = args.ceiling_file
    if ceiling_file is None and args.max_output_bytes is None:
        auto = src.parent / "Qwen-AgentWorld-35B-A3B-UD-IQ2_M.atf"
        if auto.exists() and auto.resolve() != src.resolve():
            ceiling_file = auto
    cfg = MoeConfig(
        src_path=src, dst_path=dst,
        num_experts=args.num_experts, top_k=args.top_k,
        ffe_shex=args.ffe_shex,
        router_dtype=GGUF_F32 if args.router_dtype == "f32" else GGUF_BF16,
        shexp_gate_bias=args.shexp_gate_bias,
        shexp_storage=args.shexp_storage or "reuse",
        shexp_dtype=GGUF_F32 if args.shexp_dtype == "f32" else GGUF_BF16,
        seed=args.seed, dry_run=args.dry_run,
        ceiling_file=ceiling_file, max_output_bytes=args.max_output_bytes,
        budget_margin=args.budget_margin)
    try:
        result = convert_moe(cfg)
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        return 1
    console.print(f"  Result: {json.dumps(result, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
