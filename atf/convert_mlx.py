"""MLX-native checkpoint -> ATF converter (beta/v1, v2 file format).

Converts an MLX-community style safetensors directory (config.json +
model.safetensors[.index.json], mx.quantize triplets) into the SAME v2 ATF
layout the GGUF path writes (storage="lod1-all"): every matmul weight is a
pre-packed 4-bit quantized_matmul blob (dcode 4), small tensors stay INT8,
and each block ships a merged dense FFN in the LOD1 section (engine runs it
via ffn_*.weight.dense with zero runtime merging work).

The source is dense (Qwen3.5 has one MLP per layer -- see
atf-beta-mlx-migration.md), so no expert splitting / router learning happens
here: the router is zeros and the manifest carries zero-size expert entries,
exactly like a legacy lod1-all file. HF tensor names are mapped to the
GGUF-derived names the engine expects; shapes were cross-checked against
original/Qwen3.5-9B-IQ4_NL.gguf.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import os
import time
from pathlib import Path

import numpy as np
from rich.console import Console

from .format import (
    FOOTER_SIZE, HEADER_SIZE, ArchType, ExpertEntry, Header, LayerManifest,
    LodLevel, Specificity,
)
from .quantize import quantize_int8
from .taxonomy import DOMAIN_NAMES, SUB_DOMAIN_NAMES

console = Console()

NUM_EXPERTS = 8          # header parity with legacy files (unused: dense FFN)
TOP_K = 4
DENSE_QMM_CODE = 4       # payload: packed 4-bit quantized_matmul blob


def _pack_qmm(w_np: np.ndarray, bits: int = 4) -> bytes:
    """Pack logical [in, out] weight into the loader's qmm blob layout.

    Mirrors convert.py::_pack_qmm exactly: quantize W.T (group 64),
    layout <rows u32><cols u32><wq uint32><scales f16><biases f16> where
    rows/cols describe W.T."""
    import mlx.core as mx
    wt = mx.array(np.ascontiguousarray(w_np.T, dtype=np.float32))
    wq, scales, biases = mx.quantize(wt, group_size=64, bits=bits)
    mx.eval(wq, scales, biases)
    rows, cols = int(wt.shape[0]), int(wt.shape[1])
    hdr = struct.pack("<II", rows, cols)
    return (hdr + bytes(memoryview(np.ascontiguousarray(np.array(wq))).cast("B"))
            + np.asarray(scales, dtype=np.float16).tobytes()
            + np.asarray(biases, dtype=np.float16).tobytes())


# ── source reading ────────────────────────────────────────────────────────

def _detect_ngram_section_mlx(weight_index: dict) -> tuple:
    """gamma/v2: detect n-gram tensors in the HF/MLX safetensors index.

    Returns (vocab, dim, insert_layer, context) or (0, 0, 2, 2) if no
    ngram tensors are present. Same gating as convert.py:
    ATF_NGRAM_CONVERTER=1 to enable; no-op otherwise. See
    docs/qwen38_flash_next_analysis.md section 6.8.
    """
    if os.environ.get("ATF_NGRAM_CONVERTER", "0") != "1":
        return (0, 0, 2, 2)
    for name in weight_index:
        if _NGRAM_RE.match(name):
            if "ngram_embedding_table" in name or "ngram_embedding_value" in name:
                # Probe the actual safetensors file for shape. The
                # weight_index is name -> file; we don't have shapes
                # without reading the file. For the stub we return a
                # placeholder; the real converter integration will
                # probe the safetensors metadata.
                return (0, 0, 2, 2)  # shape discovery requires file probe
            if "ngram_emb_proj" in name:
                return (0, 0, 2, 2)
    return (0, 0, 2, 2)


# gamma/v3: QSA (Qwen Sparse Attention) detection in the MLX path.
# Mirrors convert.py:_detect_qsa_section but for HF/MLX safetensors
# naming. The pinned values (4, 1, 512, 16) are from the Qwen3.8-Flash-Next
# model card per analysis doc section 1.1 + 3.3.
_QSA_RE = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.self_attn\.attn_indexer\.(q_proj|k_proj|q_norm|k_norm)\.weight$"
)


def _detect_qsa_section_mlx(weight_index: dict) -> tuple:
    """Return (indexer_heads, kv_heads, budget_blocks, block_size) or
    (0, 0, 0, 0) if no QSA tensors are present."""
    if os.environ.get("ATF_QSA_CONVERTER", "0") != "1":
        return (0, 0, 0, 0)
    for name in weight_index:
        if _QSA_RE.match(name):
            return (4, 1, 512, 16)
    return (0, 0, 0, 0)


def _read_config(source_dir: Path) -> tuple[dict, dict]:
    cfg = json.loads((source_dir / "config.json").read_text())
    return cfg, (cfg.get("text_config", {}) or {})


def _weight_index(source_dir: Path) -> dict[str, str]:
    """tensor name -> shard file name."""
    idx_path = source_dir / "model.safetensors.index.json"
    if idx_path.exists():
        return dict(json.loads(idx_path.read_text())["weight_map"])
    single = source_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors found in {source_dir}")
    import mlx.core as _mx
    keys = set(_mx.load(str(single)).keys())
    return {k: "model.safetensors" for k in keys}


# ── name mapping ──────────────────────────────────────────────────────────

_PREFIX_RE = re.compile(r"^language_model\.")
# gamma/v1: MTP (Multi-Token Prediction) drafter tensors are KEPT, not
# skipped. The nextn.* tensors ride the dense section under the mtp.*
# namespace so the engine's MTP drafter forward can find them via
# model.w("mtp.<k>.<field>.weight"). See docs/qwen38_flash_next_analysis.md
# §3.2 and atf/convert.py for the GGUF-side equivalent.
_SKIP_RE = re.compile(r"^(vision_tower\.|visual\.)")
_MTP_RE = re.compile(
    r"^layers\.(\d+)\.nextn\.(eh_proj|enorm|hnorm|shared_head_head)\.weight$"
)
# gamma/v2: n-gram embedding (Qwen4) -- the HF safetensors naming
# convention for the n-gram tensors is predicted per the analysis doc
# section 6.3 ("embedding.ngram_embedding_table" + "ngram_emb_proj" +
# "ngram_emb_norm"). Real Qwen4 weights will land in models/ and
# confirm or refute these names; until then, this is a stub gated on
# ATF_NGRAM_CONVERTER=1.
_NGRAM_RE = re.compile(
    r"^(?:model\.)?ngram_embedding(?:_table|_hash|_value)\.weight$|^ngram_emb_proj\.weight$|^ngram_emb_norm\.weight$"
)


def map_name(hf: str):
    """HF safetensors name -> (engine name, kind) or None.

    kind: "matmul"  transpose to [in,out], pack as qmm blob;
          "int8"    small / 1-D / excluded tensor, INT8 blob;
          "conv1d"  GDN causal conv, reshaped then INT8;
          "ffn_*"   merged dense MLP, qmm blob into the LOD1 section only;
          "mtp"     MTP-drafter tensor (eh_proj/enorm/hnorm/shared_head_head).
    """
    n = _PREFIX_RE.sub("", hf)
    if _SKIP_RE.match(n) or _SKIP_RE.match(hf):
        return None
    # gamma/v1: handle the MTP block(s) BEFORE the generic layers.* branch.
    # The MTP block index in the HF layout is N (= num_blocks in the trunk
    # sense). We rewrite to mtp.0 / mtp.1 / etc. slot, mirroring the GGUF
    # convert.py path. Convention: contiguous MTP slots follow the trunk.
    m_mtp = _MTP_RE.match(n)
    if m_mtp:
        # Caller (the converter driver) computes slot from the MTP block
        # id ordering; here we just emit a name with the block id as a
        # literal and let the driver rewrite the slot. But to keep this
        # pure (and avoid a global counter), we emit "mtp.b{N}.{field}..."
        # and the driver post-processes. That keeps the function stateless.
        b = int(m_mtp.group(1))
        field = m_mtp.group(2)
        if field == "shared_head_head":
            emit = f"mtp.b{b}.{field}.weight"
        else:
            emit = f"mtp.b{b}.{field}.weight"
        # eh_proj is a 2-D weight (matmul); enorm / hnorm are 1-D (int8);
        # shared_head_head is 2-D (matmul) for Qwen3.5+ MTP.
        if field in ("eh_proj", "shared_head_head"):
            kind = "matmul"
        else:
            kind = "int8"
        return (emit, kind)
    n = n.removeprefix("model.")
    m = re.match(r"^layers\.(\d+)\.(.+)$", n)
    if not m:
        table = {
            "embed_tokens.weight": ("token_embd.weight", "int8"),
            "lm_head.weight": ("output.weight", "int8"),
            "norm.weight": ("output_norm.weight", "int8"),
        }
        return table.get(n)
    b, rest = int(m.group(1)), m.group(2)
    p = f"blk.{b}."
    simple = {
        "input_layernorm.weight": (p + "attn_norm.weight", "int8"),
        "post_attention_layernorm.weight": (p + "post_attention_norm.weight", "int8"),
        "self_attn.q_proj.weight": (p + "attn_q.weight", "matmul"),
        "self_attn.k_proj.weight": (p + "attn_k.weight", "matmul"),
        "self_attn.v_proj.weight": (p + "attn_v.weight", "matmul"),
        "self_attn.o_proj.weight": (p + "attn_output.weight", "matmul"),
        "self_attn.q_norm.weight": (p + "attn_q_norm.weight", "int8"),
        "self_attn.k_norm.weight": (p + "attn_k_norm.weight", "int8"),
        "linear_attn.in_proj_qkv.weight": (p + "attn_qkv.weight", "matmul"),
        "linear_attn.in_proj_z.weight": (p + "attn_gate.weight", "matmul"),
        "linear_attn.in_proj_a.weight": (p + "ssm_alpha.weight", "f16m"),
        "linear_attn.in_proj_b.weight": (p + "ssm_beta.weight", "f16m"),
        "linear_attn.A_log": (p + "ssm_a", "a_log"),   # engine needs -exp(A_log)
        "linear_attn.dt_bias": (p + "ssm_dt.bias", "int8"),
        "linear_attn.conv1d.weight": (p + "ssm_conv1d.weight", "conv1d"),
        "linear_attn.norm.weight": (p + "ssm_norm.weight", "int8"),
        "linear_attn.out_proj.weight": (p + "ssm_out.weight", "matmul"),
        "mlp.gate_proj.weight": (p + "ffn_gate.weight", "ffn_gate"),
        "mlp.up_proj.weight": (p + "ffn_up.weight", "ffn_up"),
        "mlp.down_proj.weight": (p + "ffn_down.weight", "ffn_down"),
    }
    return simple.get(rest)


# ── conversion ────────────────────────────────────────────────────────────

def convert_mlx(source_dir: Path, output_path: Path, target_bits: int = 4) -> dict:
    """Convert an MLX safetensors checkpoint directory to a v2 ATF file."""
    start = time.time()
    source_dir, output_path = Path(source_dir), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg, tc = _read_config(source_dir)
    hidden = int(tc["hidden_size"])
    inter = int(tc["intermediate_size"])
    num_blocks = int(tc["num_hidden_layers"])
    num_heads = int(tc["num_attention_heads"])
    num_kv = int(tc.get("num_key_value_heads", num_heads))
    head_dim = int(tc.get("head_dim", hidden // num_heads))
    vocab = int(tc["vocab_size"])
    rope_cfg = tc.get("rope_parameters") or {}
    frac = float(rope_cfg.get("partial_rotary_factor",
                              tc.get("partial_rotary_factor", 1.0)))
    rope_dim = max(1, int(head_dim * frac))
    quant = tc.get("quantization") or cfg.get("quantization") or {}
    src_gs = int(quant.get("group_size", 64))
    src_bits = int(quant.get("bits", 8))

    console.rule("[bold blue]ATF Conversion (MLX source)")
    console.print(f"  Input:   {source_dir}")
    console.print(f"  Output:  {output_path} (target bits: {target_bits})")
    console.print(f"  {cfg.get('model_type', '?')}: {num_blocks} blocks, "
                  f"hidden={hidden}, ffn={inter}, heads={num_heads} "
                  f"(kv {num_kv}), vocab={vocab}, rope_dim={rope_dim}, "
                  f"source g{src_gs}/b{src_bits}")
    console.print()

    weight_map = _weight_index(source_dir)

    # plan: (hf name, engine name, kind); skip scales/biases/vision/unknown
    plan = []
    for hf in sorted(weight_map):
        got = map_name(hf)
        if got is not None:
            plan.append((hf, got[0], got[1]))
    # gamma/v1: rewrite "mtp.b{N}.X.weight" -> "mtp.<K>.X.weight" where K
    # is the 0-based slot in MTP-block-id order. This makes the
    # emitted names match what the engine's MTP forward expects
    # (mtp.0.eh_proj.weight, mtp.0.enorm.weight, mtp.0.hnorm.weight,
    # mtp.shared_head_head.weight). We do this rewrite in-place on the
    # plan list so the rest of the driver is unaware.
    mtp_b_re = re.compile(r"^mtp\.b(\d+)\.(.+)$")
    mtp_b_ids = set()
    for _, eng, _ in plan:
        m_mb = mtp_b_re.match(eng)
        if m_mb:
            mtp_b_ids.add(int(m_mb.group(1)))
    mtp_b_to_slot = {b: k for k, b in enumerate(sorted(mtp_b_ids))}
    n_mtp_slots = len(mtp_b_to_slot)
    if n_mtp_slots:
        console.print(f"  [cyan]MTP drafter: {n_mtp_slots} MTP block(s) detected "
                      f"(block ids {sorted(mtp_b_ids)})[/cyan]")
    plan = [
        (hf, (f"mtp.{mtp_b_to_slot[int(mb.group(1))]}.{mb.group(2)}"
              if (mb := mtp_b_re.match(eng)) else eng), kind)
        for (hf, eng, kind) in plan
    ]
    # Drop duplicate mtp.shared_head_head.weight entries (only the first
    # MTP block's shared_head is kept; see convert.py legacy path for the
    # same dedup rule). This is rare in practice but possible.
    seen_mtp_shared = False
    deduped_plan = []
    for hf, eng, kind in plan:
        if eng == "mtp.shared_head_head.weight":
            if seen_mtp_shared:
                continue
            seen_mtp_shared = True
        deduped_plan.append((hf, eng, kind))
    plan = deduped_plan

    seen = {eng for _, eng, _ in plan}
    missing = [g for g in ("token_embd.weight", "output.weight",
                           "output_norm.weight") if g not in seen]
    for b in range(num_blocks):
        for req in ("attn_norm.weight", "post_attention_norm.weight",
                    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            if f"blk.{b}.{req}" not in seen:
                missing.append(f"blk.{b}.{req}")
    if missing:
        raise ValueError(f"checkpoint missing required tensors, e.g.: {missing[:8]}")

    dense_tmp_path = output_path.with_suffix(".mlx-dense.tmp")
    ffn_tmp_path = output_path.with_suffix(".mlx-lod1.tmp")
    dense_tmp = open(dense_tmp_path, "w+b")
    ffn_tmp = open(ffn_tmp_path, "w+b")

    entries: list[tuple[str, tuple[int, ...], int, int]] = []
    ffn_spans: dict[int, list[tuple[int, int]]] = {}   # block -> [(off,len) x3]

    def process(i: int, get_tensor):
        hf, eng, kind = plan[i]
        base = hf[:-len(".weight")] if hf.endswith(".weight") else hf
        import mlx.core as mx
        if f"{base}.scales" in weight_map:
            arr = mx.dequantize(get_tensor(hf), get_tensor(f"{base}.scales"),
                                get_tensor(f"{base}.biases"),
                                group_size=src_gs, bits=src_bits)
            mx.eval(arr)
            w = np.asarray(arr.astype(mx.float32), dtype=np.float32)
            del arr
        else:
            t = get_tensor(hf)
            if str(t.dtype) != "float32":
                t = t.astype(mx.float32)
            mx.eval(t)
            w = np.asarray(t, dtype=np.float32)

        if kind == "f16m":
            # precision-critical tiny GDN projections (decay/beta gates):
            # raw f16, GGUF-style [in,out]; engine reads them via m.w().
            w2 = np.ascontiguousarray(w.T)
            blob = w2.astype(np.float16).tobytes()
            entries.append((eng, w2.shape, 1, len(blob)))
            dense_tmp.write(blob)
        elif kind == "matmul":
            # INT8 resident (legacy "dual" layout): the non-raw engine reads
            # attention/GDN projections through AtfModel.w() and quantizes
            # them to MLX qmm at first use -- code-4 blobs here would break
            # that path (they are only served via _qcache prebuilt).
            w2 = np.ascontiguousarray(w.T)                       # [in, out]
            blob = quantize_int8(w2)
            entries.append((eng, w2.shape, 2, len(blob)))
            dense_tmp.write(blob)
        elif kind == "conv1d":
            w2 = np.ascontiguousarray(w.reshape(w.shape[0], -1).T)
            blob = w2.astype(np.float16).tobytes()
            entries.append((eng, w2.shape, 1, len(blob)))
            dense_tmp.write(blob)
        elif kind == "a_log":
            # engine computes g = ssm_a * softplus(dt + a), state *= exp(g);
            # HF stores A_log where the true decay is A = -exp(A_log).
            # f16, NOT int8: values span [-1,-0.009]; one int8 row-scale
            # wrecks weak-decay heads (~40% rel error).
            w2 = -np.exp(w.astype(np.float32))
            blob = w2.astype(np.float16).tobytes()
            entries.append((eng, w2.shape, 1, len(blob)))
            dense_tmp.write(blob)
        elif kind == "int8":
            if eng in ("token_embd.weight", "output.weight"):
                w2 = np.ascontiguousarray(w.T)                   # [in, out]
                blob_in = w2.reshape(1, -1) if w2.ndim == 1 else w2
                blob = quantize_int8(np.ascontiguousarray(blob_in))
                entries.append((eng, w2.shape, 2, len(blob)))
                dense_tmp.write(blob)
            else:
                # norms / dt_bias: raw f16 (tiny tensors, keep precision)
                blob = np.ascontiguousarray(w, dtype=np.float16).tobytes()
                entries.append((eng, w.shape, 1, len(blob)))
                dense_tmp.write(blob)
        elif kind.startswith("ffn_"):
            w2 = np.ascontiguousarray(w.T)                       # [in, out]
            blob = _pack_qmm(w2, target_bits)
            off = ffn_tmp.tell()
            ffn_tmp.write(blob)
            b = int(re.match(r"blk\.(\d+)\.", eng).group(1))
            proj = {"gate": 0, "up": 1, "down": 2}[kind.split("_")[1]]
            ffn_spans.setdefault(b, [None, None, None])[proj] = (off, len(blob))
        else:
            raise ValueError(f"unhandled kind {kind}")

    try:
        import mlx.core as _mx
        by_shard: dict[str, list[int]] = {}
        for i, (hf, _, _) in enumerate(plan):
            by_shard.setdefault(weight_map[hf], []).append(i)
        for shard in sorted(by_shard):
            console.print(f"  reading shard {shard} …")
            # mx.load (not safe_open): safetensors' readers choke on bf16
            # tensors, which mlx-community checkpoints commonly contain.
            shard_data = _mx.load(str(source_dir / shard))

            def get_tensor(name: str, _sd=shard_data):
                return _sd[name]

            for i in by_shard[shard]:
                process(i, get_tensor)
                del shard_data[plan[i][0]]
                if i % 16 == 0:
                    console.log(f"  [{i + 1}/{len(plan)}] tensors packed")
            del shard_data
        dense_tmp.flush()
        ffn_tmp.flush()

        out_f = open(output_path, "wb")
        out_f.write(b"\x00" * HEADER_SIZE)

        def wr(b_: bytes):
            out_f.write(b_)

        # dense section: count + entries + blobs
        offset_dense = out_f.tell()
        wr(struct.pack("<I", len(entries)))
        cursor = 0
        for name, shape, code, nb in entries:
            nb_b = name.encode("utf-8")
            wr(struct.pack("<H", len(nb_b)))
            wr(nb_b)
            wr(struct.pack("<BB", code, len(shape)))
            for d in shape:
                wr(struct.pack("<I", int(d)))
            wr(struct.pack("<QQ", cursor, nb))
            cursor += nb
        assert cursor == dense_tmp.tell()
        CH = 1 << 24
        dense_tmp.seek(0)
        while True:
            chunkb = dense_tmp.read(CH)
            if not chunkb:
                break
            wr(chunkb)
        offset_experts = out_f.tell()      # zero-size experts: no blobs here

        # manifest: zero-size expert entries (lod1 parity with legacy files)
        offset_manifest = out_f.tell()
        exp_entries = [
            ExpertEntry(expert_id=b * NUM_EXPERTS + e, block_index=b,
                        domain_id=0, sub_domain_id=0,
                        specificity=Specificity.GENERAL,
                        lod_level=LodLevel.LOD_0,
                        weight_offset=0, weight_size=0)
            for b in range(num_blocks) for e in range(NUM_EXPERTS)
        ]
        from .convert import _pack_manifest
        manifest = LayerManifest(
            num_blocks=num_blocks, num_experts_per_block=NUM_EXPERTS,
            experts=exp_entries, domain_names=list(DOMAIN_NAMES),
            sub_domain_names=list(SUB_DOMAIN_NAMES))
        wr(_pack_manifest(manifest))

        # knowledge index (empty -- no real experts)
        offset_index = out_f.tell()
        index_json = json.dumps({
            "domains": {DOMAIN_NAMES[d]: {
                SUB_DOMAIN_NAMES[s]: [] for s in range(len(SUB_DOMAIN_NAMES))}
                for d in range(len(DOMAIN_NAMES))},
            "centroids": {}}).encode("utf-8")
        wr(struct.pack("<I", len(index_json)))
        wr(index_json)

        # router: zeros (exact_ffn path never reads it)
        offset_router = out_f.tell()
        router = np.zeros((num_blocks, hidden, NUM_EXPERTS), dtype=np.float32)
        wr(struct.pack("<I", int(router.nbytes)))
        wr(router.tobytes())

        # LOD1 section: magic + u32 count, per block gate/up/down qmm blobs
        if set(ffn_spans) != set(range(num_blocks)) or \
                any(len(v) != 3 or any(x is None for x in v)
                    for v in ffn_spans.values()):
            raise RuntimeError(f"incomplete FFN coverage: {len(ffn_spans)} blocks")
        wr(struct.pack("<4sI", b"LOD1", num_blocks))
        for b in range(num_blocks):
            for off, nb in ffn_spans[b]:
                ffn_tmp.seek(off)
                remaining = nb
                while remaining:
                    chunkb = ffn_tmp.read(min(CH, remaining))
                    if not chunkb:
                        raise RuntimeError("lod1 temp truncated")
                    wr(chunkb)
                    remaining -= len(chunkb)
        out_f.flush()
        out_f.close()

        # footer (sha256 of everything before it) + header patch
        size_no_footer = output_path.stat().st_size
        header = Header(
            arch=ArchType.OTHER, version_major=2, version_minor=0,
            num_blocks=num_blocks, hidden_dim=hidden, rope_dim=rope_dim,
            num_heads=num_heads, num_kv_heads=num_kv, vocab_size=vocab,
            num_experts_per_block=NUM_EXPERTS, top_k=TOP_K,
            num_lod_levels=2, flags=0x03,
            offset_manifest=offset_manifest, offset_dense=offset_dense,
            offset_experts=offset_experts, offset_index=offset_index,
            offset_router=offset_router,
            file_size=size_no_footer + FOOTER_SIZE)
        with open(output_path, "r+b") as f:
            f.seek(0)
            f.write(header.pack())
            h = hashlib.sha256()
            f.seek(0)
            while chunkb := f.read(CH):
                h.update(chunkb)
            f.seek(size_no_footer)
            f.write(h.digest())
    finally:
        dense_tmp.close()
        ffn_tmp.close()
        dense_tmp_path.unlink(missing_ok=True)
        ffn_tmp_path.unlink(missing_ok=True)

    final_size = output_path.stat().st_size
    duration = time.time() - start
    console.rule()
    console.print(f"  [green]Done in {duration / 60:.1f} min[/green]")
    console.print(f"  Output: {final_size / 1e9:.3f} GB")
    return {"output": str(output_path), "size": final_size,
            "blocks": num_blocks, "duration": duration}
