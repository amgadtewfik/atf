"""
ATF Binary Format Specification.

File layout:
  [Header: 128 bytes]
  [Layer Manifest: variable]
  [Dense Weights: variable]  (embeddings, attention, norms, lm_head — always
                               loaded, never sparsified: inference needs them
                               every token regardless of expert routing)
  [Expert Data: variable]  (all experts, all LOD levels, stored on disk)
  [Knowledge Index: variable]
  [Router Weights: variable]
  [Footer: 32 bytes SHA-256 checksum]

Only LOD 0 experts (+ all dense weights) are loaded into RAM. All other
expert LODs remain memory-mapped or paged in on demand.
"""

from __future__ import annotations

import struct
import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np


MAGIC = b"ATF1"
# HEADER_SIZE depends on the format version:
#   V1, V2, V3: 128 bytes
#   V4+:        136 bytes (adds the QSA metadata)
# The reader reads version_minor first (offset 6, 2 bytes) to know which
# HEADER_SIZE to expect, then reads the rest of the header accordingly.
HEADER_SIZE_V3 = 128
HEADER_SIZE_V4 = 136


def header_size_for(version_minor: int) -> int:
    """Return the on-disk header size for the given format version."""
    return HEADER_SIZE_V4 if version_minor >= 4 else HEADER_SIZE_V3


HEADER_SIZE = HEADER_SIZE_V3  # legacy constant; new code should call header_size_for()
_HEADER_FMT_V1 = "<4sHHhhhHhiiBBII"   # version_minor <= 1
_HEADER_FMT_V2 = "<4sHHhhhHhiiHBII"   # version_minor >= 2: wide experts field
_HEADER_FMT_V3 = "<4sHHhhhHhiiHBIIIBBH"   # gamma/v2: ngram metadata (vocab u32, dim u16, insert_layer u8, context u8)
_HEADER_FMT_V4 = "<4sHHhhhHhiiHBIIIBBHQ"    # gamma/v3: QSA metadata packed into a single u64
                                             #   (indexer_heads:16 | kv_heads:16 | budget_blocks:16 | block_size:16)
FOOTER_SIZE = 32

DTYPE_CODES = {"f32": 0, "f16": 1, "i8": 2, "i32": 3}
DTYPE_FROM_CODE = {v: k for k, v in DTYPE_CODES.items()}
DTYPE_NP = {"f32": np.float32, "f16": np.float16, "i8": np.int8, "i32": np.int32}


class ArchType(IntEnum):
    LLAMA = 0
    LLAMA2 = 1
    LLAMA3 = 2
    MISTRAL = 3
    GPT2 = 4
    OTHER = 255


class LodLevel(IntEnum):
    """Precision level for expert storage."""
    LOD_0 = 0  # INT8 + per-channel scale (in-memory, active)
    LOD_1 = 1  # INT4 + per-block(32) scale (on-disk, hot)
    LOD_2 = 2  # INT2 + per-block(64) scale (on-disk, warm)
    LOD_3 = 3  # Low-rank decomposition rank=d//8 (on-disk, cold)
    LOD_4 = 4  # Hash index / abstract pointer (archive, no dense payload)


class Specificity(IntEnum):
    GENERAL = 0
    SPECIALIZED = 1
    IDIOSYNCRATIC = 2


@dataclass
class Header:
    version_major: int = 1
    version_minor: int = 0
    arch: ArchType = ArchType.OTHER
    num_blocks: int = 0
    hidden_dim: int = 0
    rope_dim: int = 0  # RoPE rotation dimension; 0 = use full head_dim (legacy ATFs)
    num_heads: int = 0
    num_kv_heads: int = 0
    vocab_size: int = 0
    num_experts_per_block: int = 0
    top_k: int = 8
    num_lod_levels: int = 5
    flags: int = 0  # bit0: has_router, bit1: has_index

    # gamma/v2 (version_minor >= 3): n-gram embedding metadata. When
    # ngram_vocab == 0 the ngram section is absent and ATF behaves exactly
    # as before -- all behaviour is gated on this field. See
    # docs/qwen38_flash_next_analysis.md §6 for the rationale.
    ngram_vocab: int = 0          # number of n-gram entries (Qwen4: 20,000,000)
    ngram_dim: int = 0            # embedding dim per n-gram entry
    ngram_insert_layer: int = 2   # layer index where the ngram contribution is added
    ngram_context: int = 2        # 2 = bigram, 3 = trigram

    # gamma/v3 (version_minor >= 4): QSA (Qwen Sparse Attention) metadata.
    # Packed into a single u64 (16 bits each) for HEADER_SIZE-budget reasons.
    # All values default to 0; when all are 0 the QSA path is a no-op and
    # ATF behaves exactly as before. See
    # docs/qwen38_flash_next_analysis.md §3.3 for the rationale.
    qsa_indexer_heads: int = 0    # indexer query heads (Qwen4: 4)
    qsa_kv_heads: int = 0         # indexer shared key heads (Qwen4: 1)
    qsa_budget_blocks: int = 0    # top-K blocks per query (Qwen4: 512)
    qsa_block_size: int = 0       # tokens per KV block (Qwen4: 16)

    # Offsets (filled during write, via AtfWriter.finalize)
    offset_manifest: int = 0
    offset_dense: int = 0
    offset_experts: int = 0
    offset_index: int = 0
    offset_router: int = 0
    file_size: int = 0

    def pack(self) -> bytes:
        # v19.2 (version_minor >= 2): num_experts_per_block widened B -> H so
        # real MoE models (e.g. qwen35moe expert_count=256) fit the field.
        # gamma/v2 (version_minor >= 3): n-gram metadata added in trailing
        # 8 bytes of the same 128-byte header (no HEADER_SIZE bump -- the
        # ngram table payload lives in the existing dense + expert sections,
        # only the metadata travels in the header). See
        # docs/qwen38_flash_next_analysis.md §6.2.
        if self.version_minor >= 4:
            fmt = _HEADER_FMT_V4
        elif self.version_minor >= 3:
            fmt = _HEADER_FMT_V3
        elif self.version_minor >= 2:
            fmt = _HEADER_FMT_V2
        else:
            fmt = _HEADER_FMT_V1
        # n-gram metadata only travels in V3+ headers. QSA metadata
        # only travels in V4+ headers. V1/V2/V3 pack calls omit the
        # higher-version args so a V3 file written by a v4-aware tool
        # stays byte-identical to one written by the original v3 code.
        if self.version_minor >= 4:
            # Pack 4 u16s into a single u64: 16 bits each.
            qsa_packed = (
                (int(self.qsa_indexer_heads) & 0xFFFF) << 48
                | (int(self.qsa_kv_heads) & 0xFFFF) << 32
                | (int(self.qsa_budget_blocks) & 0xFFFF) << 16
                | (int(self.qsa_block_size) & 0xFFFF)
            )
            buf = struct.pack(
                fmt,
                MAGIC,
                self.version_major,
                self.version_minor,
                self.arch,
                self.num_blocks,
                self.hidden_dim,
                self.rope_dim,
                self.num_heads,
                self.num_kv_heads,
                self.vocab_size,
                self.num_experts_per_block,
                self.top_k,
                self.num_lod_levels,
                self.flags,
                self.ngram_vocab,
                self.ngram_dim,
                self.ngram_insert_layer,
                self.ngram_context,
                qsa_packed,
            )
        elif self.version_minor >= 3:
            buf = struct.pack(
                fmt,
                MAGIC,
                self.version_major,
                self.version_minor,
                self.arch,
                self.num_blocks,
                self.hidden_dim,
                self.rope_dim,
                self.num_heads,
                self.num_kv_heads,
                self.vocab_size,
                self.num_experts_per_block,
                self.top_k,
                self.num_lod_levels,
                self.flags,
                self.ngram_vocab,
                self.ngram_dim,
                self.ngram_insert_layer,
                self.ngram_context,
            )
        else:
            buf = struct.pack(
                fmt,
                MAGIC,
                self.version_major,
                self.version_minor,
                self.arch,
                self.num_blocks,
                self.hidden_dim,
                self.rope_dim,
                self.num_heads,
                self.num_kv_heads,
                self.vocab_size,
                self.num_experts_per_block,
                self.top_k,
                self.num_lod_levels,
                self.flags,
            )
        buf += struct.pack(
            "<QQQQQQ",
            self.offset_manifest,
            self.offset_dense,
            self.offset_experts,
            self.offset_index,
            self.offset_router,
            self.file_size,
        )
        buf += struct.pack("<32s", b"")  # checksum placeholder (footer holds real one)
        buf += b"\x00" * (header_size_for(self.version_minor) - len(buf))
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> Header:
        _, vmaj, vmin = struct.unpack_from("<4sHH", data, 0)
        assert data[:4] == MAGIC, f"Bad magic: {data[:4]!r}"
        if vmin >= 4:
            fmt1 = _HEADER_FMT_V4
        elif vmin >= 3:
            fmt1 = _HEADER_FMT_V3
        elif vmin >= 2:
            fmt1 = _HEADER_FMT_V2
        else:
            fmt1 = _HEADER_FMT_V1
        if vmin >= 4:
            (
                magic, vmaj, vmin, arch, nblocks, hdim, rope_dim, nheads, nkv, vocab,
                nexperts, topk, nlods, flags,
                ngram_vocab, ngram_dim, ngram_insert_layer, ngram_context,
                qsa_packed,
            ) = struct.unpack_from(fmt1, data, 0)
            qsa_indexer_heads = (qsa_packed >> 48) & 0xFFFF
            qsa_kv_heads = (qsa_packed >> 32) & 0xFFFF
            qsa_budget_blocks = (qsa_packed >> 16) & 0xFFFF
            qsa_block_size = qsa_packed & 0xFFFF
        elif vmin >= 3:
            (
                magic, vmaj, vmin, arch, nblocks, hdim, rope_dim, nheads, nkv, vocab,
                nexperts, topk, nlods, flags,
                ngram_vocab, ngram_dim, ngram_insert_layer, ngram_context,
            ) = struct.unpack_from(fmt1, data, 0)
            qsa_indexer_heads = qsa_kv_heads = qsa_budget_blocks = qsa_block_size = 0
        else:
            (
                magic, vmaj, vmin, arch, nblocks, hdim, rope_dim, nheads, nkv, vocab,
                nexperts, topk, nlods, flags,
            ) = struct.unpack_from(fmt1, data, 0)
            ngram_vocab = ngram_dim = 0
            ngram_insert_layer = 2
            ngram_context = 2
            qsa_indexer_heads = qsa_kv_heads = qsa_budget_blocks = qsa_block_size = 0
        off = struct.calcsize(fmt1)
        (off_man, off_dense, off_exp, off_idx, off_rtr, fsize) = struct.unpack_from(
            "<QQQQQQ", data, off
        )
        return cls(
            version_major=vmaj, version_minor=vmin, arch=ArchType(arch),
            num_blocks=nblocks, hidden_dim=hdim, rope_dim=rope_dim,
            num_heads=nheads, num_kv_heads=nkv, vocab_size=vocab,
            num_experts_per_block=nexperts, top_k=topk, num_lod_levels=nlods,
            flags=flags,
            ngram_vocab=ngram_vocab, ngram_dim=ngram_dim,
            ngram_insert_layer=ngram_insert_layer, ngram_context=ngram_context,
            qsa_indexer_heads=qsa_indexer_heads, qsa_kv_heads=qsa_kv_heads,
            qsa_budget_blocks=qsa_budget_blocks, qsa_block_size=qsa_block_size,
            offset_manifest=off_man, offset_dense=off_dense,
            offset_experts=off_exp, offset_index=off_idx, offset_router=off_rtr,
            file_size=fsize,
        )


@dataclass
class ExpertEntry:
    """One entry in the layer manifest."""
    expert_id: int
    block_index: int
    domain_id: int
    sub_domain_id: int
    specificity: Specificity
    lod_level: LodLevel
    weight_offset: int  # offset into expert data section
    weight_size: int
    centroid_offset: int = 0  # offset to centroid embedding (float32), unused for now
    centroid_dim: int = 0


@dataclass
class LayerManifest:
    num_blocks: int
    num_experts_per_block: int
    experts: list[ExpertEntry] = field(default_factory=list)
    domain_names: list[str] = field(default_factory=list)
    sub_domain_names: list[str] = field(default_factory=list)


@dataclass
class KnowledgeIndex:
    """Maps domain taxonomy to experts for dynamic expansion."""
    domains: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    expert_centroids: dict[int, list[float]] = field(default_factory=dict)


@dataclass
class QSAIndex:
    """gamma/v3: per-block sparse-attention metadata.

    Each QSA block carries:
      - indexer weights (q_proj, k_proj) for computing per-block scores
      - main-attention q/k/v (same projection family as full-attn)
      - a per-block selection cache (`QSAIndex` instance) of shape
        [num_blocks] holding the indexer's top-k selection

    The HEADER carries the metadata (qsa_indexer_heads, qsa_kv_heads,
    qsa_budget_blocks, qsa_block_size). The PAYLOAD lives in the dense
    section: indexer weights are tiny (4 query heads + 1 key head on
    128-dim each = ~1 KB) and main-attn q/k/v are the same shape as a
    full-attn block.

    Like NgramSection, this dataclass is the in-memory shape used in
    tests + roundtrip code. Production loaders stream the payload via
    the existing model.w() path.
    """
    indexer_heads: int
    kv_heads: int
    budget_blocks: int
    block_size: int
    # Per-block indexer scores. Populated during forward and discarded
    # at the end of each step (QSA scores are local, not persistent).
    scores: np.ndarray | None = None  # [num_blocks] fp16

    @property
    def is_enabled(self) -> bool:
        return (self.indexer_heads > 0 and self.kv_heads > 0
                and self.budget_blocks > 0 and self.block_size > 0)


@dataclass
class NgramSection:
    """gamma/v2: container for the n-gram table payload.

    The HEADER carries the metadata (ngram_vocab / ngram_dim /
    ngram_insert_layer / ngram_context). The PAYLOAD lives in two pieces:

      - ngram_emb.weight: [ngram_vocab, ngram_dim] main table. Stored in
        the EXPERT section under LOD_1 (int4) by default, with per-block
        (32) scales -- per docs/qwen38_flash_next_analysis.md §6.3.
        The bigram/trigram locality means the LRU paging already in
        model.py keeps the working set bounded.
      - ngram_emb_proj.weight: [ngram_dim, hidden] projection. Stored in
        the DENSE section as LOD-0 (raw f16). Tiny -- ngram_dim is small
        (Qwen4: 64-256).
      - ngram_emb_norm.weight: [ngram_dim] RMSNorm. Stored in the DENSE
        section as raw f16 (precision-critical, same rule ATF already
        uses for ssm_a / ssm_dt.bias per update.md "Precision-critical
        tiny GDN tensors").

    This dataclass just holds the in-memory representation for tests
    and for the synthetic roundtrip; production loaders stream the
    payload via the existing _raw_base / w() / _dequant_expert_slice
    paths.
    """
    vocab: int                        # ngram_vocab
    dim: int                          # ngram_dim
    insert_layer: int                 # ngram_insert_layer
    context: int                      # ngram_context (2 = bigram, 3 = trigram)
    table: np.ndarray | None = None   # [vocab, dim] -- LOD-1 int8 in tests
    proj: np.ndarray | None = None    # [dim, hidden] f16
    norm: np.ndarray | None = None    # [dim] f16

    @property
    def is_enabled(self) -> bool:
        return self.vocab > 0 and self.dim > 0 and self.table is not None


def compute_checksum(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def verify_checksum(file_path: Path) -> bool:
    data = file_path.read_bytes()
    if len(data) < HEADER_SIZE_V4 + FOOTER_SIZE:
        return False
    stored = data[-FOOTER_SIZE:]
    content = data[:-FOOTER_SIZE]
    return compute_checksum(content) == stored


class AtfWriter:
    """Writes an ATF file from in-memory structures.

    Each write_* call appends a section and records its start offset, so
    finalize() just copies those recorded offsets into the header instead of
    trying to recompute them (the original version of this class referenced
    a self._manifest_size that was never set, which would have raised
    AttributeError the first time finalize() ran — fixed here).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.buf = bytearray()
        self._offsets: dict[str, int] = {}

    def _tell(self) -> int:
        return len(self.buf)

    def write_header(self, h: Header):
        self._offsets["header"] = self._tell()
        self.buf.extend(h.pack())

    def write_manifest(self, manifest: LayerManifest):
        self._offsets["manifest"] = self._tell()
        self.buf.extend(struct.pack("<HH", manifest.num_blocks, manifest.num_experts_per_block))
        self.buf.extend(struct.pack("<H", len(manifest.domain_names)))
        for name in manifest.domain_names:
            encoded = name.encode("utf-8")
            self.buf.extend(struct.pack("<H", len(encoded)))
            self.buf.extend(encoded)
        self.buf.extend(struct.pack("<H", len(manifest.sub_domain_names)))
        for name in manifest.sub_domain_names:
            encoded = name.encode("utf-8")
            self.buf.extend(struct.pack("<H", len(encoded)))
            self.buf.extend(encoded)

        self.buf.extend(struct.pack("<I", len(manifest.experts)))
        for e in manifest.experts:
            self.buf.extend(struct.pack(
                "<QBBBBBQQQI",
                e.expert_id,
                e.block_index,
                e.domain_id,
                e.sub_domain_id,
                int(e.specificity),
                int(e.lod_level),
                e.weight_offset,
                e.weight_size,
                e.centroid_offset,
                e.centroid_dim,
            ))

    def write_dense(self, tensors: dict[str, np.ndarray], dtype_map: dict[str, str] | None = None):
        """Dense (always-loaded) tensors: embeddings, attention, norms, lm_head.
        Stored as: [count][per-tensor manifest entries][raw blob]."""
        dtype_map = dtype_map or {}
        self._offsets["dense"] = self._tell()
        manifest = bytearray()
        blob = bytearray()
        manifest.extend(struct.pack("<I", len(tensors)))
        for name, arr in tensors.items():
            dt = dtype_map.get(name, "f16")
            np_dt = DTYPE_NP[dt]
            data = np.ascontiguousarray(arr, dtype=np_dt).tobytes()
            name_b = name.encode("utf-8")
            manifest.extend(struct.pack("<H", len(name_b)))
            manifest.extend(name_b)
            manifest.extend(struct.pack("<B", DTYPE_CODES[dt]))
            manifest.extend(struct.pack("<B", arr.ndim))
            for d in arr.shape:
                manifest.extend(struct.pack("<I", d))
            manifest.extend(struct.pack("<QQ", len(blob), len(data)))
            blob.extend(data)
        self.buf.extend(manifest)
        self.buf.extend(blob)

    def write_expert_data(self, data: bytes):
        self._offsets["experts"] = self._tell()
        self.buf.extend(data)

    def write_index(self, index: KnowledgeIndex):
        import json
        self._offsets["index"] = self._tell()
        serialized = json.dumps({
            "domains": index.domains,
            "centroids": {str(k): v for k, v in index.expert_centroids.items()},
        }).encode("utf-8")
        self.buf.extend(struct.pack("<I", len(serialized)))
        self.buf.extend(serialized)

    def write_router(self, data: bytes):
        self._offsets["router"] = self._tell()
        self.buf.extend(struct.pack("<I", len(data)))
        self.buf.extend(data)

    def finalize(self, header: Header):
        header.offset_manifest = self._offsets.get("manifest", 0)
        header.offset_dense = self._offsets.get("dense", 0)
        header.offset_experts = self._offsets.get("experts", 0)
        header.offset_index = self._offsets.get("index", 0)
        header.offset_router = self._offsets.get("router", 0)
        header.file_size = self._tell() + FOOTER_SIZE
        self.buf[:header_size_for(header.version_minor)] = header.pack()
        self.buf.extend(compute_checksum(bytes(self.buf)))
        self.path.write_bytes(bytes(self.buf))


class AtfReader:
    """Memory-mapped reader for ATF files. Only LOD 0 expert data + dense
    weights are meant to be loaded to RAM by callers (see atf/model.py);
    this class itself just exposes random access to any section."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._mm = None
        self._fh = None
        self._all_bytes: bytes | None = None
        self.header: Header = self._read_header()
        self.manifest: LayerManifest = self._read_manifest()

    def _data(self) -> bytes:
        if self._all_bytes is None:
            self._all_bytes = self.path.read_bytes()
        return self._all_bytes

    def _read_header(self) -> Header:
        data = self._data()[:HEADER_SIZE_V4]  # >= max header size across versions
        return Header.unpack(data)

    def _read_manifest(self) -> LayerManifest:
        data = self._data()
        off = self.header.offset_manifest
        (nblocks, nexp_per_blk) = struct.unpack_from("<HH", data, off)
        off += 4
        ndomains = struct.unpack_from("<H", data, off)[0]
        off += 2
        domains = []
        for _ in range(ndomains):
            slen = struct.unpack_from("<H", data, off)[0]
            off += 2
            domains.append(data[off:off + slen].decode("utf-8"))
            off += slen
        nsubs = struct.unpack_from("<H", data, off)[0]
        off += 2
        subs = []
        for _ in range(nsubs):
            slen = struct.unpack_from("<H", data, off)[0]
            off += 2
            subs.append(data[off:off + slen].decode("utf-8"))
            off += slen
        nexperts = struct.unpack_from("<I", data, off)[0]
        off += 4
        entry_size = struct.calcsize("<QBBBBBQQQI")
        experts = []
        for _ in range(nexperts):
            (eid, bidx, did, sid, spec, lod, woff, wsize, coff, cdim) = \
                struct.unpack_from("<QBBBBBQQQI", data, off)
            off += entry_size
            experts.append(ExpertEntry(
                expert_id=eid, block_index=bidx, domain_id=did, sub_domain_id=sid,
                specificity=Specificity(spec), lod_level=LodLevel(lod),
                weight_offset=woff, weight_size=wsize,
                centroid_offset=coff, centroid_dim=cdim,
            ))
        return LayerManifest(
            num_blocks=nblocks, num_experts_per_block=nexp_per_blk,
            experts=experts, domain_names=domains, sub_domain_names=subs,
        )

    def read_dense(self) -> dict[str, np.ndarray]:
        data = self._data()
        off = self.header.offset_dense
        (count,) = struct.unpack_from("<I", data, off)
        off += 4
        blob_start_relative_entries = []
        for _ in range(count):
            nlen = struct.unpack_from("<H", data, off)[0]
            off += 2
            name = data[off:off + nlen].decode("utf-8")
            off += nlen
            dtype_code = struct.unpack_from("<B", data, off)[0]
            off += 1
            ndim = struct.unpack_from("<B", data, off)[0]
            off += 1
            shape = struct.unpack_from(f"<{ndim}I", data, off)
            off += 4 * ndim
            blob_off, nbytes = struct.unpack_from("<QQ", data, off)
            off += 16
            blob_start_relative_entries.append((name, dtype_code, shape, blob_off, nbytes))
        blob_base = off  # blob immediately follows the manifest we just walked
        result = {}
        for name, dtype_code, shape, blob_off, nbytes in blob_start_relative_entries:
            dt = DTYPE_NP[DTYPE_FROM_CODE[dtype_code]]
            arr = np.frombuffer(data, dtype=dt, count=int(np.prod(shape)) if shape else 1,
                                 offset=blob_base + blob_off).reshape(shape)
            result[name] = arr
        return result

    def mmap(self):
        """Open memory-mapped file handle (kept for large-file random access
        of expert data without holding the whole file as a bytes object)."""
        self._fh = open(self.path, "rb")
        self._mm = memoryview(self._fh.read())

    def read_expert_weights(self, entry: ExpertEntry) -> bytes:
        """Read raw weight bytes for a single expert (one LOD) from disk."""
        if self._mm is None:
            self.mmap()
        base = self.header.offset_experts
        return bytes(self._mm[base + entry.weight_offset: base + entry.weight_offset + entry.weight_size])

    def read_all_lod0(self) -> dict[int, bytes]:
        """Raw LOD-0 expert bytes, keyed by expert_id. See atf/model.py for
        the higher-level loader that dequantizes these into numpy arrays."""
        result = {}
        for e in self.manifest.experts:
            if e.lod_level == LodLevel.LOD_0:
                result[e.expert_id] = self.read_expert_weights(e)
        return result

    def close(self):
        if self._fh:
            self._fh.close()
        self._fh = None
        self._mm = None
        self._all_bytes = None
