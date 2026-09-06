"""Fused GGUF-dequantize matmul Metal kernels.

Runs y = x @ W directly on GGUF-packed weight bytes -- NO dequantization to
f32/int8 anywhere: the original quantized blocks ARE the GPU-resident
weights. One Metal kernel family serves every format; per-type logic lives
in a `gguf_dequant` device function selected at build time.

Weight bytes are the GGUF tensor payload exactly as stored in the .atf
(numpy-style row-major with REVERSED logical shape, i.e. a logical
[in, out] tensor is stored as `out` rows of `in` elements). The kernel
computes, for logical W[in, out]:

    y[t, n] = sum_k x[t, k] * dequant(row=n_base + n, col=k_base + k)

with n_base/k_base offsets so MoE expert slices (contiguous sub-ranges of
merged FFN tensors) are served without copying.

Supported types (SUPPORTED_DTYPES):
    Q8_0 (8), Q2_K (10)   -- tier 1
Others fall back to CPU decode in model.py until ported.

Every dequant function must be validated against the canonical gguf package
decoder on real tensors before being listed here.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

# GGMLQuantizationType values
Q8_0 = 8
Q2_K = 10
Q3_K = 11
Q4_K = 12
Q5_K = 13
IQ2_XXS = 16
IQ2_XS = 17
IQ3_XXS = 18
Q6_K = 14
IQ1_S = 19
IQ4_NL = 20
IQ3_S = 21
IQ2_S = 22
IQ4_XS = 23

SUPPORTED_DTYPES = {Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
                    IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS, IQ1_S,
                    IQ4_NL}

# Per-type constant tables (grid/codebook bytes), generated from the
# canonical gguf package (validated bit-exact -- see tmp/iq_mirror_test.py).
def _table_bytes(dtype_code: int) -> bytes | None:
    import numpy as np
    import gguf.quants as Q
    if dtype_code == IQ2_XXS:
        Q.IQ2_XXS.init_grid()
        return (bytes(np.frombuffer(Q.IQ2_XXS.ksigns, dtype=np.uint8))
                + Q.IQ2_XXS.grid.reshape(-1).astype(np.uint8).tobytes())
    if dtype_code == IQ2_XS:
        Q.IQ2_XS.init_grid()
        return (bytes(np.frombuffer(Q.IQ2_XXS.ksigns, dtype=np.uint8))
                + Q.IQ2_XS.grid.reshape(-1).astype(np.uint8).tobytes())
    if dtype_code == IQ2_S:
        Q.IQ2_S.init_grid()
        return Q.IQ2_S.grid.reshape(-1).astype(np.uint8).tobytes()
    if dtype_code == IQ3_S:
        Q.IQ3_S.init_grid()
        return Q.IQ3_S.grid.reshape(-1).astype(np.uint8).tobytes()
    if dtype_code == IQ1_S:
        Q.IQ1_S.init_grid()
        return Q.IQ1_S.grid.reshape(-1).astype(np.uint8).tobytes()
    if dtype_code == IQ3_XXS:
        Q.IQ3_XXS.init_grid()
        return (bytes(np.frombuffer(Q.IQ2_XXS.ksigns, dtype=np.uint8))
                + Q.IQ3_XXS.grid.reshape(-1).astype(np.uint8).tobytes())
    return None

# ---------------------------------------------------------------------------
# Metal source
# ---------------------------------------------------------------------------

_HEADER = """
static inline float rd_f16(const device uchar* p) {
    ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(__builtin_bit_cast(half, u));
}

// Q8_0: 32 elems/block -> 34 bytes: f16 d at +0, int8 qs[32] at +2.
static inline float dq_q8_0(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 5) * 34;
    float d = rd_f16(blk);
    int8_t q = as_type<int8_t>(blk[2 + (k & 31)]);
    return d * float(q);
}

// Q2_K: 256-elem superblock -> 84 bytes:
//   [0:16]   scales[j]: lo nibble = scale, hi nibble = min (j = e >> 4)
//   [16:80]  qs[64]: 2-bit codes, interleaved halves of 32 bytes per 128 elts
//   [80:82]  d (f16), [82:84] dmin (f16)
// value = d*scale*q - dmin*min, q in [0,3]
// Shared by Q4_K / Q5_K: 6-bit scale/min pairs packed in 12 bytes
// (layout per gguf-py Q4_K.get_scale_min).
static inline void get_scale_min(const device uchar* S, uint m,
                                 thread uchar* sc, thread uchar* mn) {
    if (m < 4u) {
        *sc = S[m] & 0x3Fu;
        *mn = S[m + 4] & 0x3Fu;
    } else {
        uint j = m - 4u;
        *sc = (S[8 + j] & 0xFu) | ((S[j] >> 2) & 0x30u);
        *mn = (S[8 + j] >> 4) | ((S[4 + j] >> 2) & 0x30u);
    }
}

// Q4_K: 256-elem superblock -> 144 bytes:
//   [0:2] d (f16), [2:4] dmin (f16), [4:16] scales[12], [16:144] qs[128]
// 8 sub-blocks of 32; value = d*sc*q - dmin*mn, q 4-bit lo/hi nibbles.
static inline float dq_q4_k(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 144;
    uint e = k & 255u;
    uint chunk = e >> 6;                 // 64-elem group -> 32 qs bytes
    uint halfw = (e >> 5) & 1u;          // low / high nibble
    uint pos = e & 31u;
    uchar sc, mn;
    get_scale_min(blk + 4, e >> 5, &sc, &mn);
    uchar b = blk[16 + chunk * 32 + pos];
    uint q = halfw ? (b >> 4) : (b & 15u);
    return rd_f16(blk) * float(sc) * float(q) - rd_f16(blk + 2) * float(mn);
}

// Q5_K: 256-elem superblock -> 176 bytes:
//   [0:2] d, [2:4] dmin, [4:16] scales[12], [16:48] qh[32], [48:176] qs[128]
// Same sub-block/scale scheme as Q4_K plus one high bit per element.
static inline float dq_q5_k(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 176;
    uint e = k & 255u;
    uint chunk = e >> 6;
    uint halfw = (e >> 5) & 1u;
    uint pos = e & 31u;
    uchar sc, mn;
    get_scale_min(blk + 4, e >> 5, &sc, &mn);
    uchar b = blk[48 + chunk * 32 + pos];
    uint qlo = halfw ? (b >> 4) : (b & 15u);
    uint hbit = (blk[16 + (e & 31u)] >> (e / 32u)) & 1u; // qh: bit plane, t-major
    uint q = qlo | (hbit << 4);
    return rd_f16(blk) * float(sc) * float(q) - rd_f16(blk + 2) * float(mn);
}

// Q3_K: 256-elem superblock -> 120 bytes:
//   [0:32] hmask[32], [32:96] qs[64], [96:108] scales[12], [108:110] d (f16)
// 16 sub-blocks of 16; q = 2-bit code minus ((~hmask_bit) << 2) in [-4, 3]
// (gguf-py broadcasts the shift axes BEFORE the byte axes -- hence the
//  t-major bit-plane indexing below).
// value = d * (6-bit scale - 32) * q.
static inline float dq_q3_k(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 110;
    uint e = k & 255u;
    uint a = e >> 7;             // which 32-byte qs half
    uint s = (e >> 5) & 3u;      // 2-bit group within the half
    uint p = e & 31u;
    uint qlo = (blk[32 + a * 32 + p] >> (2 * s)) & 3u;
    uint hbit = (blk[e & 31u] >> (e / 32u)) & 1u; // hmask: bit plane, t-major
    int q = int(qlo) - (int((hbit ^ 1u)) << 2);
    uint j = e >> 4;             // sub-block -> 6-bit scale
    uint lo = (blk[96 + (j & 7u)] >> ((j >> 3) * 4)) & 0xFu;
    uint hi = (blk[104 + (j & 3u)] >> (2 * (j >> 2))) & 3u;
    int sc = int(lo | (hi << 4)) - 32;
    return rd_f16(blk + 108) * float(sc) * float(q);
}

static inline float dq_q2_k(const device uchar* row, uint k) {
    const device uchar* sb = row + (k >> 8) * 84;
    uint e = k & 255u;
    uint hsel = e >> 7;                       // which 32-byte qs half
    uint ih = e & 127u;                       // index within the half
    uint pair = ih >> 5;                      // bit shift = 2 * pair
    uint pos = ih & 15u;
    uint sel = (ih & 31u) >= 16u ? 16u : 0u;  // low / high window half
    uint q = (sb[16 + hsel * 32 + sel + pos] >> (pair * 2)) & 3u;
    uchar scb = sb[e >> 4];
    float d = rd_f16(sb + 80);
    float dmin = rd_f16(sb + 82);
    return d * float(scb & 15u) * float(q) - dmin * float(scb >> 4);
}

// Q6_K -- canonical ggml block_q6_K (210 bytes), matching gguf-py and
// the fast kernels in gguf_fast_q.py:
//   [0:128]   ql -- low 4 bits (two interleaved 32-elem windows per half)
//   [128:192] qh -- high 2 bits, 4 codes per byte
//   [192:208] scales -- signed int8, one per 16-element group
//   [208:210] d (f16)
// Element e in superblock: half c=e>>7, slot var=(e&127)>>5, l=e&31.
// Code is biased by 32.
static inline float dq_q6_k(const device uchar* row, uint k) {
    const device uchar* base = row + (size_t)(k >> 8) * 210u;
    const uint c = (k >> 7) & 1u;             // 128-element half
    const uint r = k & 127u;
    const uint var = r >> 5;                  // output slot 0..3 (+0/32/64/96)
    const uint l = r & 31u;
    uint lo = base[c * 64u + l];
    uint hi = base[c * 64u + l + 32u];
    uint hb = base[128u + c * 32u + l];
    uint q;
    if (var == 0u)      { q = (lo & 0xFu) | ((hb         & 3u) << 4); }
    else if (var == 1u) { q = (hi & 0xFu) | ((hb >> 2 & 3u) << 4); }
    else if (var == 2u) { q = (lo >> 4)   | ((hb >> 4 & 3u) << 4); }
    else                { q = (hi >> 4)   | ((hb >> 6 & 3u) << 4); }
    int qq = (int)q - 32;                     // stored as q6 (biased by 32)
    int8_t s = (int8_t)base[192u + c * 8u + var * 2u + (l >> 4)];
    return rd_f16(base + 208u) * (float)s * (float)qq;
}
"""

_IQ_FUNCS = r'''
constant float DELTA_IQ1S = 0.125f;  // hibit-safe default; matches gguf-py IQ1_S.delta
constant int8_t KV4[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

// IQ4_NL: 32-element block, 18 bytes (f16 d + 16 nibble bytes). Nibble
// order is canonical lows-first: elements 0..15 from the LOW nibbles of
// qs[0..15], elements 16..31 from the HIGH nibbles (v14 fix, matches
// gguf-py and the CPU decoder in gguf_io).
static inline float dq_iq4_nl(const device uchar* row, uint k) {
    const device uchar* blk = row + (size_t)(k >> 5) * 18u;
    const uint r = k & 31u;
    uint q = blk[2u + (r & 15u)];
    q = (r < 16u) ? (q & 0x0Fu) : (q >> 4);
    return (float)rd_f16(blk) * (float)KV4[q];
}

static inline float dq_iq2_xxs(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 66;
    uint e = k & 255u;
    float d = rd_f16(blk);
    uint g = e >> 5, l = (e >> 3) & 3u, t = e & 7u;
    const device uchar* Bp = blk + 6 + 8u * g;
    uint B = (uint)Bp[0] | ((uint)Bp[1] << 8) | ((uint)Bp[2] << 16) | ((uint)Bp[3] << 24);
    float db = d * (0.5f + (float)(B >> 28)) * 0.25f;
    uint sidx = (B >> (7u * l)) & 0x7Fu;
    uint bit = (TBL[sidx] >> t) & 1u;
    float gv = (float)(int8_t)TBL[128 + (uint)blk[2 + 8 * g + l] * 8u + t];
    return db * gv * (bit ? -1.0f : 1.0f);
}
static inline float dq_iq2_xs(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 74;
    uint e = k & 255u;
    float d = rd_f16(blk);
    uint j = e >> 3, t = e & 7u, sb = e >> 4;
    uint q = (uint)blk[2 + 2 * j] | ((uint)blk[3 + 2 * j] << 8);
    uint bit = (TBL[q >> 9] >> t) & 1u;
    uint sc = (blk[66 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu;
    float db = d * (0.5f + (float)sc) * 0.25f;
    float gv = (float)(int8_t)TBL[128 + (q & 511u) * 8u + t];
    return db * gv * (bit ? -1.0f : 1.0f);
}
static inline float dq_iq2_s(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 82;
    uint e = k & 255u;
    float d = rd_f16(blk);
    uint j = e >> 3, t = e & 7u, sb = e >> 4;
    uint hb = (blk[66 + (j >> 2)] >> ((j & 3u) * 2u)) & 3u;
    uint idx = (uint)blk[2 + j] | (hb << 8);
    uint bit = (blk[34 + j] >> t) & 1u;
    uint sc = (blk[74 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu;
    float db = d * (0.5f + (float)sc) * 0.25f;
    float gv = (float)(int8_t)TBL[idx * 8u + t];
    return db * gv * (bit ? -1.0f : 1.0f);
}
static inline float dq_iq3_s(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 110;
    uint e = k & 255u;
    float d = rd_f16(blk);
    uint j = e >> 2, t = e & 3u, sb = e >> 5;
    uint hb = (blk[66 + sb] >> (j & 7u)) & 1u;
    uint idx = (uint)blk[2 + j] | (hb << 8);
    uint sc = (blk[106 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu;
    float db = d * (1.0f + 2.0f * (float)sc);
    uint bit = (blk[74 + (e >> 3)] >> (e & 7u)) & 1u;
    float gv = (float)(int8_t)TBL[idx * 4u + t];
    return db * gv * (bit ? -1.0f : 1.0f);
}
static inline float dq_iq3_xxs(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 98;
    uint e = k & 255u;
    float d = rd_f16(blk);
    uint g = e >> 5, p = e & 31u;
    uint j = g * 8u + (p >> 2), t = e & 3u, a = p >> 3;
    const device uchar* Sp = blk + 66 + 4u * g;
    uint S = (uint)Sp[0] | ((uint)Sp[1] << 8) | ((uint)Sp[2] << 16) | ((uint)Sp[3] << 24);
    float db = d * (0.5f + (float)(S >> 28)) * 0.5f;
    uint sidx = (S >> (7u * a)) & 0x7Fu;
    uint bit = (TBL[sidx] >> (e & 7u)) & 1u;
    float gv = (float)(int8_t)TBL[128 + (uint)blk[2 + j] * 4u + t];
    return db * gv * (bit ? -1.0f : 1.0f);
}

static inline float dq_iq4_xs(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 136;
    uint e = k & 255u;
    float d = rd_f16(blk);
    uint sb = e >> 5;
    uint scl = (blk[4 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu;
    uint sch = (((uint)blk[3] << 8 | (uint)blk[2]) >> (2u * sb)) & 3u;
    int sc = (int)(scl | (sch << 4)) - 32;
    uint p = e & 31u;
    uchar b = blk[8 + sb * 16u + (p & 15u)];
    uint nib = (p < 16u) ? (b & 15u) : ((b >> 4) & 15u);
    float kv = (float)KV4[nib];
    return d * (float)sc * kv;
}
static inline float dq_iq1_s(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 50;
    uint e = k & 255u;
    float d = rd_f16(blk);
    uint sb = e >> 5;
    const device uchar* qp = blk + 34 + 2u * sb;
    uint qu = (uint)qp[0] | ((uint)qp[1] << 8);
    float dl = d * (float)(2u * ((qu >> 12) & 7u) + 1u);
    float delta = (qu & 0x8000u) ? -DELTA_IQ1S : DELTA_IQ1S;
    uint l = (e >> 3) & 3u;
    uint t = e & 7u;
    uint hb = (qu >> (3u * l)) & 7u;
    uint j = e >> 3;
    uint idx = (uint)blk[2 + j] | (hb << 8);
    float gv = (float)(int8_t)TBL[idx * 8u + t];
    return dl * (gv + delta);
}
'''

_DQ_DISPATCH = {
    Q8_0: "case 8: v = dq_q8_0(row, k); break;",
    Q2_K: "case 10: v = dq_q2_k(row, k); break;",
    Q3_K: "case 11: v = dq_q3_k(row, k); break;",
    Q4_K: "case 12: v = dq_q4_k(row, k); break;",
    Q5_K: "case 13: v = dq_q5_k(row, k); break;",
    IQ2_XXS: "case 16: v = dq_iq2_xxs(row, k); break;",
    IQ2_XS: "case 17: v = dq_iq2_xs(row, k); break;",
    IQ3_XXS: "case 18: v = dq_iq3_xxs(row, k); break;",
    IQ3_S: "case 21: v = dq_iq3_s(row, k); break;",
    IQ2_S: "case 22: v = dq_iq2_s(row, k); break;",
    IQ4_XS: "case 23: v = dq_iq4_xs(row, k); break;",
    IQ1_S: "case 19: v = dq_iq1_s(row, k); break;",
    Q6_K: "case 14: v = dq_q6_k(row, k); break;",
    IQ4_NL: "case 20: v = dq_iq4_nl(row, k); break;",
}

_NEEDS_TABLE = {IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ3_XXS, IQ1_S}
# Any dtype whose dq_* lives in _IQ_FUNCS must have those functions emitted
# into the header even if it needs no constant table (IQ4_XS uses KV4 but no
# TBL -- without this its kernel source references an undeclared function).
# Note Metal compiles the WHOLE translation unit, so once _IQ_FUNCS is
# emitted the TBL constant must be declared too (dummy size is fine -- the
# table-consuming functions are never dispatched).
_NEEDS_FUNCS = _NEEDS_TABLE | {IQ4_XS, IQ4_NL}   # both consume KV4

# kvalues_iq4nl: the 16 int8 nibble codes shared by IQ4_NL / IQ4_XS
_KVALUES_IQ4NL = [-8, -4, 2, 6, -1, -3, -5, -7, 0, 1, 3, 7, 4, 2, -6, -8]

_ROWBYTES = {
    Q8_0: "(KFULL >> 5) * 34",
    Q2_K: "(KFULL >> 8) * 84",
    Q3_K: "(KFULL >> 8) * 110",
    Q4_K: "(KFULL >> 8) * 144",
    Q5_K: "(KFULL >> 8) * 176",
    IQ2_XXS: "(KFULL >> 8) * 66",
    IQ2_XS: "(KFULL >> 8) * 74",
    IQ2_S: "(KFULL >> 8) * 82",
    IQ3_S: "(KFULL >> 8) * 110",
    IQ3_XXS: "(KFULL >> 8) * 98",
    IQ4_XS: "(KFULL >> 8) * 136",
    IQ1_S: "(KFULL >> 8) * 50",
    Q6_K: "(KFULL >> 8) * 210",
    IQ4_NL: "(KFULL >> 5) * 18",
}

_BODY = """
// One 32-lane SIMD group per output element (was: one thread doing the
// ENTIRE K-length reduction serially -- see perf note in gguf_matmul()).
// Each lane strides over K, accumulates a partial sum, then simd_sum()
// reduces across the group in hardware. ~32x less serial work per output
// element; same dequant math, so results are numerically equivalent
// modulo float accumulation order.
uint gid = thread_position_in_grid.x;
uint out_idx = gid >> 5;         // one SIMD group (32 threads) per element
uint lane = gid & 31u;
if (out_idx >= total) { return; }
uint t = out_idx / NOUT;
uint n = n_base + out_idx % NOUT;
const device uchar* row = w + (size_t)n * ROWBYTES;
const device float* xrow = x + (size_t)t * KFULL + k_base;
// v11.2: 4-way unrolled independent accumulators -- same dequant math and
// the same elements per lane, but 4 FMA+dequant chains in flight instead of
// one dependency-bound chain, so the shader can pipeline loads/ALU.
float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
uint ki = lane;
for (; ki + 96u < KLEN; ki += 128u) {
    a0 += xrow[ki]        * gguf_dequant(row, k_base + ki);
    a1 += xrow[ki + 32u]  * gguf_dequant(row, k_base + ki + 32u);
    a2 += xrow[ki + 64u]  * gguf_dequant(row, k_base + ki + 64u);
    a3 += xrow[ki + 96u]  * gguf_dequant(row, k_base + ki + 96u);
}
float acc = (a0 + a1) + (a2 + a3);
for (; ki < KLEN; ki += 32u) {
    acc += xrow[ki] * gguf_dequant(row, k_base + ki);
}
acc = simd_sum(acc);
if (lane == 0u) {
    out[out_idx] = acc;
}
"""

_kernel_cache: dict[int, object] = {}

# v11.3 fast-path registry (per-format vectorized kernels). Loaded lazily so
# a missing/broken module can never take down the engine.
_FAST_MODS = None
def _fast_impl(dtype_code: int):
    global _FAST_MODS
    import os as _os
    if _os.environ.get("ATF_LEGACY_KERNELS") == "1":
        return None
    if _FAST_MODS is None:
        try:
            from .gguf_fast_iq1s import matmul as m1s
            from .gguf_fast_iq4nl import matmul as m4nl
            from .gguf_fast_iq2 import matmul as m2
            from .gguf_fast_iq34 import matmul as m34
            from .gguf_fast_iq3x import matmul as m3x
            from .gguf_fast_q import matmul as mq
            _FAST_MODS = {IQ2_XXS: m2, IQ2_XS: m2, IQ2_S: m2,
                          IQ3_S: m34, IQ4_XS: m34,
                          Q2_K: mq, Q3_K: mq, Q4_K: mq, Q5_K: mq,
                          IQ3_XXS: lambda x, w, n_out, k_full, dt,
                                   n_base=0, k_base=0, k_len=None:
                               m3x(x, w, n_out, k_full,
                                   n_base=n_base, k_base=k_base, k_len=k_len),
                          IQ1_S: lambda x, w, n_out, k_full, dt,
                                 n_base=0, k_base=0, k_len=None:
                              m1s(x, w, n_out, k_full, dt,
                                  n_base=n_base, k_base=k_base, k_len=k_len),
                          IQ4_NL: lambda x, w, n_out, k_full, dt,
                                  n_base=0, k_base=0, k_len=None:
                              m4nl(x, w, n_out, k_full, dt,
                                   n_base=n_base, k_base=k_base, k_len=k_len),
                          Q6_K: lambda x, w, n_out, k_full, dt,
                                n_base=0, k_base=0, k_len=None:
                              m4nl(x, w, n_out, k_full, dt,
                                   n_base=n_base, k_base=k_base, k_len=k_len)}
        except Exception:
            _FAST_MODS = {}
    return _FAST_MODS.get(dtype_code)

# v10 perf fix: gguf_matmul() previously wrapped 6 plain Python ints into a
# fresh mx.array(...) on every single call. Decode calls the SAME weight
# (same n_out/k_full/k_len/n_base/k_base, T always 1) every token, so these
# scalar args are identical call after call -- cache the wrapped mx.array
# tuple instead of rebuilding it ~350x/token.
_scalar_args_cache: dict[tuple, tuple] = {}


def _scalar_args(T: int, n_out: int, k_full: int, k_len: int,
                 n_base: int, k_base: int):
    key = (T, n_out, k_full, k_len, n_base, k_base)
    ent = _scalar_args_cache.get(key)
    if ent is None:
        ent = (mx.array(T * n_out), mx.array(n_out), mx.array(k_full),
               mx.array(k_len), mx.array(n_base), mx.array(k_base))
        _scalar_args_cache[key] = ent
    return ent


def _build_kernel(dtype_code: int):
    if dtype_code in _kernel_cache:
        return _kernel_cache[dtype_code]

    needs_tbl = dtype_code in _NEEDS_TABLE
    tbl_bytes = _table_bytes(dtype_code) if needs_tbl else (
        b"\x00" * 256 if dtype_code in _NEEDS_FUNCS else b"\x00")
    header = (
        "constant uint DT = %d;\n" % dtype_code
        + ("constant uchar TBL[%d] = {%s};\n"
           % (len(tbl_bytes), ",".join("0x%02x" % b for b in tbl_bytes)))
        + _HEADER
        + (_IQ_FUNCS if dtype_code in _NEEDS_FUNCS else "")
        + "static inline float gguf_dequant(const device uchar* row, uint k  ) {\n"
        + "    float v = 0.f;\n"
        + "    switch (DT) {\n        "
        + _DQ_DISPATCH[dtype_code]
        + "\n    }\n    return v;\n}\n"
    )
    body = _BODY.replace("ROWBYTES", "(%s)" % _ROWBYTES[dtype_code])

    input_names = ["x", "w", "total", "NOUT", "KFULL", "KLEN", "n_base", "k_base"]
    kern = mx.fast.metal_kernel(
        name=f"gguf_mm_{dtype_code}",
        input_names=input_names,
        output_names=["out"],
        header=header,
        source=body,
        ensure_row_contiguous=False,
    )
    _kernel_cache[dtype_code] = kern
    return kern


def gguf_matmul(
    x: mx.array,
    w_bytes: mx.array,
    n_out: int,
    k_full: int,
    dtype_code: int,
    n_base: int = 0,
    k_base: int = 0,
    k_len: int | None = None,
) -> mx.array:
    """y[T, n_out] = x[T, k_len] @ W[n_base:n_base+n_out, k_base:k_base+k_len].

    x: f32 [T, >=k_base+k_len]; w_bytes: uint8 payload (verbatim GGUF blocks).
    A 1-D x[k] is treated as a single row and a 1-D [n_out] result is
    returned (matches the dense bf16 `vec @ W` path in model.mm).
    """
    # v11.3: dispatch to per-format vectorized kernels where available
    # (2.5-4x faster inner loops, validated vs canonical decoders).
    # Set ATF_LEGACY_KERNELS=1 to force the generic scalar path.
    fast = _fast_impl(dtype_code)
    if fast is not None:
        eff_k_len = k_len if k_len is not None else k_full - k_base
        if dtype_code == IQ3_XXS and (k_base % 32 or eff_k_len % 32):
            pass                      # iq3x fast path needs 32-aligned k
        else:
            return fast(x, w_bytes, n_out, k_full, dtype_code,
                        n_base=n_base, k_base=k_base, k_len=k_len)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    T = x.shape[0]
    if k_len is None:
        k_len = k_full - k_base
    kern = _build_kernel(dtype_code)
    # v11 perf fix: threadgroup bumped 256 -> 1024 (32 simdgroups instead of
    # 8) -- same total thread count (grid unchanged), but each dispatch now
    # covers 32 output elements per threadgroup instead of 8, improving GPU
    # occupancy/scheduling per launch. Must stay a multiple of 32 (SIMD
    # width) -- 1024 is also the max threadgroup size on Apple GPUs.
    # v10 perf fix: one 32-lane SIMD group per output element (see _BODY) --
    # grid is total_output_elements * 32, NOT total_output_elements.
    # Scalar args cached across calls (see _scalar_args) -- decode repeats
    # identical (T, n_out, k_full, k_len, n_base, k_base) every token.
    total_a, nout_a, kfull_a, klen_a, nbase_a, kbase_a = _scalar_args(
        T, n_out, k_full, k_len, n_base, k_base)
    out = kern(
        inputs=[x.astype(mx.float32), w_bytes,
                total_a, nout_a, kfull_a, klen_a, nbase_a, kbase_a],
        grid=(T * n_out * 32, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(T * n_out,)],
        output_dtypes=[mx.float32],
        verbose=False,
    )
    out = out[0].reshape(T, n_out)
    return out[0] if squeeze else out


def supported(dtype_code: int) -> bool:
    return dtype_code in SUPPORTED_DTYPES

_GATHER_HEADER_EXTRA = """
"""

def gguf_gather(w_bytes: mx.array, ids: mx.array, k_full: int, dtype_code: int,
                k_base: int = 0, k_len: int | None = None) -> mx.array:
    """out[T, K] = dequant rows of W selected by ids (row index = out dim).

    Used for token-embedding lookup: the stored tensor is [vocab, in] rows,
    ids are vocab indices.

    v16: optional k_base/k_len restrict the K range written per row (reads
    still touch only the requested bytes' blocks). This lets prefill
    materialize a K-slice of a record WITHOUT first materializing the full
    row width -- down-proj records stack every expert along K, so a full-
    width gather there would transiently allocate gigabytes of f32.
    """
    if k_len is None:
        k_len = int(k_full)
    T = int(ids.shape[0])
    total = T * int(k_len)
    kern = _build_gather(dtype_code)
    out = kern(
        inputs=[w_bytes, ids.astype(mx.uint32), mx.array(total),
                mx.array(int(k_len)), mx.array(int(k_base)),
                mx.array(int(k_full))],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
        verbose=False,
    )
    return out[0].reshape(T, int(k_len))

_gather_cache: dict[int, object] = {}

def _build_gather(dtype_code: int):
    if dtype_code in _gather_cache:
        return _gather_cache[dtype_code]
    needs_tbl = dtype_code in _NEEDS_TABLE
    tbl_bytes = _table_bytes(dtype_code) if needs_tbl else (
        b"\x00" * 256 if dtype_code in _NEEDS_FUNCS else b"\x00")
    header = (
        "constant uint DT = %d;\n" % dtype_code
        + (("constant uchar TBL[%d] = {%s};\n"
           % (len(tbl_bytes), ",".join("0x%02x" % b for b in tbl_bytes))))
        + _HEADER
        + (_IQ_FUNCS if dtype_code in _NEEDS_FUNCS else "")
        + "static inline float gguf_dequant(const device uchar* row, uint k) {\n"
        + "    float v = 0.f;\n"
        + "    switch (DT) {\n        "
        + _DQ_DISPATCH[dtype_code]
        + "\n    }\n    return v;\n}\n"
    )
    body = """
uint idx = thread_position_in_grid.x;
if (idx >= total) { return; }
uint t = idx / KLEN;
uint k = idx % KLEN;
const device uchar* row = w + (size_t)ids[t] * ROWBYTES;
out[idx] = gguf_dequant(row, KBASE + k);
""".replace("ROWBYTES", "(%s)" % _ROWBYTES[dtype_code])
    kern = mx.fast.metal_kernel(
        name=f"gguf_gather_{dtype_code}",
        input_names=["w", "ids", "total", "KLEN", "KBASE", "KFULL"],
        output_names=["out"],
        header=header,
        source=body,
        ensure_row_contiguous=False,
    )
    _gather_cache[dtype_code] = kern
    return kern
