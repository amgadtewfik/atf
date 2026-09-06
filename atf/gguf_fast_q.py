"""Optimized GGUF dequant-matmul Metal kernels -- ARITHMETIC formats
(Q2_K first-class, plus Q4_K / Q5_K / Q3_K / Q8_0).

API mirrors atf.gguf_metal.gguf_matmul:
    matmul(x, w_bytes, n_out, k_full, dtype_code,
           n_base=0, k_base=0, k_len=None)

vs the production baseline (one ELEMENT per lane per iteration):
  * each lane decodes one contiguous 32-element SUB-BLOCK per iteration
    (stride 1024 across the SIMD group); block scale fetched once;
  * x is consumed via dot(float4, float4);
  * Q2_K/Q4_K/Q5_K load packed codes with aligned uint32 words instead of
    byte loads (superblock sizes 84/144/176 and window offsets are all
    multiples of 4/16, so windows keep natural alignment when the weight
    buffer itself is 4-byte aligned -- true for every Metal allocation);
  * Q2_K gotcha found while validating: the 2-bit shift wraps mod 128
    within a super-block (pair index uses e & 127).
Assumptions: w_bytes buffer base 4-byte aligned; K a multiple of 256
(like the production kernel). Any k_base/k_len slice is supported; a
32-misaligned k_base falls back to an element-strided path.
"""
from __future__ import annotations

import mlx.core as mx

Q8_0, Q2_K, Q3_K, Q4_K, Q5_K = 8, 10, 11, 12, 13
SUPPORTED = {Q8_0, Q2_K, Q3_K, Q4_K, Q5_K}

_ROWBYTES = {
    Q8_0: "(KFULL >> 5) * 34",
    Q2_K: "(KFULL >> 8) * 84",
    Q3_K: "(KFULL >> 8) * 110",
    Q4_K: "(KFULL >> 8) * 144",
    Q5_K: "(KFULL >> 8) * 176",
}

_HELPERS = r"""
static inline float rd_f16(const device uchar* p) {
    ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(__builtin_bit_cast(half, u));
}
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
/* scalar per-element decoders: used ONLY for ragged edges / misaligned
   k_base fallback. Bit-exact ports of the validated baseline formulas. */
static inline float dq_q8_0(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 5) * 34;
    float d = rd_f16(blk);
    int8_t q = as_type<int8_t>(blk[2 + (k & 31)]);
    return d * float(q);
}
static inline float dq_q2_k(const device uchar* row, uint k) {
    const device uchar* sb = row + (k >> 8) * 84;
    uint e = k & 255u;
    uint hsel = e >> 7;
    uint ih = e & 127u;
    uint pair = ih >> 5;
    uint pos = ih & 15u;
    uint sel = (ih & 31u) >= 16u ? 16u : 0u;
    uint q = (sb[16 + hsel * 32 + sel + pos] >> (pair * 2)) & 3u;
    uchar scb = sb[e >> 4];
    return rd_f16(sb + 80) * float(scb & 15u) * float(q)
         - rd_f16(sb + 82) * float(scb >> 4);
}
static inline float dq_q4_k(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 144;
    uint e = k & 255u;
    uint chunk = e >> 6;
    uint halfw = (e >> 5) & 1u;
    uint pos = e & 31u;
    uchar sc, mn;
    get_scale_min(blk + 4, e >> 5, &sc, &mn);
    uchar b = blk[16 + chunk * 32 + pos];
    uint q = halfw ? (b >> 4) : (b & 15u);
    return rd_f16(blk) * float(sc) * float(q) - rd_f16(blk + 2) * float(mn);
}
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
    uint hbit = (blk[16 + (e & 31u)] >> (e / 32u)) & 1u;
    uint q = qlo | (hbit << 4);
    return rd_f16(blk) * float(sc) * float(q) - rd_f16(blk + 2) * float(mn);
}
static inline float dq_q3_k(const device uchar* row, uint k) {
    const device uchar* blk = row + (k >> 8) * 110;
    uint e = k & 255u;
    uint a = e >> 7;
    uint s = (e >> 5) & 3u;
    uint p = e & 31u;
    uint qlo = (blk[32 + a * 32 + p] >> (2 * s)) & 3u;
    uint hbit = (blk[e & 31u] >> (e / 32u)) & 1u;
    int q = int(qlo) - (int(hbit ^ 1u) << 2);
    uint j = e >> 4;
    uint lo = (blk[96 + (j & 7u)] >> ((j >> 3) * 4)) & 0xFu;
    uint hi2 = (blk[104 + (j & 3u)] >> (2 * (j >> 2))) & 3u;
    int sc = int(lo | (hi2 << 4)) - 32;
    return rd_f16(blk + 108) * float(sc) * float(q);
}
"""

_CHUNK_Q2_K = r"""
/* kb % 32 == 0. One chunk = constant 2-bit shift (WRAPS mod 128!), one
   contiguous 32-byte window, exactly two 16-element scale/min pairs. */
static inline float dot_chunk(const device uchar* row, uint kb,
                              const device float* xf) {
    const device uchar* sb = row + (kb >> 8) * 84;
    uint b = kb & 255u;
    uint shift = ((b & 127u) >> 4) & 14u;
    const device uint* qp = (const device uint*)(sb + 16u + (b >> 7) * 32u);
    uchar sa = sb[b >> 4];
    uchar scb = sb[(b >> 4) + 1u];
    float d = rd_f16(sb + 80), dmin = rd_f16(sb + 82);
    float ds0 = d * float(sa & 15u), dm0 = dmin * float(sa >> 4);
    float ds1 = d * float(scb & 15u), dm1 = dmin * float(scb >> 4);
    const device float4* xp = (const device float4*)xf;
    float4 one(1.f);
    float s0 = 0.f, sx0 = 0.f, s1 = 0.f, sx1 = 0.f;
    #pragma unroll(4)
    for (uint g = 0u; g < 4u; ++g) {
        float4 xv = xp[g];
        uint v = qp[g] >> shift;
        float4 qv(float(v & 3u), float((v >> 8) & 3u),
                  float((v >> 16) & 3u), float((v >> 24) & 3u));
        s0 += dot(xv, qv);
        sx0 += dot(xv, one);
        v = qp[g + 4u] >> shift;
        float4 xb = xp[g + 4u];
        qv = float4(float(v & 3u), float((v >> 8) & 3u),
                    float((v >> 16) & 3u), float((v >> 24) & 3u));
        s1 += dot(xb, qv);
        sx1 += dot(xb, one);
    }
    return ds0*s0 - dm0*sx0 + ds1*s1 - dm1*sx1;
}
"""

_CHUNK_Q4_K = r"""
/* kb % 32 == 0: constant nibble side, ONE scale/min pair per chunk. */
static inline float dot_chunk(const device uchar* row, uint kb,
                              const device float* xf) {
    const device uchar* blk = row + (kb >> 8) * 144;
    uint e = kb & 255u;
    uint side = (e >> 5) & 1u;
    uchar sc, mn;
    get_scale_min(blk + 4, e >> 5, &sc, &mn);
    float ds = rd_f16(blk) * float(sc);
    float dm = rd_f16(blk + 2) * float(mn);
    const device uint* qp = (const device uint*)(blk + 16u + ((e >> 6) << 5));
    const device float4* xp = (const device float4*)xf;
    float4 one(1.f);
    float s = 0.f, sx = 0.f;
    #pragma unroll(8)
    for (uint g = 0u; g < 8u; ++g) {
        float4 xv = xp[g];
        uint v = side ? ((qp[g] >> 4) & 0x0F0F0F0Fu) : (qp[g] & 0x0F0F0F0Fu);
        float4 qv(float(v & 15u), float((v >> 8) & 15u),
                  float((v >> 16) & 15u), float((v >> 24) & 15u));
        s += dot(xv, qv);
        sx += dot(xv, one);
    }
    return ds * s - dm * sx;
}
"""

_CHUNK_Q5_K = r"""
/* Q4_K plus one high bit per element from the t-major qh plane. */
static inline float dot_chunk(const device uchar* row, uint kb,
                              const device float* xf) {
    const device uchar* blk = row + (kb >> 8) * 176;
    uint e = kb & 255u;
    uint side = (e >> 5) & 1u;
    uchar sc, mn;
    get_scale_min(blk + 4, e >> 5, &sc, &mn);
    float ds = rd_f16(blk) * float(sc);
    float dm = rd_f16(blk + 2) * float(mn);
    const device uint* qp = (const device uint*)(blk + 48u + ((e >> 6) << 5));
    const device uint* hp = (const device uint*)(blk + 16u);
    uint bitpos = e >> 5;
    const device float4* xp = (const device float4*)xf;
    float4 one(1.f);
    float s = 0.f, sh = 0.f, sx = 0.f;
    #pragma unroll(8)
    for (uint g = 0u; g < 8u; ++g) {
        float4 xv = xp[g];
        uint v = side ? ((qp[g] >> 4) & 0x0F0F0F0Fu) : (qp[g] & 0x0F0F0F0Fu);
        float4 qv(float(v & 15u), float((v >> 8) & 15u),
                  float((v >> 16) & 15u), float((v >> 24) & 15u));
        uint uh = hp[g] >> bitpos;
        float4 hv(float(uh & 1u), float((uh >> 8) & 1u),
                  float((uh >> 16) & 1u), float((uh >> 24) & 1u));
        s += dot(xv, qv);
        sh += dot(xv, hv);
        sx += dot(xv, one);
    }
    /* value = ds*(nib | h<<4) - dm ; nib|h<<4 == nib + 16*h */
    return ds * (s + 16.f * sh) - dm * sx;
}
"""

_CHUNK_Q3_K = r"""
/* kb % 32 == 0: constant qs shift + hmask plane pos; TWO 6-bit scales.
   Superblocks are 110 bytes (not 4-aligned), so byte loads here. */
static inline float q3_sc6(const device uchar* blk, uint m) {
    uint lo = (blk[96 + (m & 7u)] >> ((m >> 3) * 4)) & 0xFu;
    uint hi2 = (blk[104 + (m & 3u)] >> (2 * (m >> 2))) & 3u;
    return float(int(lo | (hi2 << 4)) - 32);
}
static inline float dot_chunk(const device uchar* row, uint kb,
                              const device float* xp) {
    const device uchar* blk = row + (kb >> 8) * 110;
    uint e = kb & 255u;
    uint shift = 2u * ((e >> 5) & 3u);
    const device uchar* qp = blk + 32u + (e >> 7) * 32u;
    const device uchar* hp = blk;
    uint bitpos = e >> 5;
    float d = rd_f16(blk + 108);
    float sc0 = q3_sc6(blk, e >> 4);
    float sc1 = q3_sc6(blk, (e >> 4) + 1u);
    float s0 = 0.f, hb0 = 0.f, sx0 = 0.f;
    float s1 = 0.f, hb1 = 0.f, sx1 = 0.f;
    for (uint g = 0u; g < 8u; ++g) {
        // elements 4g..4g+3 live in qs/hmask bytes 4g..4g+3
        uint b0 = 4u * g;
        float4 xv = *(const device float4*)(xp + b0);
        float4 one(1.f);
        float4 qv(float((qp[b0]      >> shift) & 3u),
                  float((qp[b0 + 1u] >> shift) & 3u),
                  float((qp[b0 + 2u] >> shift) & 3u),
                  float((qp[b0 + 3u] >> shift) & 3u));
        float4 hv(float((hp[b0]      >> bitpos) & 1u),
                  float((hp[b0 + 1u] >> bitpos) & 1u),
                  float((hp[b0 + 2u] >> bitpos) & 1u),
                  float((hp[b0 + 3u] >> bitpos) & 1u));
        if (g < 4u) {
            s0 += dot(xv, qv); hb0 += dot(xv, hv); sx0 += dot(xv, one);
        } else {
            s1 += dot(xv, qv); hb1 += dot(xv, hv); sx1 += dot(xv, one);
        }
    }
    /* value = d * sc * (qlo + 4*hbit - 4) */
    return d * (sc0 * (s0 + 4.f*hb0 - 4.f*sx0)
              + sc1 * (s1 + 4.f*hb1 - 4.f*sx1));
}
"""

_CHUNK_Q8_0 = r"""
/* One 34-byte block per chunk; qs not word-aligned -> byte loads. */
static inline float dot_chunk(const device uchar* row, uint kb,
                              const device float* xf) {
    const device uchar* blk = row + (kb >> 5) * 34;
    float d = rd_f16(blk);
    const device float4* xp = (const device float4*)xf;
    float s = 0.f;
    #pragma unroll(8)
    for (uint g = 0u; g < 8u; ++g) {
        float4 qv(float(as_type<int8_t>(blk[2 + 4u*g])),
                  float(as_type<int8_t>(blk[3 + 4u*g])),
                  float(as_type<int8_t>(blk[4 + 4u*g])),
                  float(as_type<int8_t>(blk[5 + 4u*g])));
        s += dot(xp[g], qv);
    }
    return d * s;
}
"""

_CHUNKS = {Q2_K: _CHUNK_Q2_K, Q4_K: _CHUNK_Q4_K, Q5_K: _CHUNK_Q5_K,
           Q3_K: _CHUNK_Q3_K, Q8_0: _CHUNK_Q8_0}

_DQ_EDGE = {Q2_K: "dq_q2_k", Q4_K: "dq_q4_k", Q5_K: "dq_q5_k",
            Q3_K: "dq_q3_k", Q8_0: "dq_q8_0"}

_BODY_TMPL = r"""
uint gid = thread_position_in_grid.x;
uint out_idx = gid >> 5;
uint lane = gid & 31u;
if (out_idx >= total) { return; }
uint t = out_idx / NOUT;
uint n = n_base + out_idx % NOUT;
const device uchar* row = w + (size_t)n * ROWBYTES;
const device float* xr = x + (size_t)t * KFULL;
uint kend = k_base + KLEN;
uint lo = (k_base + 31u) & ~31u;
uint hi = kend & ~31u;
float acc = 0.f;
if ((k_base & 31u) != 0u) {
    /* misaligned slice: element-strided fallback (correct for any offset) */
    for (uint k = k_base + lane; k < kend; k += 32u)
        acc += xr[k] * DQF(row, k);
} else {
    for (uint c = lo + (lane << 5); c < hi; c += 1024u)
        acc += dot_chunk(row, c, xr + c);
    if (lane < (lo - k_base)) acc += xr[k_base + lane] * DQF(row, k_base + lane);
    if (lane < (kend - hi))   acc += xr[hi + lane]       * DQF(row, hi + lane);
}
acc = simd_sum(acc);
if (lane == 0u) out[out_idx] = acc;
"""

_kernel_cache: dict[int, object] = {}
_args_cache: dict[tuple, tuple] = {}


def _scalar_args(T, n_out, k_full, k_len, n_base, k_base):
    key = (T, n_out, k_full, k_len, n_base, k_base)
    ent = _args_cache.get(key)
    if ent is None:
        ent = (mx.array(T * n_out), mx.array(n_out), mx.array(k_full),
               mx.array(k_len), mx.array(n_base), mx.array(k_base))
        _args_cache[key] = ent
    return ent


def _build_kernel(dtype_code: int):
    if dtype_code in _kernel_cache:
        return _kernel_cache[dtype_code]
    header = _HELPERS + _CHUNKS[dtype_code]
    body = (_BODY_TMPL
            .replace("ROWBYTES", "(%s)" % _ROWBYTES[dtype_code])
            .replace("DQF", _DQ_EDGE[dtype_code]))
    kern = mx.fast.metal_kernel(
        name=f"gguf_mm_opt_{dtype_code}",
        input_names=["x", "w", "total", "NOUT", "KFULL", "KLEN",
                     "n_base", "k_base"],
        output_names=["out"],
        header=header,
        source=body,
        ensure_row_contiguous=False,
    )
    _kernel_cache[dtype_code] = kern
    return kern


def matmul(x, w_bytes, n_out, k_full, dtype_code,
           n_base=0, k_base=0, k_len=None):
    """y[T, n_out] = x[T, k_len] @ W[n_base:n_base+n_out, k_base:k_base+k_len]
    with W stored as verbatim GGUF quantized payload. Matches
    atf.gguf_metal.gguf_matmul semantics (incl. 1-D squeeze)."""
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    T = x.shape[0]
    if k_len is None:
        k_len = k_full - k_base
    kern = _build_kernel(int(dtype_code))
    ta, na, ka, kl, nb, kb = _scalar_args(T, n_out, k_full, k_len,
                                          int(n_base), int(k_base))
    out = kern(
        inputs=[x.astype(mx.float32), w_bytes, ta, na, ka, kl, nb, kb],
        grid=(T * n_out * 32, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(T * n_out,)],
        output_dtypes=[mx.float32],
        verbose=False,
    )
    out = out[0].reshape(T, n_out)
    return out[0] if squeeze else out
