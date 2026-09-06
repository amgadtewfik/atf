"""Optimized GGUF dequant-matmul Metal kernels for IQ4_NL (=20) and Q6_K (=14).

matmul(x, w_bytes, n_out, k_full, dtype_code, n_base=0, k_base=0, k_len=None)
mirrors atf.gguf_metal.gguf_matmul semantics for these two formats.

Design (same family as gguf_fast_q.py):
  * each lane decodes one contiguous 32-element chunk per iteration;
    block/superblock headers fetched once per chunk;
  * x consumed via dot(float4,float4);
  * weight bytes loaded via ld_u4(): two 2-byte ushort loads assembled into
    an uchar4. Needed because neither format keeps 4-byte row alignment
    (IQ4_NL rows are 18*B bytes, Q6_K rows 210 bytes); ushort loads only
    need the 2-byte alignment every GGUF block layout guarantees.
IQ4_NL: 18 B / 32 elems: d f16 + 16 nibble bytes; value = d*kvalues[nib].
Q6_K:   210 B / 256 elems: ql[128] | qh[64] | int8 scales[16] | d f16;
        6-bit code = lo-nibble | high2<<4 - 32; 16 scale groups of 16 elems.
Any k_base % 32 != 0 falls back to an element-wise scalar path.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

IQ4_NL = 20
Q6_K = 14

_RD_F16 = """
static inline float rd_f16(const device uchar* p) {
    ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(__builtin_bit_cast(half, u));
}
/* 2-byte-aligned safe uchar4 load: two ushort halves (rows of these formats
   are not 4-byte multiples, so a plain uchar4 load can straddle). */
static inline uchar4 ld_u4(const device uchar* p) {
    ushort lo = *(device const ushort*)(p);
    ushort hi = *(device const ushort*)(p + 2);
    return uchar4(lo & 0xFFu, lo >> 8, hi & 0xFFu, hi >> 8);
}
"""

# ---- IQ4_NL ----------------------------------------------------------------
_KV = """
constant int8_t KV[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
"""

_SCALAR_IQ4NL = r"""
static inline float dot_scalar_iq4nl(const device uchar* row,
                                     const device float* xr, uint k0, uint kn,
                                     uint koff) {
    float acc = 0.f;
    for (uint k = k0; k < kn; k += 32u) {
        uint ka = koff + k;
        const device uchar* blk = row + (ka >> 5) * 18u;
        float d = rd_f16(blk);
        uint e = ka & 31u;
        uint qb = blk[2u + (e & 15u)];
        uint nib = (e < 16u) ? (qb & 0xFu) : (qb >> 4);
        acc += xr[k] * d * (float)KV[nib];
    }
    return acc;
}
"""

_CHUNK_IQ4NL = r"""
    {   // IQ4_NL chunk: one 32-elem block; elements 0-15 = LOW nibbles of
        // qs[0..15], elements 16-31 = HIGH nibbles (gguf-py hsplit order)
        const device uchar* qs = blk + 2u;
        uchar4 b0 = ld_u4(qs);
        uchar4 b1 = ld_u4(qs + 4u);
        uchar4 b2 = ld_u4(qs + 8u);
        uchar4 b3 = ld_u4(qs + 12u);
        float4 lo0; lo0[0]=(int8_t)KV[b0[0]&15u]; lo0[1]=(int8_t)KV[b0[1]&15u];
                    lo0[2]=(int8_t)KV[b0[2]&15u]; lo0[3]=(int8_t)KV[b0[3]&15u];
        float4 lo1; lo1[0]=(int8_t)KV[b1[0]&15u]; lo1[1]=(int8_t)KV[b1[1]&15u];
                    lo1[2]=(int8_t)KV[b1[2]&15u]; lo1[3]=(int8_t)KV[b1[3]&15u];
        float4 lo2; lo2[0]=(int8_t)KV[b2[0]&15u]; lo2[1]=(int8_t)KV[b2[1]&15u];
                    lo2[2]=(int8_t)KV[b2[2]&15u]; lo2[3]=(int8_t)KV[b2[3]&15u];
        float4 lo3; lo3[0]=(int8_t)KV[b3[0]&15u]; lo3[1]=(int8_t)KV[b3[1]&15u];
                    lo3[2]=(int8_t)KV[b3[2]&15u]; lo3[3]=(int8_t)KV[b3[3]&15u];
        acc += d * (dot(*(device const float4*)xrel,       lo0)
                  + dot(*(device const float4*)(xrel+4),   lo1)
                  + dot(*(device const float4*)(xrel+8),   lo2)
                  + dot(*(device const float4*)(xrel+12),  lo3));
        xrel += 16;
        float4 hi0; hi0[0]=(int8_t)KV[b0[0]>>4]; hi0[1]=(int8_t)KV[b0[1]>>4];
                    hi0[2]=(int8_t)KV[b0[2]>>4]; hi0[3]=(int8_t)KV[b0[3]>>4];
        float4 hi1; hi1[0]=(int8_t)KV[b1[0]>>4]; hi1[1]=(int8_t)KV[b1[1]>>4];
                    hi1[2]=(int8_t)KV[b1[2]>>4]; hi1[3]=(int8_t)KV[b1[3]>>4];
        float4 hi2; hi2[0]=(int8_t)KV[b2[0]>>4]; hi2[1]=(int8_t)KV[b2[1]>>4];
                    hi2[2]=(int8_t)KV[b2[2]>>4]; hi2[3]=(int8_t)KV[b2[3]>>4];
        float4 hi3; hi3[0]=(int8_t)KV[b3[0]>>4]; hi3[1]=(int8_t)KV[b3[1]>>4];
                    hi3[2]=(int8_t)KV[b3[2]>>4]; hi3[3]=(int8_t)KV[b3[3]>>4];
        acc += d * (dot(*(device const float4*)xrel,       hi0)
                  + dot(*(device const float4*)(xrel+4),   hi1)
                  + dot(*(device const float4*)(xrel+8),   hi2)
                  + dot(*(device const float4*)(xrel+12),  hi3));
        xrel += 16;
    }
"""

# ---- Q6_K ------------------------------------------------------------------
_SCALAR_Q6K = r"""
static inline float dot_scalar_q6k(const device uchar* row,
                                   const device float* xr, uint k0, uint kn,
                                   uint koff) {
    float acc = 0.f;
    for (uint k = k0; k < kn; k += 32u) {
        uint ka = koff + k;
        const device uchar* blk = row + (ka >> 8) * 210u;
        uint e = ka & 255u;
        uint n = e >> 7;                 // 128-elem half of the superblock
        uint p = e & 127u;
        uint j = p & 31u, quad = p >> 5;
        uint nibsh = (quad & 2u) ? 4u : 0u;
        uint hbsh = quad << 1;
        const device uchar* ql = blk + (n << 6) + ((quad & 1u) << 5);
        int q = (int)((ql[j] >> nibsh) & 0xFu)
              + (int)(((blk[128u + (n << 5) + j] >> hbsh) & 3u) << 4) - 32;
        int sc = (int)(int8_t)blk[192u + (n << 3) + (quad << 1) + (j >> 4)];
        float d = rd_f16(blk + 208u);
        acc += xr[k] * d * (float)sc * (float)q;
    }
    return acc;
}
"""

_CHUNK_Q6K = r"""
    {   // Q6_K chunk: 32 consecutive elems share one ql byte run + fixed quad
        uint n = (e0 >> 7) & 1u;         // superblock half
        uint quad = (e0 & 127u) >> 5;    // 0..3 within the half
        uint nibsh = (quad & 2u) ? 4u : 0u;
        uint hbsh = quad << 1;
        const device uchar* qlb = blk + (n << 6) + ((quad & 1u) << 5);
        const device uchar* qhb = blk + 128u + (n << 5);
        const device uchar* scb = blk + 192u + (n << 3) + (quad << 1);
        float d = rd_f16(blk + 208u);
        float s0 = d * (float)(int8_t)scb[0];   // scale idx = n*8 + quad*2 + (j>>4)
        float s1 = d * (float)(int8_t)scb[1];
        for (uint j = 0u; j < 16u; j += 4u) {
            float4 qlo = (float4)(((ld_u4(qlb + j) >> nibsh) & 15u));
            float4 h   = (float4)(((ld_u4(qhb + j) >> hbsh) & 3u));
            float4 q   = qlo + 16.f * h - 32.f;
            acc += s0 * dot(*(device const float4*)xrel, q);
            xrel += 4;
        }
        for (uint j = 16u; j < 32u; j += 4u) {
            float4 qlo = (float4)(((ld_u4(qlb + j) >> nibsh) & 15u));
            float4 h   = (float4)(((ld_u4(qhb + j) >> hbsh) & 3u));
            float4 q   = qlo + 16.f * h - 32.f;
            acc += s1 * dot(*(device const float4*)xrel, q);
            xrel += 4;
        }
    }
"""

# shared body template --------------------------------------------------------
_BODY_TMPL = """
uint gid = thread_position_in_grid.x;
uint out_idx = gid >> 5u;
uint lane = gid & 31u;

if (out_idx >= total) {{ return; }}
uint t_row = out_idx / NOUT;
uint n = n_base + out_idx % NOUT;
const device uchar* row = w + (size_t)n * ({ROWBYTES});
const device float* xrow = x + (size_t)t_row * KFULL + k_base;

float acc = 0.f;
if ((k_base & 31u) == 0u) {{
    uint nfull = KLEN >> 5;
    for (uint c = lane; c < nfull; c += 32u) {{
        uint ka = k_base + (c << 5);           // absolute k of chunk start
        uint e0 = ka & 255u;
        const device uchar* blk = row + (ka >> {SHIFT}) * {BLOCK}u;
        const device float* xrel = xrow + (ka - k_base);
        float d = rd_f16(blk{D_OFF});
{CHUNK}
    }}
    uint tail0 = nfull << 5;
    for (uint tk = tail0 + lane; tk < KLEN; tk += 32u) {{
        acc += {SCALAR}(row, xrow, tk, tk + 1u, k_base);
    }}
}} else {{
    acc = {SCALAR}(row, xrow, lane, KLEN, k_base);
}}
acc = simd_sum(acc);
if (lane == 0u) {{ out[out_idx] = acc; }}
"""

_FORMATS = {
    IQ4_NL: dict(
        name="gguf_mm_o_iq4_nl",
        rowbytes="(KFULL >> 5) * 18", shift="5", block=18,
        d_off="", chunk=_CHUNK_IQ4NL, scalar="dot_scalar_iq4nl",
        scalar_src=_SCALAR_IQ4NL,
        header=_KV,
    ),
    Q6_K: dict(
        name="gguf_mm_o_q6_k",
        rowbytes="(KFULL >> 8) * 210", shift="8", block=210,
        d_off=" + 208u", chunk=_CHUNK_Q6K, scalar="dot_scalar_q6k",
        scalar_src=_SCALAR_Q6K,
        header="",
    ),
}

_kernel_cache: dict[int, object] = {}


def _build_kernel(dtype_code: int):
    if dtype_code in _kernel_cache:
        return _kernel_cache[dtype_code]
    f = _FORMATS[dtype_code]
    body = _BODY_TMPL.format(
        ROWBYTES=f["rowbytes"], SHIFT=f["shift"], BLOCK=f["block"],
        D_OFF=f["d_off"], CHUNK=f["chunk"], SCALAR=f["scalar"],
    )
    kern = mx.fast.metal_kernel(
        name=f["name"],
        input_names=["x", "w", "total", "NOUT", "KFULL", "KLEN",
                     "n_base", "k_base"],
        output_names=["out"],
        header=_RD_F16 + f["header"] + f["scalar_src"],
        source=body,
        ensure_row_contiguous=False,
    )
    _kernel_cache[dtype_code] = kern
    return kern


_scal_cache: dict[tuple, tuple] = {}


def matmul(
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

    Semantics match atf.gguf_metal.gguf_matmul for IQ4_NL / Q6_K."""
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    T = x.shape[0]
    if k_len is None:
        k_len = k_full - k_base
    kern = _build_kernel(dtype_code)
    key = (T, n_out, k_full, k_len, n_base, k_base)
    args = _scal_cache.get(key)
    if args is None:
        args = (mx.array(T * n_out), mx.array(n_out), mx.array(k_full),
                mx.array(k_len), mx.array(n_base), mx.array(k_base))
        _scal_cache[key] = args
    out = kern(
        inputs=[x.astype(mx.float32), w_bytes, *args],
        grid=(T * n_out * 32, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(T * n_out,)],
        output_dtypes=[mx.float32],
        verbose=False,
    )[0].reshape(T, n_out)
    return out[0] if squeeze else out


SUPPORTED = {IQ4_NL, Q6_K}
