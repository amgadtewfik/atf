"""Optimized GGUF dequant-matmul Metal kernels for IQ2_S / IQ2_XXS / IQ2_XS.

vs. atf/gguf_metal.py baseline:
  * each of the 32 lanes in an output element's SIMD group decodes 32
    CONSECUTIVE elements per iteration (baseline: stride-32, one element),
    so block scales / d / packed codes are loaded once per chunk and
    codebook rows are read as uchar4 vector loads from THREADGROUP memory.
  * grid + sign tables are copied once per threadgroup into threadgroup
    arrays at kernel start (baseline: constant address space lookups).
  * sign bits are pre-expanded on the host into a 1024-byte mask table
    (mask[i*8+t] = bit ? 255 : 0); applying a sign is branchless:
    (char4)(grid ^ msk) + (char4)(msk).

matmul(x, w_bytes, n_out, k_full, dtype_code, n_base, k_base, k_len)
matches atf.gguf_metal.gguf_matmul semantics (incl. 1-D squeeze) for
dtype_code in {IQ2_XXS=16, IQ2_XS=17, IQ2_S=22}. Any k_base not a
multiple of 32 falls back to a correct scalar path inside the kernel.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

IQ2_XXS = 16
IQ2_XS = 17
IQ2_S = 22

_RD_F16 = """
static inline float rd_f16(const device uchar* p) {
    ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(__builtin_bit_cast(half, u));
}
"""

_SH4 = """
constant uchar4 SH4A = {0,1,2,3};
constant uchar4 SH4B = {4,5,6,7};
"""

# signed_grid: apply pre-expanded sign mask to one 4-byte grid half.
# (gv ^ 0xFF) + 0xFF == -gv-1-1? -> char(gv^255)=~gv=-gv-1 ; plus char(255)=-1 => -gv. OK
_SGRID = r"""
static inline char4 sgrid4(threadgroup uchar* g, threadgroup uchar* m) {
    uchar4 gv = *(threadgroup uchar4*)g;
    uchar4 mk = *(threadgroup uchar4*)m;
    return as_type<char4>(gv ^ mk) - as_type<char4>(mk);
}
"""

# ---- scalar fallbacks (tails + k_base % 32 != 0); tables passed in -------
_SCALAR_HELPERS = r"""
static inline float dot_scalar_xs(const device uchar* row,
                                  const device float* xr, uint k0, uint kn,
                                  uint koff,
                                  threadgroup uchar* G, threadgroup uchar* M) {
    float acc = 0.f;
    for (uint k = k0; k < kn; k += 32u) {
        uint ka = koff + k;
        const device uchar* blk = row + (ka >> 8) * 74u;
        uint e = ka & 255u;
        float d = rd_f16(blk);
        uint j = e >> 3, t = e & 7u, sb = e >> 4;
        uint q = (uint)blk[2 + 2 * j] | ((uint)blk[3 + 2 * j] << 8);
        float db = d * (0.5f + (float)((blk[66 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu)) * 0.25f;
        float gv;
        if (t < 4u) gv = (float)sgrid4(G + (q & 511u) * 8u,     M + (q >> 9) * 8u)[t];
        else        gv = (float)sgrid4(G + (q & 511u) * 8u + 4u, M + (q >> 9) * 8u + 4u)[t - 4u];
        acc += xr[k] * db * gv;
    }
    return acc;
}
static inline float dot_scalar_xxs(const device uchar* row,
                                   const device float* xr, uint k0, uint kn,
                                   uint koff,
                                   threadgroup uchar* G, threadgroup uchar* M) {
    float acc = 0.f;
    for (uint k = k0; k < kn; k += 32u) {
        uint ka = koff + k;
        const device uchar* blk = row + (ka >> 8) * 66u;
        uint e = ka & 255u;
        float d = rd_f16(blk);
        uint g = e >> 5, l = (e >> 3) & 3u, t = e & 7u;
        const device uchar* Bp = blk + 6 + 8u * g;
        uint B = (uint)Bp[0] | ((uint)Bp[1] << 8) | ((uint)Bp[2] << 16) | ((uint)Bp[3] << 24);
        float db = d * (0.5f + (float)(B >> 28)) * 0.25f;
        uint qb = (uint)blk[2 + 8 * g + l];
        uint sidx = (B >> (7u * l)) & 0x7Fu;
        float gv;
        if (t < 4u) gv = (float)sgrid4(G + qb * 8u,      M + sidx * 8u)[t];
        else        gv = (float)sgrid4(G + qb * 8u + 4u, M + sidx * 8u + 4u)[t - 4u];
        acc += xr[k] * db * gv;
    }
    return acc;
}
static inline float dot_scalar_s(const device uchar* row,
                                 const device float* xr, uint k0, uint kn,
                                 uint koff,
                                 threadgroup uchar* G, threadgroup uchar* M) {
    float acc = 0.f;
    for (uint k = k0; k < kn; k += 32u) {
        uint ka = koff + k;
        const device uchar* blk = row + (ka >> 8) * 82u;
        uint e = ka & 255u;
        float d = rd_f16(blk);
        uint j = e >> 3, t = e & 7u, sb = e >> 4;
        uint hb = (blk[66 + (j >> 2)] >> ((j & 3u) * 2u)) & 3u;
        uint idx = (uint)blk[2 + j] | (hb << 8);
        uint bit = (blk[34 + j] >> t) & 1u;
        float db = d * (0.5f + (float)((blk[74 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu)) * 0.25f;
        float gvv;
        if (t < 4u) gvv = (float)(int)(as_type<char4>(*(threadgroup uchar4*)(G + idx * 8u)))[t];
        else        gvv = (float)(int)(as_type<char4>(*(threadgroup uchar4*)(G + idx * 8u + 4u)))[t - 4u];
        acc += xr[k] * db * gvv * (bit ? -1.0f : 1.0f);
    }
    return acc;
}
"""

# ---- vectorized 32-element chunk bodies -----------------------------------
# Available names: blk, e0, d, xrel (device float* at chunk start),
# acc, G (=tg_grid), M (=tg_msk), SH4A/SH4B.
_CHUNK_XXS = r"""
    {   // IQ2_XXS chunk: one 32-elem group g, one scale word B
        uint gg = e0 >> 5;
        const device uchar* Bp = blk + 6 + 8u * gg;
        uint B = (uint)Bp[0] | ((uint)Bp[1] << 8)
               | ((uint)Bp[2] << 16) | ((uint)Bp[3] << 24);
        float db = d * (0.5f + (float)(B >> 28)) * 0.25f;
        float s = 0.f;
        for (uint l = 0u; l < 4u; ++l) {
            uint sidx = (B >> (7u * l)) & 0x7Fu;
            uint qb   = (uint)blk[2 + 8u * gg + l];
            char4 ga = sgrid4(G + qb * 8u,      M + sidx * 8u);
            char4 gb = sgrid4(G + qb * 8u + 4u, M + sidx * 8u + 4u);
            const device float* xp = xrel + 8u * l;
            s += dot(*(device const float4*)xp,
                     (float4)ga)
               + dot(*(device const float4*)(xp + 4),
                     (float4)gb);
        }
        acc += db * s;
    }
"""

_CHUNK_XS = r"""
    {   // IQ2_XS chunk: 4 groups of 8, one uint16 code each; scales db0/db1
        uint j0 = e0 >> 3;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (uint l = 0u; l < 4u; ++l) {
            uint j = j0 + l;
            uint q = (uint)blk[2 + 2 * j] | ((uint)blk[3 + 2 * j] << 8);
            char4 ga = sgrid4(G + (q & 511u) * 8u,      M + (q >> 9) * 8u);
            char4 gb = sgrid4(G + (q & 511u) * 8u + 4u, M + (q >> 9) * 8u + 4u);
            const device float* xp = xrel + 8u * l;
            float sv = dot(*(device const float4*)xp,
                           (float4)ga)
                     + dot(*(device const float4*)(xp + 4),
                           (float4)gb);
            if (l == 0u) s0 = sv; else if (l == 1u) s1 = sv;
            else if (l == 2u) s2 = sv; else s3 = sv;
        }
        acc += db0 * s0 + db0 * s1 + db1 * s2 + db1 * s3;
    }
"""

_CHUNK_S = r"""
    {   // IQ2_S chunk: 4 groups of 8, explicit sign byte per group
        uint j0 = e0 >> 3;
        uchar hbb = blk[66 + (j0 >> 2)];
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (uint l = 0u; l < 4u; ++l) {
            uint j   = j0 + l;
            uint idx = (uint)blk[2 + j]
                     + (((uint)(hbb >> (2u * l)) & 3u) << 8);
            uchar sbits = blk[34 + j];
            uchar4 mlo = (uchar4)0u - (((uchar4)sbits >> SH4A) & (uchar4)1u);
            uchar4 mhi = (uchar4)0u - (((uchar4)sbits >> SH4B) & (uchar4)1u);
            char4 ga = as_type<char4>(*(threadgroup uchar4*)(G + idx * 8u)     ^ mlo)
                     - as_type<char4>(mlo);
            char4 gb = as_type<char4>(*(threadgroup uchar4*)(G + idx * 8u + 4u) ^ mhi)
                     - as_type<char4>(mhi);
            const device float* xp = xrel + 8u * l;
            float sv = dot(*(device const float4*)xp,
                           (float4)ga)
                     + dot(*(device const float4*)(xp + 4),
                           (float4)gb);
            if (l == 0u) s0 = sv; else if (l == 1u) s1 = sv;
            else if (l == 2u) s2 = sv; else s3 = sv;
        }
        acc += db0 * s0 + db0 * s1 + db1 * s2 + db1 * s3;
    }
"""

_BODY_TMPL = """
uint gid = thread_position_in_grid.x;
uint out_idx = gid >> 5u;
uint lane = gid & 31u;

// ---- threadgroup tables (declared here: program-scope TG not allowed) ----
threadgroup uchar tg_msk[1024];   // expanded sign masks (or zeros for IQ2_S)
threadgroup uchar tg_grid[{GSIZE}];
for (uint i = thread_position_in_threadgroup.x; i < 1024u; i += TG)
    tg_msk[i] = TBL[i];
for (uint i = thread_position_in_threadgroup.x; i < {GSIZE}u; i += TG)
    tg_grid[i] = TBL[1024u + i];
threadgroup_barrier(mem_flags::mem_threadgroup);

if (out_idx >= total) {{ return; }}
uint t_row = out_idx / NOUT;
uint n = n_base + out_idx % NOUT;
const device uchar* row = w + (size_t)n * ({ROWBYTES});
const device float* xrow = x + (size_t)t_row * KFULL + k_base;
threadgroup uchar* G = tg_grid;
threadgroup uchar* M = tg_msk;

float acc = 0.f;
if ((k_base & 31u) == 0u) {{
    // ---- vectorized path: lane handles 32 consecutive elements/chunk ----
    uint nfull = KLEN >> 5;
    for (uint c = lane; c < nfull; c += 32u) {{
        uint ka = k_base + (c << 5);           // absolute k of chunk start
        uint e0 = ka & 255u;
        const device uchar* blk = row + (ka >> 8) * {BLOCK};
        const device float* xrel = xrow + (ka - k_base);
        float d = rd_f16(blk);
{SCALEPRE}{CHUNK}
    }}
    // ---- tail: KLEN % 32 leftover elements ----
    uint tail0 = nfull << 5;
    uint tk = tail0 + lane;
    if (tk < KLEN) {{
        acc += {SCALAR}(row, xrow, tk, tk + 1u, k_base, G, M);
    }}
}} else {{
    acc = {SCALAR}(row, xrow, lane, KLEN, k_base, G, M);
}}
acc = simd_sum(acc);
if (lane == 0u) {{ out[out_idx] = acc; }}
"""

# per-chunk block-scale prelude (XS/S have two 16-elem sub-blocks per chunk)
_SCALE_XS = """\
        uchar scb = blk[66 + (e0 >> 5)];
        float db0 = d * (0.5f + (float)(scb & 15u)) * 0.25f;
        float db1 = d * (0.5f + (float)(scb >> 4)) * 0.25f;
"""
_SCALE_S = """\
        uchar scb = blk[74 + (e0 >> 5)];
        float db0 = d * (0.5f + (float)(scb & 15u)) * 0.25f;
        float db1 = d * (0.5f + (float)(scb >> 4)) * 0.25f;
"""

# per-format parameters
_FORMATS = {
    IQ2_XXS: dict(
        name="gguf_mm_o_iq2_xxs",
        block=66, gsize=2048, chunk=_CHUNK_XXS, scalar="dot_scalar_xxs",
        rowbytes="(KFULL >> 8) * 66", scalepre="",
        need_ksigns=True, need_grid="IQ2_XXS",
    ),
    IQ2_XS: dict(
        name="gguf_mm_o_iq2_xs",
        block=74, gsize=4096, chunk=_CHUNK_XS, scalar="dot_scalar_xs",
        rowbytes="(KFULL >> 8) * 74", scalepre=_SCALE_XS,
        need_ksigns=True, need_grid="IQ2_XS",
    ),
    IQ2_S: dict(
        name="gguf_mm_o_iq2_s",
        block=82, gsize=8192, chunk=_CHUNK_S, scalar="dot_scalar_s",
        rowbytes="(KFULL >> 8) * 82", scalepre=_SCALE_S,
        need_ksigns=False, need_grid="IQ2_S",
    ),
}


def _tables(dtype_code: int):
    import gguf.quants as Q
    f = _FORMATS[dtype_code]
    grid = getattr(Q, f["need_grid"])
    grid.init_grid()
    g = grid.grid.reshape(-1).astype(np.uint8)          # gsize bytes
    assert len(g) == f["gsize"]
    if f["need_ksigns"]:
        ks = np.frombuffer(Q.IQ2_XXS.ksigns, dtype=np.uint8)   # 128 bytes
        bits = ((ks[:, None] >> np.arange(8, dtype=np.uint8)[None, :])
                & np.uint8(1)).astype(np.uint8) * np.uint8(255)
        msk = bits.reshape(-1)
        assert len(msk) == 1024
        return np.concatenate([msk, g]).tobytes()
    # IQ2_S: no ksigns table -> zero-pad mask section (scalar path reads
    # sign bytes straight from the data)
    return b"\x00" * 1024 + g.tobytes()


_kernel_cache: dict[int, object] = {}


def _build_kernel(dtype_code: int, tg: int = 1024):
    key = (dtype_code, tg)
    if key in _kernel_cache:
        return _kernel_cache[key]
    f = _FORMATS[dtype_code]
    tbl = _tables(dtype_code)
    header = (
        "constant uint DT = %d;\n" % dtype_code
        + ("constant uchar TBL[%d] = {%s};\n"
           % (len(tbl), ",".join("0x%02x" % b for b in tbl)))
        + "constant uint TG = %d;\n" % tg
        + _SH4
        + _RD_F16
        + _SGRID
        + _SCALAR_HELPERS
    )
    body = _BODY_TMPL.format(
        GSIZE=f["gsize"],
        ROWBYTES=f["rowbytes"],
        CHUNK=f["chunk"],
        SCALEPRE=f["scalepre"],
        SCALAR=f["scalar"],
        BLOCK=f["block"],
    )
    kern = mx.fast.metal_kernel(
        name=f["name"],
        input_names=["x", "w", "total", "NOUT", "KFULL", "KLEN",
                     "n_base", "k_base"],
        output_names=["out"],
        header=header,
        source=body,
        ensure_row_contiguous=False,
    )
    _kernel_cache[key] = kern
    return kern


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

    Semantics match atf.gguf_metal.gguf_matmul for IQ2-family formats."""
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


SUPPORTED = {IQ2_XXS, IQ2_XS, IQ2_S}

_scal_cache: dict[tuple, tuple] = {}

_scal_cache: dict[tuple, tuple] = {}
