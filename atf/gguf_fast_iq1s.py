"""Optimized GGUF dequant-matmul Metal kernel for IQ1_S.

matmul(x, w_bytes, n_out, k_full, dtype_code, n_base=0, k_base=0, k_len=None)
mirrors atf.gguf_metal.gguf_matmul semantics for IQ1_S (=19).

vs the production generic kernel (one ELEMENT per lane, scalar loads):
  * each lane decodes one contiguous 32-element SUB-BLOCK per iteration;
    d, qh word (qu), dl and delta are fetched ONCE per chunk;
  * grid rows are read as uchar4 VECTOR loads straight from the CONSTANT
    address-space TBL (exp2 ablation: beats a threadgroup-resident copy --
    the table is shared by every output element so it stays hot in cache,
    and dropping the per-group copy/barrier raises occupancy);
  * x consumed via dot(float4, float4); the +delta term folds into
    dl*delta*sum(x_chunk) -- no per-element add.
IQ1_S block layout (50 bytes / 256 elems):
  [0:2] d f16 | [2:34] qs[32] low grid byte per 8-elem group |
  [34:50] qh[16] -> uint16 qu per 32-elem sub-block:
    bits 0-8: three 3-bit high-grid-bits (one per group l=0..3)
    bits 12-14: scale code (dl = d*(2*code+1)), bit 15: delta sign (-0.125).
Any k_base % 32 != 0 falls back to a scalar path inside the kernel.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

IQ1_S = 19

_RD_F16 = """
static inline float rd_f16(const device uchar* p) {
    ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(__builtin_bit_cast(half, u));
}
"""

# scalar fallback (tails + k_base % 32 != 0); table passed in.
_SCALAR = r"""
static inline float dot_scalar_iq1s(const device uchar* row,
                                    const device float* xr, uint k0, uint kn,
                                    uint koff) {
    float acc = 0.f;
    for (uint k = k0; k < kn; k += 32u) {
        uint ka = koff + k;
        const device uchar* blk = row + (ka >> 8) * 50u;
        uint e = ka & 255u;
        float d = rd_f16(blk);
        uint sb = e >> 5;
        uint qu = (uint)blk[34u + 2u*sb] | ((uint)blk[35u + 2u*sb] << 8);
        float dl = d * (float)(2u * ((qu >> 12) & 7u) + 1u);
        float delta = (qu & 0x8000u) ? -0.125f : 0.125f;
        uint l = (e >> 3) & 3u;
        uint idx = (uint)blk[2u + (e >> 3)] + (((qu >> (3u*l)) & 7u) << 8);
        float gv;
        if ((e & 7u) < 4u)
            gv = (float)as_type<char4>(*(constant uchar4*)(TBL + idx*8u))[e & 7u];
        else
            gv = (float)as_type<char4>(*(constant uchar4*)(TBL + idx*8u + 4u))[(e & 7u) - 4u];
        acc += xr[k] * dl * (gv + delta);
    }
    return acc;
}
"""

_CHUNK = r"""
    {   // IQ1_S chunk: one 32-elem sub-block, 4 grid rows of 8
        uint sb = e0 >> 5;
        const device uchar* qp = blk + 34u + 2u*sb;
        uint qu = (uint)qp[0] | ((uint)qp[1] << 8);
        float dl = d * (float)(2u * ((qu >> 12) & 7u) + 1u);
        float delta = (qu & 0x8000u) ? -0.125f : 0.125f;
        const device uchar* qs4 = blk + 2u + 4u*sb;
        float4 sx4 = (float4)0.f;
        float s = 0.f;
        for (uint l = 0u; l < 4u; ++l) {
            uint idx = ((uint)qs4[l]) + (((qu >> (3u*l)) & 7u) << 8);
            char4 ga = as_type<char4>(*(constant uchar4*)(TBL + idx*8u));
            char4 gb = as_type<char4>(*(constant uchar4*)(TBL + idx*8u + 4u));
            float4 xv0 = *(device const float4*)xrel;
            float4 xv1 = *(device const float4*)(xrel + 4);
            s  += dot(xv0, (float4)ga) + dot(xv1, (float4)gb);
            sx4 += xv0 + xv1;
            xrel += 8u;
        }
        acc += dl * s + dl * delta * (sx4.x + sx4.y + sx4.z + sx4.w);
    }
"""

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
    // ---- vectorized path: lane handles 32 consecutive elements/chunk ----
    uint nfull = KLEN >> 5;
    for (uint c = lane; c < nfull; c += 32u) {{
        uint ka = k_base + (c << 5);           // absolute k of chunk start
        uint e0 = ka & 255u;
        const device uchar* blk = row + (ka >> 8) * 50u;
        const device float* xrel = xrow + (ka - k_base);
        float d = rd_f16(blk);
{CHUNK}
    }}
    // ---- tail: KLEN % 32 leftover elements ----
    uint tail0 = nfull << 5;
    uint tk = tail0 + lane;
    if (tk < KLEN) {{
        acc += dot_scalar_iq1s(row, xrow, tk, tk + 1u, k_base);
    }}
}} else {{
    acc = dot_scalar_iq1s(row, xrow, lane, KLEN, k_base);
}}
acc = simd_sum(acc);
if (lane == 0u) {{ out[out_idx] = acc; }}
"""


def _tables() -> bytes:
    import gguf.quants as Q
    Q.IQ1_S.init_grid()
    g = Q.IQ1_S.grid.reshape(-1).astype(np.uint8)
    assert g.size == 16384
    return g.tobytes()


_kernel_cache: dict[int, object] = {}


def _build_kernel(dtype_code: int = IQ1_S, tg: int = 1024):
    if dtype_code in _kernel_cache:
        return _kernel_cache[dtype_code]
    tbl = _tables()
    header = (
        "constant uint DT = %d;\n" % dtype_code
        + ("constant uchar TBL[%d] = {%s};\n"
           % (len(tbl), ",".join("0x%02x" % b for b in tbl)))
        + _RD_F16
        + _SCALAR
    )
    body = _BODY_TMPL.format(ROWBYTES="(KFULL >> 8) * 50", CHUNK=_CHUNK)
    kern = mx.fast.metal_kernel(
        name="gguf_mm_o_iq1_s",
        input_names=["x", "w", "total", "NOUT", "KFULL", "KLEN",
                     "n_base", "k_base"],
        output_names=["out"],
        header=header,
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

    Semantics match atf.gguf_metal.gguf_matmul for IQ1_S."""
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


SUPPORTED = {IQ1_S}
