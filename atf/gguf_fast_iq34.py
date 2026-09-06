"""llama.cpp-style vectorized Metal inner loops for GGUF IQ3_S / IQ4_XS.

Key changes vs atf/gguf_metal.py scalar loops:
  * Each lane decodes a whole 8-element slice of every 256-element block
    (vectorized uint32/ushort/uchar loads) instead of striding K by 32 with
    one scalar element per iteration.
  * Block scales + f16 d are loaded once per block per lane (not per element).
  * Codebook tables live in threadgroup memory (copied once from constant),
    so random per-lane lookups hit TGEM instead of replaying constant loads.
  * x rows are consumed as float4 vector loads.

matmul() matches atf.gguf_metal.gguf_matmul semantics:
    y[T,n_out] = x[:, k_base:k_base+k_len] @ dequant(W)[n_base:n_base+n_out]
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

IQ3_S = 21
IQ4_XS = 23

_KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10,
                  1, 13, 25, 38, 53, 69, 89, 113]


def _table_bytes(dtype_code: int) -> bytes | None:
    import gguf.quants as Q
    if dtype_code == IQ3_S:
        Q.IQ3_S.init_grid()
        return Q.IQ3_S.grid.reshape(-1).astype(np.uint8).tobytes()
    return None


_ROWBYTES = {IQ3_S: 110, IQ4_XS: 136}

# --- shared header: f16 reader + scalar fallback dq funcs (edges only) -----
_COMMON = r"""
static inline float rd_f16(const device uchar* p) {
    ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(__builtin_bit_cast(half, u));
}
"""

_IQ3_S_SCALAR = r"""
constant int8_t KV4[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
// edge-only scalar path (same math as production dq_iq3_s)
static inline float dq_scalar(const device uchar* row, uint k) {
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
"""

_IQ4_XS_SCALAR = r"""
// edge-only scalar path (same math as production dq_iq4_xs)
static inline float dq_scalar(const device uchar* row, uint k) {
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
    float kv = (float)(int8_t)TBL[nib];
    return d * (float)sc * kv;
}
"""

# Vectorized body. Per 256-elem block, lane l owns elements [l*8, l*8+8):
_BODY_TMPL = """
uint gid = thread_position_in_grid.x;
uint out_idx = gid >> 5;         // one SIMD group (32 threads) per output elt
uint lane = gid & 31u;

// ---- cooperative init of the threadgroup codebook table ----
threadgroup uchar tg_tbl[TBLN];
for (uint i = thread_position_in_threadgroup.x; i < TBLN; i += TGSZ) {
    tg_tbl[i] = TBL[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (out_idx >= total) { return; }
uint t = out_idx / NOUT;
uint n = n_base + out_idx % NOUT;
const device uchar* row = w + (size_t)n * ROWBYTES;
const device float* xp = x + (size_t)t * KFULL;

uint k0 = k_base;
uint k1 = k_base + KLEN;
uint b0 = (k0 + 255u) >> 8;      // first full block
uint b1 = k1 >> 8;               // one past last full block

float acc = 0.f;
float acc2 = 0.f;
float acc3 = 0.f;
float acc4 = 0.f;
{FAST_LOOP}
acc = (acc + acc2) + (acc3 + acc4);
// ---- edges (partial blocks at range ends): scalar fallback ----
uint e_end = min(k1, b0 << 8);
for (uint k = k0 + lane; k < e_end; k += 32u) {
    acc += xp[k] * dq_scalar(row, k);
}
for (uint k = max(k0, b1 << 8) + lane; k < k1; k += 32u) {
    acc += xp[k] * dq_scalar(row, k);
}

acc = simd_sum(acc);
if (lane == 0u) {
    out[out_idx] = acc;
}
"""

_IQ3_S_FAST = r"""
const device uchar* blk = row + b * 110u;
float d = rd_f16(blk);
uint sb = lane >> 2;
uint scl = (blk[106 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu;
float db = d * (1.0f + 2.0f * (float)scl);
ushort q2 = *(device const ushort*)(blk + 2 + lane * 2);
uint qh = blk[66 + sb];
uint sg = blk[74 + lane];
// qh bit positions for j = 2*lane .. 2*lane+1 (wrap-safe rotate)
uint p0 = (lane * 2u) & 7u;
// rotate right by p0: hrot bit g = qh bit (p0 + g) mod 8
uint hrot = ((qh >> p0) | (qh << ((8u - p0) & 7u))) & 0xFFu;
uint hb0 = hrot & 1u;
uint hb1 = (hrot >> 1) & 1u;
device const float4* xv = (device const float4*)(xp + b * 256u + lane * 8u);
float4 xa = xv[0];
float4 xb = xv[1];

uint idx0 = (q2 & 0xFFu) | (hb0 << 8);
const threadgroup uchar* gp0 = tg_tbl + idx0 * 4u;
float g0 = (float)(int8_t)gp0[0];
float g1 = (float)(int8_t)gp0[1];
float g2 = (float)(int8_t)gp0[2];
float g3 = (float)(int8_t)gp0[3];
g0 = (sg & 0x01u) ? -g0 : g0;
g1 = (sg & 0x02u) ? -g1 : g1;
g2 = (sg & 0x04u) ? -g2 : g2;
g3 = (sg & 0x08u) ? -g3 : g3;

uint idx1 = (q2 >> 8) | (hb1 << 8);
const threadgroup uchar* gp1 = tg_tbl + idx1 * 4u;
float h0 = (float)(int8_t)gp1[0];
float h1 = (float)(int8_t)gp1[1];
float h2 = (float)(int8_t)gp1[2];
float h3 = (float)(int8_t)gp1[3];
h0 = (sg & 0x10u) ? -h0 : h0;
h1 = (sg & 0x20u) ? -h1 : h1;
h2 = (sg & 0x40u) ? -h2 : h2;
h3 = (sg & 0x80u) ? -h3 : h3;

acc  = fma(xa.x, db * g0, acc);
acc2 = fma(xa.y, db * g1, acc2);
acc3 = fma(xa.z, db * g2, acc3);
acc4 = fma(xa.w, db * g3, acc4);
acc  = fma(xb.x, db * h0, acc);
acc2 = fma(xb.y, db * h1, acc2);
acc3 = fma(xb.z, db * h2, acc3);
acc4 = fma(xb.w, db * h3, acc4);
"""

_IQ4_XS_FAST = r"""
const device uchar* blk = row + b * 136u;
float d = rd_f16(blk);
uint sb = lane >> 2;
uint scl = (blk[4 + (sb >> 1)] >> ((sb & 1u) * 4u)) & 0xFu;
uint sch = (((uint)blk[3] << 8 | (uint)blk[2]) >> (2u * sb)) & 3u;
float db = d * (float)((int)(scl | (sch << 4)) - 32);
// qs byte j of the sub-block holds element j (low nibble) and element
// j+16 (high nibble); lane group cpos covers bytes base..base+7
uint cpos = lane & 3u;
uint base = (cpos & 1u) * 8u;
uint sh = (cpos >> 1) << 2;
const device uchar* qp = blk + 8u + sb * 16u + base;
uint q0 = *(device const uint*)(qp) >> sh;
uint q1 = *(device const uint*)(qp + 4u) >> sh;
device const float4* xv = (device const float4*)(xp + b * 256u + lane * 8u);
float4 xa = xv[0];
float4 xb = xv[1];
float g0 = (float)(int8_t)tg_tbl[q0 & 15u];
float g1 = (float)(int8_t)tg_tbl[(q0 >> 8) & 15u];
float g2 = (float)(int8_t)tg_tbl[(q0 >> 16) & 15u];
float g3 = (float)(int8_t)tg_tbl[(q0 >> 24) & 15u];
float g4 = (float)(int8_t)tg_tbl[q1 & 15u];
float g5 = (float)(int8_t)tg_tbl[(q1 >> 8) & 15u];
float g6 = (float)(int8_t)tg_tbl[(q1 >> 16) & 15u];
float g7 = (float)(int8_t)tg_tbl[(q1 >> 24) & 15u];
acc  = fma(xa.x, db * g0, acc);
acc2 = fma(xa.y, db * g1, acc2);
acc3 = fma(xa.z, db * g2, acc3);
acc4 = fma(xa.w, db * g3, acc4);
acc  = fma(xb.x, db * g4, acc);
acc2 = fma(xb.y, db * g5, acc2);
acc3 = fma(xb.z, db * g6, acc3);
acc4 = fma(xb.w, db * g7, acc4);
"""
_kernels: dict[int, object] = {}
_scalar_args_cache: dict[tuple, tuple] = {}


def _scalar_args(T, n_out, k_full, k_len, n_base, k_base):
    key = (T, n_out, k_full, k_len, n_base, k_base)
    ent = _scalar_args_cache.get(key)
    if ent is None:
        ent = (mx.array(T * n_out), mx.array(n_out), mx.array(k_full),
               mx.array(k_len), mx.array(k_base), mx.array(n_base))
        _scalar_args_cache[key] = ent
    return ent


def _build_kernel(dtype_code: int, tg: int = 1024):
    ck = (dtype_code, tg)
    if ck in _kernels:
        return _kernels[ck]
    if dtype_code not in _ROWBYTES:
        raise ValueError(f"unsupported dtype_code {dtype_code}")

    tbl_bytes = _table_bytes(dtype_code)
    if tbl_bytes is None:
        tbl_bytes = bytes(((v + 256) % 256) for v in _KVALUES_IQ4NL)

    scalar_src = _IQ3_S_SCALAR if dtype_code == IQ3_S else _IQ4_XS_SCALAR
    # TBL doubles as init source for the threadgroup copy and as backing for
    # the scalar edge path.
    header = (
        "constant uint DT = %d;\n" % dtype_code
        + ("constant uchar TBL[%d] = {%s};\n"
           % (len(tbl_bytes), ",".join("0x%02x" % b for b in tbl_bytes)))
        + "constant uint TBLN = %d;\n" % len(tbl_bytes)
        + _COMMON
        + scalar_src
    )
    fast = _IQ3_S_FAST if dtype_code == IQ3_S else _IQ4_XS_FAST
    loop = "for (uint b = b0; b < b1; ++b) {\n" + fast + "\n}"
    body = (_BODY_TMPL.replace("{FAST_LOOP}", loop)
            .replace("ROWBYTES", "(KFULL >> 8) * %du" % _ROWBYTES[dtype_code])
            .replace("TGSZ", "%du" % tg))

    kern = mx.fast.metal_kernel(
        name=f"gguf_mm_v_{dtype_code}_tg{tg}",
        input_names=["x", "w", "total", "NOUT", "KFULL", "KLEN",
                     "n_base", "k_base"],
        output_names=["out"],
        header=header,
        source=body,
        ensure_row_contiguous=False,
    )
    _kernels[ck] = kern
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
    tg: int = 1024,
) -> mx.array:
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    if k_len is None:
        k_len = k_full - k_base
    if dtype_code not in _ROWBYTES:
        raise ValueError(f"unsupported dtype_code {dtype_code}")
    T = x.shape[0]
    kern = _build_kernel(dtype_code, tg)
    total_a, nout_a, kfull_a, klen_a, kbase_a, nbase_a = _scalar_args(
        T, n_out, k_full, k_len, n_base, k_base)
    # Pad the grid to whole 1024-thread threadgroups: MLX dispatches by
    # threads, and a partial last threadgroup would leave part of tg_tbl
    # uninitialized (init loop strides by 1024).
    g0 = T * n_out * 32
    grid0 = ((g0 + tg - 1) // tg) * tg
    out = kern(
        inputs=[x.astype(mx.float32), w_bytes,
                total_a, nout_a, kfull_a, klen_a, nbase_a, kbase_a],
        grid=(grid0, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(T * n_out,)],
        output_dtypes=[mx.float32],
        verbose=False,
    )
    out = out[0].reshape(T, n_out)
    return out[0] if squeeze else out
