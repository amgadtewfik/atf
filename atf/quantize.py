"""Quantization utilities for ATF expert weights.

Each function pair (quantize_X / dequantize_X) is self-contained: the
quantized bytes carry their own small header (shape/block-size/padding) so a
blob can be dequantized without external context beyond knowing which LOD it
came from.
"""
from __future__ import annotations

import struct

import numpy as np

from .format import LodLevel


# ---------------------------------------------------------------- LOD 0: INT8
def quantize_int8(w: np.ndarray) -> bytes:
    """Per-row (per-output-channel) INT8 quant."""
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    scales = np.max(np.abs(w), axis=1, keepdims=True) / 127.0
    scales = np.where(scales == 0, 1e-8, scales)
    q = np.clip(np.round(w / scales), -127, 127).astype(np.int8)
    header = struct.pack("<II", rows, cols)
    return header + scales.astype(np.float32).tobytes() + q.tobytes()


def dequantize_int8(data: bytes) -> np.ndarray:
    rows, cols = struct.unpack_from("<II", data, 0)
    off = 8
    scales = np.frombuffer(data, dtype=np.float32, count=rows, offset=off).reshape(rows, 1)
    off += rows * 4
    q = np.frombuffer(data, dtype=np.int8, count=rows * cols, offset=off).reshape(rows, cols)
    return q.astype(np.float32) * scales


# ------------------------------------------------------- LOD 1: INT4 (block 32)
def quantize_int4(w: np.ndarray, block: int = 32) -> bytes:
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    pad = (-cols) % block
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    cols_p = w.shape[1]
    nblk = cols_p // block
    wb = w.reshape(rows, nblk, block)
    scales = np.max(np.abs(wb), axis=2, keepdims=True) / 7.0
    scales = np.where(scales == 0, 1e-8, scales)
    q = (np.clip(np.round(wb / scales), -7, 7).astype(np.int16) + 8).astype(np.uint8)  # 0..15
    lo = q[..., 0::2] & 0x0F
    hi = (q[..., 1::2] & 0x0F) << 4
    packed = (lo | hi).astype(np.uint8)
    header = struct.pack("<IIII", rows, cols, block, pad)
    return header + scales.astype(np.float32).tobytes() + packed.tobytes()


def dequantize_int4(data: bytes) -> np.ndarray:
    rows, cols, block, pad = struct.unpack_from("<IIII", data, 0)
    off = 16
    cols_p = cols + pad
    nblk = cols_p // block
    scales = np.frombuffer(data, dtype=np.float32, count=rows * nblk, offset=off).reshape(rows, nblk, 1)
    off += rows * nblk * 4
    packed = np.frombuffer(data, dtype=np.uint8, count=rows * nblk * (block // 2), offset=off).reshape(
        rows, nblk, block // 2
    )
    lo = (packed & 0x0F).astype(np.int16) - 8
    hi = ((packed >> 4) & 0x0F).astype(np.int16) - 8
    q = np.empty((rows, nblk, block), dtype=np.int16)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    w = (q.astype(np.float32) * scales).reshape(rows, cols_p)
    return w[:, :cols]


# ------------------------------------------------------- LOD 2: INT2 (block 64)
_LEVELS = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float32)


def quantize_int2(w: np.ndarray, block: int = 64) -> bytes:
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    pad = (-cols) % block
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    cols_p = w.shape[1]
    nblk = cols_p // block
    wb = w.reshape(rows, nblk, block)
    scales = np.max(np.abs(wb), axis=2, keepdims=True)
    scales = np.where(scales == 0, 1e-8, scales)
    wn = wb / scales
    idx = np.argmin(np.abs(wn[..., None] - _LEVELS), axis=-1).astype(np.uint8)  # 0..3, shape (rows,nblk,block)
    packed = np.zeros((rows, nblk, block // 4), dtype=np.uint8)
    for k in range(4):
        packed |= (idx[..., k::4] & 0x03) << (2 * k)
    header = struct.pack("<IIII", rows, cols, block, pad)
    return header + scales.astype(np.float32).tobytes() + packed.tobytes()


def dequantize_int2(data: bytes) -> np.ndarray:
    rows, cols, block, pad = struct.unpack_from("<IIII", data, 0)
    off = 16
    cols_p = cols + pad
    nblk = cols_p // block
    scales = np.frombuffer(data, dtype=np.float32, count=rows * nblk, offset=off).reshape(rows, nblk, 1)
    off += rows * nblk * 4
    packed = np.frombuffer(data, dtype=np.uint8, count=rows * nblk * (block // 4), offset=off).reshape(
        rows, nblk, block // 4
    )
    idx = np.empty((rows, nblk, block), dtype=np.uint8)
    for k in range(4):
        idx[..., k::4] = (packed >> (2 * k)) & 0x03
    wn = _LEVELS[idx]
    w = (wn * scales).reshape(rows, cols_p)
    return w[:, :cols]


# --------------------------------------------------------------- LOD 3: low-rank
def quantize_lowrank(w: np.ndarray, rank: int) -> bytes:
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    rank = max(1, min(rank, min(rows, cols)))
    U, S, Vt = np.linalg.svd(w, full_matrices=False)
    U = (U[:, :rank] * S[:rank]).astype(np.float32)
    Vt = Vt[:rank, :].astype(np.float32)
    header = struct.pack("<III", rows, cols, rank)
    return header + U.tobytes() + Vt.tobytes()


def dequantize_lowrank(data: bytes) -> np.ndarray:
    rows, cols, rank = struct.unpack_from("<III", data, 0)
    off = 12
    U = np.frombuffer(data, dtype=np.float32, count=rows * rank, offset=off).reshape(rows, rank)
    off += rows * rank * 4
    Vt = np.frombuffer(data, dtype=np.float32, count=rank * cols, offset=off).reshape(rank, cols)
    return U @ Vt


# ---------------------------------------------------------------- LOD dispatch
def quantize_for_lod(w: np.ndarray, lod: LodLevel, lowrank_div: int = 8) -> bytes:
    if lod == LodLevel.LOD_0:
        return quantize_int8(w)
    if lod == LodLevel.LOD_1:
        return quantize_int4(w, block=32)
    if lod == LodLevel.LOD_2:
        return quantize_int2(w, block=64)
    if lod == LodLevel.LOD_3:
        rank = max(1, w.shape[1] // lowrank_div)
        return quantize_lowrank(w, rank)
    raise ValueError(f"No dense quantizer for {lod} (LOD_4 is index-only, no payload)")


def dequantize_for_lod(data: bytes, lod: LodLevel) -> np.ndarray:
    if lod == LodLevel.LOD_0:
        return dequantize_int8(data)
    if lod == LodLevel.LOD_1:
        return dequantize_int4(data)
    if lod == LodLevel.LOD_2:
        return dequantize_int2(data)
    if lod == LodLevel.LOD_3:
        return dequantize_lowrank(data)
    raise ValueError(f"No dense dequantizer for {lod}")


# ------------------------------------------------------- FP4 (E2M1, group 64)
_FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def quantize_fp4(w: np.ndarray, group: int = 64) -> bytes:
    """MXFP4-style quant: E2M1 nibbles + one f32 scale per `group` elements.

    Blob layout: <rows u32><cols u32><scales f32 [rows*cols/group]><nibbles u8>.
    Each byte holds two consecutive elements (low nibble first).
    """
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    assert cols % group == 0, f"cols {cols} not divisible by group {group}"
    g = w.reshape(rows, cols // group, group)
    scales = np.max(np.abs(g), axis=2) / 6.0            # E2M1 max is 6.0
    scales = np.where(scales == 0, 1e-8, scales).astype(np.float32)
    norm = g / scales[:, :, None]
    absn = np.abs(norm)
    # nearest E2M1 code: LUT midpoints between codes i and i+1
    mids = (_FP4_LUT[:-1] + _FP4_LUT[1:]) / 2           # [7]
    idx = np.sum(absn[..., None] > mids, axis=-1)       # [rows, g, group] in 0..7
    sign = (np.signbit(norm).astype(np.uint8) << 3)
    codes = (idx.astype(np.uint8) | sign).reshape(rows, cols)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    if hi.shape[1] < lo.shape[1]:                       # odd col count pad
        hi = np.pad(hi, ((0, 0), (0, 1)))
    packed = (lo | (hi << 4)).astype(np.uint8)
    header = struct.pack("<II", rows, cols)
    return header + scales.tobytes() + packed.tobytes()


def dequantize_fp4(data: bytes) -> np.ndarray:
    rows, cols = struct.unpack_from("<II", data, 0)
    off = 8
    n_groups = rows * (cols // 64)
    scales = np.frombuffer(data, dtype=np.float32, count=n_groups,
                           offset=off).reshape(rows, cols // 64)
    off += n_groups * 4
    packed = np.frombuffer(data, dtype=np.uint8, count=rows * ((cols + 1) // 2),
                           offset=off).reshape(rows, (cols + 1) // 2)
    codes = np.empty((rows, cols), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    vals = _FP4_LUT[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    return (vals.reshape(rows, cols // 64, 64) * scales[:, :, None]).reshape(rows, cols)
