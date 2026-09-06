"""Dequantize GGUF tensor data (raw bytes) to numpy float32 arrays.

Implements dequantizers for the specific types present in Qwen3.5-9B-IQ4_NL:
  F32 (0), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14), IQ4_NL (20)
"""
from __future__ import annotations

import struct
import numpy as np

# GGML quant type sizes (bytes per block of 256/32 elements)
Q8_0_K = 32       # elements per block
Q8_0_SIZE = 34    # 1 scale (f16) + 32 int8

Q4_K_K = 256      # elements per super-block
Q4_K_SIZE = 144   # 2 + 2 + 12 + 128

Q5_K_K = 256
Q5_K_SIZE = 160   # 2 + 2 + 12 + 128 + 16

Q6_K_K = 256
Q6_K_SIZE = 210   # 2 + 128 + 64 + 16

IQ4_NL_K = 32     # ggml block_iq4_nl: 32 elements per block (NOT a K-quant super-block)
IQ4_NL_SIZE = 18  # ggml_fp16_t d (2 bytes) + qs[QK4_NL/2] (16 bytes) -- no dmin, no sub-scales


def _f16_to_f32(data: bytes, count: int, offset: int = 0) -> np.ndarray:
    """Convert raw half-precision bytes to float32 array."""
    raw = np.frombuffer(data, dtype=np.float16, count=count, offset=offset)
    return raw.astype(np.float32)


def dequant_f32(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    return np.frombuffer(data, dtype=np.float32, count=n).reshape(shape).copy()


def dequant_q8_0(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Q8_0: per-block(32) scale + int8 values. Block = 34 bytes."""
    n = int(np.prod(shape))
    n_blocks = n // Q8_0_K
    assert n_blocks * Q8_0_K == n, f"Q8_0: {n} not divisible by {Q8_0_K}"

    out = np.empty(n, dtype=np.float32)
    for b in range(n_blocks):
        base = b * Q8_0_SIZE
        scale = struct.unpack_from("<e", data, base)[0]  # half float
        vals = np.frombuffer(data, dtype=np.int8, count=Q8_0_K, offset=base + 2)
        out[b * Q8_0_K:(b + 1) * Q8_0_K] = vals.astype(np.float32) * scale
    return out.reshape(shape)


def _read_qk_scales(data: bytes, offset: int) -> np.ndarray:
    """Read 16 packed 4-bit scale values from 12 bytes in K-quant super-block.
    Returns array of 16 float32 values (each 0..15 scaled)."""
    raw = np.frombuffer(data, dtype=np.uint8, count=12, offset=offset)
    # 16 values packed in 12 bytes: 2 per byte for first 8 bytes (16 values)
    # Actually: 12 bytes = 24 nibbles, but we only use 16.
    # Layout: bytes 0-7 have high/low nibbles for scales 0-15
    scales = np.empty(16, dtype=np.float32)
    for i in range(16):
        byte_idx = i // 2
        if i % 2 == 0:
            scales[i] = float(raw[byte_idx] & 0x0F)
        else:
            scales[i] = float(raw[byte_idx] >> 4)
    return scales


def dequant_q4_K(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Q4_K: 256 elements per super-block, 144 bytes each.
    Layout per super-block:
      [0:2]   d (f16) - super-block scale
      [2:4]   dmin (f16) - super-block min scale
      [4:16]  scales (12 bytes) - 16 sub-block scales (4-bit each, 2 per byte)
      [16:144] qs (128 bytes) - 256 4-bit quantized values (2 per byte)
    """
    n = int(np.prod(shape))
    n_blocks = n // Q4_K_K
    assert n_blocks * Q4_K_K == n, f"Q4_K: {n} not divisible by {Q4_K_K}"

    out = np.empty(n, dtype=np.float32)

    # Pre-compute 4-bit lookup: values 0-15 mapped to [-8..7]
    q4_lut = np.arange(16, dtype=np.float32) - 8.0

    for b in range(n_blocks):
        base = b * Q4_K_SIZE
        d = struct.unpack_from("<e", data, base)[0]
        dmin = struct.unpack_from("<e", data, base + 2)[0]

        # 16 sub-block scales (4-bit each)
        scales_raw = np.frombuffer(data, dtype=np.uint8, count=12, offset=base + 4)
        scales = np.empty(16, dtype=np.float32)
        for i in range(16):
            if i < 8:
                scales[i] = float(scales_raw[i] & 0x0F)
            else:
                scales[i] = float(scales_raw[i - 8] >> 4)
        # Scale the sub-block scales
        scales = (scales * d - dmin * 8.0)

        # 256 4-bit values
        qs = np.frombuffer(data, dtype=np.uint8, count=128, offset=base + 16)
        lo = (qs & 0x0F).astype(np.int32)
        hi = (qs >> 4).astype(np.int32)
        q = np.empty(256, dtype=np.float32)
        q[0::2] = q4_lut[lo]
        q[1::2] = q4_lut[hi]

        # Apply sub-block scales: 16 elements per sub-block
        for s in range(16):
            out[b * 256 + s * 16:(b * 256 + (s + 1) * 16)] = q[s * 16:(s + 1) * 16] * scales[s]

    return out.reshape(shape)


def dequant_q5_K(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Q5_K: 256 elements per super-block, 160 bytes each.
    Layout per super-block:
      [0:2]   d (f16)
      [2:4]   dmin (f16)
      [4:16]  scales (12 bytes) - 16 sub-block scales
      [16:144] qs (128 bytes) - 256 4-bit base values
      [144:160]qh (16 bytes) - 256 1-bit high bits (for 5th bit)
    """
    n = int(np.prod(shape))
    n_blocks = n // Q5_K_K
    assert n_blocks * Q5_K_K == n, f"Q5_K: {n} not divisible by {Q5_K_K}"

    out = np.empty(n, dtype=np.float32)
    q4_lut = np.arange(16, dtype=np.float32) - 8.0

    for b in range(n_blocks):
        base = b * Q5_K_SIZE
        d = struct.unpack_from("<e", data, base)[0]
        dmin = struct.unpack_from("<e", data, base + 2)[0]

        scales_raw = np.frombuffer(data, dtype=np.uint8, count=12, offset=base + 4)
        scales = np.empty(16, dtype=np.float32)
        for i in range(16):
            if i < 8:
                scales[i] = float(scales_raw[i] & 0x0F)
            else:
                scales[i] = float(scales_raw[i - 8] >> 4)
        scales = (scales * d - dmin * 8.0)

        qs = np.frombuffer(data, dtype=np.uint8, count=128, offset=base + 16)
        qh = np.frombuffer(data, dtype=np.uint8, count=16, offset=base + 144)

        lo = (qs & 0x0F).astype(np.int32)
        hi = (qs >> 4).astype(np.int32)
        q = np.empty(256, dtype=np.float32)
        q[0::2] = q4_lut[lo]
        q[1::2] = q4_lut[hi]

        # Add 5th bit from qh
        for i in range(256):
            byte_idx = i // 8
            bit_idx = i % 8
            high_bit = (qh[byte_idx] >> bit_idx) & 1
            q[i] += high_bit * 16.0

        for s in range(16):
            out[b * 256 + s * 16:(b * 256 + (s + 1) * 16)] = q[s * 16:(s + 1) * 16] * scales[s]

    return out.reshape(shape)


def dequant_q6_K(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Q6_K: 256 elements per super-block, 210 bytes each.
    Layout:
      [0:2]    d (f16)
      [2:130]  ql (128 bytes) - low 4 bits for 256 values
      [130:194]qh (64 bytes) - high 2 bits for 256 values (4 per byte)
      [194:210]scales (16 bytes) - 16 scale values
    """
    n = int(np.prod(shape))
    n_blocks = n // Q6_K_K
    assert n_blocks * Q6_K_K == n, f"Q6_K: {n} not divisible by {Q6_K_K}"

    out = np.empty(n, dtype=np.float32)

    for b in range(n_blocks):
        base = b * Q6_K_SIZE
        d = struct.unpack_from("<e", data, base)[0]

        # Low 4 bits (2 per byte)
        ql = np.frombuffer(data, dtype=np.uint8, count=128, offset=base + 2)
        lo = (ql & 0x0F).astype(np.float32)
        hi = (ql >> 4).astype(np.float32)
        q_base = np.empty(256, dtype=np.float32)
        q_base[0::2] = lo
        q_base[1::2] = hi

        # High 2 bits (4 per byte)
        qh = np.frombuffer(data, dtype=np.uint8, count=64, offset=base + 130)
        for i in range(256):
            byte_idx = i // 4
            bit_idx = (i % 4) * 2
            high_bits = (qh[byte_idx] >> bit_idx) & 0x03
            q_base[i] += high_bits * 16.0

        # Scales: 16 values, one per 16-element sub-block
        scales_raw = np.frombuffer(data, dtype=np.uint8, count=16, offset=base + 194)
        scales = scales_raw.astype(np.float32)
        # Apply signed interpretation
        for i in range(16):
            if scales[i] >= 128:
                scales[i] -= 256

        for s in range(16):
            out[b * 256 + s * 16:(b * 256 + (s + 1) * 16)] = q_base[s * 16:(s + 1) * 16] * scales[s] * d

    return out.reshape(shape)


def dequant_iq4_nl(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """IQ4_NL: ggml block_iq4_nl -- 32 elements per block, 18 bytes each.
    This is NOT a K-quant super-block format (no dmin, no per-sub-block
    scales). Real layout, from ggml-quants.h:
        typedef struct {
            ggml_fp16_t d;       // single scale for the whole 32-elem block
            uint8_t qs[QK4_NL/2]; // 16 bytes = 32 packed 4-bit indices
        } block_iq4_nl;
    Each 4-bit index looks up a *fixed* non-linear codebook (kvalues_iq4nl
    in ggml-quants.c) and the result is scaled by d directly (no /127).
    """
    n = int(np.prod(shape))
    n_blocks = n // IQ4_NL_K
    assert n_blocks * IQ4_NL_K == n, f"IQ4_NL: {n} not divisible by {IQ4_NL_K}"

    # Fixed IQ4_NL codebook (kvalues_iq4nl, ggml-quants.c) -- NOT symmetric,
    # NOT divided by 127; d already carries the correct scale.
    IQ4NL_VALUES = np.array([
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    ], dtype=np.float32)

    out = np.empty(n, dtype=np.float32)

    for b in range(n_blocks):
        base = b * IQ4_NL_SIZE
        d = struct.unpack_from("<e", data, base)[0]
        qs = np.frombuffer(data, dtype=np.uint8, count=16, offset=base + 2)
        # v14 fix: element order is NOT byte-interleaved -- elements 0..15
        # are the LOW nibbles of qs[0..15], elements 16..31 the HIGH nibbles
        # (matches gguf-py's canonical reference and the Metal kernel).
        lo = (qs & 0x0F).astype(np.int32)
        hi = (qs >> 4).astype(np.int32)
        q = np.empty(32, dtype=np.float32)
        q[:16] = IQ4NL_VALUES[lo]
        q[16:] = IQ4NL_VALUES[hi]
        out[b * 32:(b + 1) * 32] = q * d

    return out.reshape(shape)


def dequant_bf16(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """BF16: 16-bit brain float. Upper 16 bits of float32."""
    n = int(np.prod(shape))
    bf16 = np.frombuffer(data, dtype=np.uint16, count=n)
    fp32 = (bf16.astype(np.uint32) << 16).view(np.float32)
    return fp32.reshape(shape).copy()


# Dispatch table
DEQUANTIZERS = {
    0: dequant_f32,       # F32
    30: dequant_bf16,     # BF16
    20: dequant_iq4_nl,   # IQ4_NL (v14: registered; nibble order fixed to
                          #  match gguf-py -- lows of bytes 0..15 are elems
                          #  0..15, highs are elems 16..31)
}
# v8: ALL quantized types (Q4_0..Q8_0, K-quants, IQ quants) now dequantize
# through the gguf package's reference implementations via the fallback in
# dequantize() -- the hand-rolled local K-quant versions above had bugs
# (dequant_q5_K index overflow) and are no longer used.


def dequantize(data: bytes, tensor_type: int, shape: tuple[int, ...]) -> np.ndarray:
    """Dispatch to the correct dequantizer.

    `shape` is the GGUF-native logical shape [in, out]. The byte stream is
    laid out numpy-style with the REVERSED shape (out, in) -- reshaping with
    the logical shape directly transpose-scrambles every non-square tensor.
    Dequantize with the reversed shape, then transpose 2D arrays so callers
    always receive logical [in, out] orientation.
    """
    fn = DEQUANTIZERS.get(tensor_type)
    rev = tuple(int(s) for s in reversed(shape))
    if fn is not None:
        arr = fn(data, rev)
    else:
        # v8: K-quants (Q2_K..Q6_K etc.) via the gguf package's reference
        # dequantizers -- required for pre-quantized sources like Q2_K_XL.
        import gguf as _gguf
        try:
            qt = _gguf.GGMLQuantizationType(tensor_type)
        except ValueError:
            raise ValueError(f"No dequantizer for tensor_type={tensor_type}")
        n = int(np.prod(rev))
        import gguf.quants as _q
        cls = next((getattr(_q, name) for name in dir(_q)
                    if isinstance(getattr(_q, name), type)
                    and getattr(getattr(_q, name), "qtype", None) == qt), None)
        if cls is None or n % cls.block_size:
            raise ValueError(f"No dequantizer for tensor_type={tensor_type}")
        if hasattr(cls, "init_grid"):
            cls.init_grid()          # IQ quants lazily build their lookup grid
        nbytes = n // cls.block_size * cls.type_size
        if len(data) < nbytes:
            raise ValueError(f"Q{tensor_type}: need {nbytes} bytes, got {len(data)}")
        flat = cls.dequantize_rows(np.frombuffer(data[:nbytes], np.uint8).reshape(-1, cls.type_size))
        arr = flat.reshape(rev)
    if arr.ndim == 2:
        arr = np.ascontiguousarray(arr.T)
    return arr
