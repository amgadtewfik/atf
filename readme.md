# ATF Chat v0.10.0

Release package for **ATF Chat v0.10.0**.

[View the v0.10.0 release on GitHub](https://github.com/amgadtewfik/atf/releases/tag/v0.10.0)

---

## What's New in v0.10.0

**Latest v5 model format Support (2026-09-07)**: QSA was promoted from structural-only
coverage to an exercised prefill-plus-decode path. Its indexer keys now stay
aligned with the main KV cache across chunks, partial final blocks are
handled safely, main attention applies RoPE, and sparse attention uses the
correct per-query gather layout. PLE now performs causal dilated sequence
convolution and preserves the signed gate. True-MoE routing batches all
token assignments for each unique selected expert, avoiding repeated gate,
up, and down projection launches. These changes reduce redundant work and
prevent previously hidden shape/runtime failures; real Qwen4 performance
and quality gains remain unmeasured because no compatible production
checkpoint is available locally.

## Latest measured performance

**Latest measured performance** (M4 16 GB, greedy seed=42): 9B IQ4_NL
11.5 tok/s decode; 27B Q2_K_XL 5.75 tok/s decode
(OOM-guard protected, needs `ATF_OM_SKIP=1` at this context size); 35B-A3B
MoE 0.02 tok/s (memory-bound at 16 GB — edge of viability, not yet a
usable configuration). Decode is DRAM-bandwidth-bound (~84 GB/s M4
ceiling); prefill on a fused-GEMM path reaches ~85 tok/s on the 9B (v16).

| Model | File format | Prefill | Decode | Peak memory | Notes |
|---|---|---:|---:|---:|---|
| Qwen3.5-9B IQ4_NL | ATF v4-era | ~85 tok/s | 11.5 tok/s | 9.1 GB | Fused-GEMM prefill |
| Qwen3.8-27B Q2_K_XL | ATF v4-era | not recorded | 5.75 tok/s | 9.8 GB | OOM guard used |
| Qwen-AgentWorld-35B-A3B | v2 file, v5 reader | 0.1 tok/s | 4.67 tok/s | 11.57 GB | 256 experts, top-k 8 |

## Download

| File | Platform | Architecture | Size |
| --- | --- | --- | --- |
| `ATF Chat-arm64.dmg` | macOS | Apple Silicon (arm64) | ~202 MB |

## Installation

1. Download `ATF Chat-arm64.dmg`.
2. Open the DMG file.
3. Drag **ATF Chat** to the Applications folder.
4. Launch ATF Chat from Applications.

## Requirements

- macOS on an Apple Silicon Mac.

## Checksum

SHA-256:

```text
ec8d1e17459beaa3ee5b61ee3eaa987e2f8b6ef37f7366f7533ff23b7cec77ac
```
