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