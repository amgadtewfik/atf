# ATF Chat v0.11.0

Release package for **ATF Chat v0.11.0**.

[View the v0.11.0 release on GitHub](https://github.com/amgadtewfik/atf/releases/tag/v0.11.0)

---

## What's New in v0.11.0

### ⚡ Performance & Inference
- **Metal Kernel Optimizations**: Added and improved support for various quantization formats, specifically including `IQ1_M`, significantly enhancing decoding speed.
- **Prefill & Generation Timing**: Fixed timing issues for the 27B model's prefill and generation stages.
- **New Metrics**: 
  - Introduced **TFT (Time for First Token)** tracking.
  - Added detailed timing for the transition from prefill to the first decode token.
- **Benchmarking**: Integrated comprehensive benchmark scripts and results to track model performance.

### 🐞 Bug Fixes & Stability
- **API Streaming**: Fixed critical engine and model errors encountered during API streaming.
- **Model Naming**: Standardized model referencing by removing the trailing `_v5` suffix in favor of version numbering.
- **Electron Build**: Fixed resource mapping for the packaged app to ensure the Python bridge and venv are correctly located.

### 🎨 User Experience
- **Status Labels**: Updated the UI to change "Thinking" to **"Processing"** during the prefill stage for better clarity.
- **UI Polish**: Minor capitalization and labeling adjustments for model states.


## Latest measured performance

**Latest measured performance** (M4 16 GB, greedy seed=42): 9B IQ4_NL
Prompt: fixed filler + question (217 words), max_tokens=128, temperature=0 (greedy), thinking=off. Same prompt/config for every model.

| Model | Size | Prefill | Decode |
|---|---|---|---|
| Qwen3.5-9B-IQ4_NL.atf | 5.52 GB | 65.00 tok/s (274 tok, 4.20s) | 14.04 tok/s (104 tok, 7.40s) |
| Qwen3.8-27B-UD-Q2_K_XL.atf | 9.52 GB | 18.90 tok/s (274 tok, 14.50s) | 4.97 tok/s (96 tok, 19.30s) |
| Qwen-AgentWorld-35B-A3B-UD-IQ2_M.atf | 11.64 GB | 24.30 tok/s (274 tok, 11.30s) | 6.93 tok/s (105 tok, 15.20s) |

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
