# ATF Chat v0.11.0 - Release Notes

## 🚀 Overview
Version `0.11.0` focuses on significant performance optimizations for the Metal kernel, improved timing metrics for inference, and general stability fixes for the Electron wrapper and API streaming.

[View the v0.11.0 release on GitHub](https://github.com/amgadtewfik/atf/releases/tag/v0.11.0)

## 🛠 Changes

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

## 📦 Distribution
This build includes the pre-bundled Python environment and the ATF bridge, optimized for macOS (arm64).

---
*Generated on 2026-09-11*
