# ATF Chat v0.12.0 - Release Notes


## Overview
Version 0.12.0 introduces significant foundational work for model quantization and conversion, along with UI refinements and improved build stability. This release also marks a major milestone: the first code contribution made by the ATF model using prime-agent and qwen3.8:27b.

## New Features & Enhancements

### 🛠 Multi-tier Conversion CLI Pipeline
Introduced an experimental CLI pipeline for model quantization and conversion. This includes:
- **New Conversion Logic**: Enhanced `atf/convert.py` and `atf/format.py` to support multi-tier quantization paths.
- **Quantization Tools**: New `atf/quantize.py` for fine-grained control over model precision.
- **Memory Management**: Added `atf/memory_guard.py` to optimize memory usage during large-scale model conversions.
- **Metal Optimizations**: Updated `atf/gguf_metal.py` to improve GGUF compatibility with Metal kernels.
- **CLI Integration**: Added new commands to `atf/cli.py` to trigger the conversion pipeline.

### 🎨 UI & UX Improvements
- **Temperature Display**: Added a new "temperature pill" in the chat interface to provide real-time visibility into the generation temperature.
- **Terminology Update**: Renamed `tft` (Time for first Token) to `TTFT` (Time to First Token) for better alignment with industry standards.
- **Visual Polish**: Minor adjustments to the renderer to improve the layout of metric pills.

## Bug Fixes & Stability
- **DMG Build Process**: Streamlined the `electron/build.sh` script and cleaned up `package.json` to ensure more reliable DMG packaging.
- **Testing**: Expanded the test suite with new coverage for:
  - Renderer logic (`tests/test_renderer_logic.js`)
  - Storage tiering systems (`tests/test_storage_tier.py`)

## 🚀 Milestone
- **AI-Driven Development**: This version includes the first successful implementation of a code change generated and applied by the ATF model itself, leveraging the prime-agent framework and the qwen3.8:27b model.
