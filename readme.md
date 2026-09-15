# ATF Chat 0.13.0 Release Notes
**Release:** [v0.13.0 on GitHub](https://github.com/amgadtewfik/atf/releases/tag/v0.13.0)


## 🚀 Overview
Version 0.13.0 marks a significant leap in inference flexibility and system intelligence. The headline feature is **Storage-Tier Adaptivity**, which allows the engine to dynamically toggle between a lightweight "Base" quantization for speed and a high-precision "Residual" tier for complex reasoning. This release also integrates a robust system health monitor to prevent memory exhaustion and celebrates a milestone in AI-driven development, featuring the first direct code contributions from the ATF model via prime-agent.

## New Features & Enhancements

### 🧠 Storage-Tier Adaptivity (Core Engine)
Implemented a dynamic system to balance inference quality and resource usage:
- **Dual-Tier Execution**: Models can now operate in **Base Tier** (optimized for instant chat) or **Residual Tier** (optimized for deep reasoning/high precision).
- **Fused Residual GEMV Metal Kernel**: A high-performance Metal kernel in `atf/gguf_metal.py` that performs dual int8 buffer SIMD-lane accumulation in a single dispatch, eliminating the need for intermediate dequantized buffers.
- **Adaptive Router**: The engine now uses a router (`atf/router.py`) to automatically select the appropriate tier based on prompt difficulty.
- **Intelligent Memory Eviction**: Added support for `MADV_DONTNEED` via `atf/model.py` to efficiently evict residual tier memory when reverting to the Base tier.

### 🛡️ System Health & Memory Guard
Introduced `atf/memory_guard.py` to ensure system stability:
- **Real-time Headroom Tracking**: Monitors available system memory via Darwin `host_statistics64`.
- **Automatic Tier Clamping**: The system automatically forces "Base Tier" execution if available memory drops below a safety threshold (1.5 GB), preventing OOM crashes during high-precision inference.

### 🛠️ Experimental Quantization Tools
Laid the foundation for multi-tier model creation:
- **Residual Quantization**: Implemented `quantize_residual` and `dequantize_residual` in `atf/quantize.py` to compute and apply precision corrections.
- **ResidualSection Format**: Introduced the `b"DELT"` binary section in `atf/format.py` for storing residual tier data with 16KB-page alignment.
- **CLI Pipeline (Experimental)**: Initial implementation of the conversion pipeline in `atf/cli.py` and `atf/convert.py` to support the generation of tiered `.atf` files.

### 🎨 UI & UX Improvements
- **Temperature Display**: Added a "temperature pill" to the chat interface for real-time visibility into generation settings.
- **Industry Standard Metrics**: Renamed `tft` to `TTFT` (Time to First Token) to align with LLM benchmarking standards.
- **Visual Polish**: Refined the metric pill layout in the renderer for better readability.

## Bug Fixes & Stability
- **DMG Build Process**: Streamlined `electron/build.sh` and cleaned up `package.json` for more reliable macOS packaging.
- **Enhanced Testing**: Added comprehensive tests for:
  - Renderer logic (`tests/test_renderer_logic.js`)
  - Storage tiering and memory guard systems (`tests/test_storage_tier.py`)

## v13 goals (in priority order)

1. **Fix the two stale tests** — `tests/test_inference.py` imports
   `_rmsnorm_per_head` (removed in v12) and `tests/test_q4nl_conversion.py`
   expects an IQ4_NL decoder at `gguf_io.DEQUANTIZERS[20]`. Port the IQ4_NL
   dequantizer and refresh the inference tests so the suite is fully green.
2. **Speculative decoding with a draft model** — the one big architectural
   decode lever left. Kernels are at the DRAM floor (~84 GB/s), so further
   throughput requires verifying multiple tokens per weight pass. Draft
   candidate: a small Q4/Q8 quant of a same-tokenizer model; verify against
   the 27B logits, accept-prefix semantics.
3. **Logit trimming** — 248k vocab lm_head costs ~10 ms/token even via the
   fast Q4_K kernel. Trim to the active byte-level candidate set where
   safe (temperature/top-k aware), or defer full-vocab softmax until the
   sampler needs it.
