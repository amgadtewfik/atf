# ATF Chat v0.15.0 — Release Notes

**Release:** [v0.15.0 on GitHub](https://github.com/amgadtewfik/atf/releases/tag/v0.15.0)

This release focuses on memory stability, prefill performance, and the transition to a more robust multi-backend architecture. The core objective of v0.15.0 is to eliminate "auto-clamp" memory pressure and reduce first-token latency.

## 🚀 Key Highlights

### 1. Memory & Prefill Optimizations
- **Increased Prefill Throughput**: `GenConfig.prefill_chunk` increased from **1024 $\to$ 4096**, reducing iteration overhead and improving GPU utilization.
- **Eager KV Allocation**: Updated `KVCache._ensure` to pre-allocate the full `max_context` capacity upfront. This eliminates synchronous growth stalls during long-context prefills.
- **Intelligent Memory Guard**:
    - Added explicit guidance messages in `atf/engine.py` to steer users toward `ATF_KV_SSD=1` and `prefill_chunk` reductions when memory pressure is detected.
    - **Auto-SSD Tiering**: The engine now automatically enables the SSD-paged KV cache if the GPU-only budget covers less than 50% of the requested context, preventing silent truncation.

### 2. Generation & Latency Improvements
- **Sync-Free Sampling**: Gated the NaN/Inf trap in `_sample` behind diagnostic flags. This removes a costly host/GPU synchronization on every single decode token, significantly reducing per-token latency.
- **GDN Session-Level Compilation**: Moved Gated DeltaNet (GDN) recurrence compilation from per-block to session-level using packed weight tensors. This removes the $\sim$80-90s first-token stall caused by JIT-tracing.

### 3. Architectural Evolution
- **Multi-Backend Proxy**: Implemented a backend-agnostic tensor proxy layer (`atf/backend.py`). The engine now supports both **MLX (macOS/Metal)** and **PyTorch (Windows/Linux/CUDA/CPU)** backends, providing a correctness baseline for cross-platform validation.
- **KV-SSD Orphan Sweep**: Added an automated startup sweep to identify and remove orphaned `.kvmm` and `.meta.npz` files left by abnormal process exits, keeping the runtime cache clean.

### 4. Infrastructure & Tooling
- **Formal Benchmarking Suite**: Introduced `benchmarks/` covering MoE routing, context scaling, and prefill performance.
- **Enhanced Test Harness**: Expanded `tests/` to include automated validation for prefill progress and KV SSD persistence.
- **Optimization Roadmap**: Formalized the long-term plan in `docs/Rec&Impl.md`.

## 🛠 Technical Changes
- **Version**: Bumped to `0.15.0`.
- **Packaging**: Fixed `electron-builder` configuration to correctly include `update.js` and `electron-updater` in the packaged `.app` bundle.
- **Env Vars**: 
    - `ATF_KV_SSD=1`: Recommended for large models/contexts to avoid memory clamping.
    - `ATF_OM_SKIP=1`: Bypasses the memory guard (use with caution).

## 📝 Summary of Fixes
- Fixed silent context truncation when GPU KV budget was exceeded.
- Resolved "module not found" crashes in the packaged Electron app.
- Fixed stale data accumulation in the SSD KV cache.
- Eliminated per-token synchronization stalls during sampling.

---
*For detailed status and historical changes, please refer to `docs/STATUS.md`.*
