# ATF — Adaptive Tensor Format for Apple Silicon

[![Release v0.4.0](https://img.shields.io/github/v/release/amgadtewfik/atf?label=release&sort=semver)](https://github.com/amgadtewfik/atf/releases/tag/v0.4.0)
[![macOS DMG](https://img.shields.io/badge/macOS-DMG-blue)](https://github.com/amgadtewfik/atf/releases/tag/v0.4.0)
[![License](https://img.shields.io/github/license/amgadtewfik/atf)](https://github.com/amgadtewfik/atf)

Single-file `.atf` container with custom Metal kernels pushing decode close to the memory-bandwidth ceiling on Apple Silicon.

## ✨ What's new in v0.4.0

- **MoE convert balloon bug fix** — Pre-quantized GGUF sources (UD-Q2_K_XL, Q4_K_M, IQ4_NL) no longer double in size when converted to ATF. `Dirk-Qwen3.8-27B-UD-Q2_K_XL.gguf` (9.83 GB) now produces a **9.48 GB** ATF file instead of 19.4 GB.
- **Subject Relevance LOD** — Engine produces larger output for high-relevance requests while staying smaller for low-relevance ones via per-domain LOD tables and expert classification on `ffn_gate` statistics.
- **MTP streaming support** — Multi-Token Prediction tensors detected and excluded from base engine; trunk executes independently.
- **Qwen3.5 / 3.6 / 3.8 architecture support** — Full support for dense, MoE, and MTP variants including gated RMSNorm, per-head-interleaved attn_q, GDN recurrence, and tiled V-head pairing.
- **Electron UI overhaul (v15)** — Persistent conversations with multi-chat sidebar, stateless engine protocol, real context management (65536-token window), full sampling controls, markdown rendering, message actions, engine resilience with auto-restart, and native macOS shell.
- **OpenAI-compatible server** — Built-in `/v1/chat/completions` with SSE streaming; port verified with real `/v1/models` probe.

## 📥 Download

**Latest release: [v0.4.0](https://github.com/amgadtewfik/atf/releases/tag/v0.4.0)** (August 2026)

Grab the macOS DMG from the [v0.4.0 release page](https://github.com/amgadtewfik/atf/releases/tag/v0.4.0) →

## 🔗 Links

- 🏠 Repository: https://github.com/amgadtewfik/atf
- 🚀 Releases: https://github.com/amgadtewfik/atf/releases
- 📦 Latest (v0.4.0): https://github.com/amgadtewfik/atf/releases/tag/v0.4.0
