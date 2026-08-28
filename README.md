# ATF — Adaptive Tensor Format for Apple Silicon

[![Release v0.2.0](https://img.shields.io/github/v/release/amgadtewfik/atf?label=release&sort=semver)](https://github.com/amgadtewfik/atf/releases/tag/v0.2.0)
[![macOS DMG](https://img.shields.io/badge/macOS-DMG-blue)](https://github.com/amgadtewfik/atf/releases/tag/v0.2.0)
[![License](https://img.shields.io/github/license/amgadtewfik/atf)](https://github.com/amgadtewfik/atf)

Single-file `.atf` container with custom Metal kernels pushing decode close to the memory-bandwidth ceiling on Apple Silicon.

## ✨ What's new in v0.2.0

- **Tool-call parsing for the OpenAI-compatible server** — `<tool_call>` / `<function=…>` markup is now parsed out of the model stream and emitted as proper `tool_calls` deltas (`finish_reason: "tool_calls"`), so coding agents that speak the OpenAI tool-call format get real tool calls instead of raw markup leaking into chat.
- **Dashboard / Inference / API tabs** — the landing tab is now a live Dashboard with stat tiles (model, engine phase + live prefill %, decode speed, GPU memory, context meter, API server state). Loading a model flips to Inference automatically.
- **Live prefill progress** — long prompts no longer look hung; the status panel, loading bar, and API SSE stream now report `Prefilling N/T tokens` as prefill chunks complete.

## 📥 Download

**Latest release: [v0.2.0](https://github.com/amgadtewfik/atf/releases/tag/v0.2.0)** (August 2026)

Grab the macOS DMG from the [v0.2.0 release page](https://github.com/amgadtewfik/atf/releases/tag/v0.2.0) →

## 🔗 Links

- 🏠 Repository: https://github.com/amgadtewfik/atf
- 🚀 Releases: https://github.com/amgadtewfik/atf/releases
- 📦 Latest (v0.2.0): https://github.com/amgadtewfik/atf/releases/tag/v0.2.0
