# ATF — Adaptive Tensor Format for Apple Silicon

[![Release v0.3.0](https://img.shields.io/github/v/release/amgadtewfik/atf?label=release&sort=semver)](https://github.com/amgadtewfik/atf/releases/tag/v0.3.0)
[![macOS DMG](https://img.shields.io/badge/macOS-DMG-blue)](https://github.com/amgadtewfik/atf/releases/tag/v0.3.0)
[![License](https://img.shields.io/github/license/amgadtewfik/atf)](https://github.com/amgadtewfik/atf)

Single-file `.atf` container with custom Metal kernels pushing decode close to the memory-bandwidth ceiling on Apple Silicon.

## ✨ What's new in v0.3.0

- **Professional local inference UI (Electron + MLX)** — Qwen3.5-9B desktop chat with a **65536-token context window** (chunked prefill, preallocated KV cache), real context management, persistent conversations, and an optional **OpenAI-compatible server** (`/v1/chat/completions`, SSE streaming).
- **Persistent conversations** — multi-chat sidebar (rename / delete / export to Markdown or JSON via right-click). Sessions survive reloads and app restarts.
- **Real context management** — history is truncated newest-first against a tokenizer-measured budget; the context meter shows true prompt tokens from the engine (`usage` events).
- **Full sampling controls** — temperature, top_p, repeat penalty, system prompt, max response tokens (up to 32k).
- **Proper markdown rendering** — markdown-it + highlight.js + DOMPurify, throttled to animation frames during streaming.
- **Message actions** — copy answer, regenerate, edit-and-resend any user turn.
- **Engine resilience** — crash banner with exponential-backoff auto-restart; errors surface as toasts.
- **Native shell** — application menu (⌘N new chat, ⌘K clear, ⌘, settings), window-state persistence, sandboxed renderer.

## 📥 Download

**Latest release: [v0.3.0](https://github.com/amgadtewfik/atf/releases/tag/v0.3.0)** (August 2026)

Grab the macOS DMG from the [v0.3.0 release page](https://github.com/amgadtewfik/atf/releases/tag/v0.3.0) →

## 🔗 Links

- 🏠 Repository: https://github.com/amgadtewfik/atf
- 🚀 Releases: https://github.com/amgadtewfik/atf/releases
- 📦 Latest (v0.3.0): https://github.com/amgadtewfik/atf/releases/tag/v0.3.0
