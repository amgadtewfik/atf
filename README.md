# ATF — Adaptive Tensor Format for Apple Silicon

[![Release v0.5.0](https://img.shields.io/github/v/release/amgadtewfik/atf?label=release&sort=semver)](https://github.com/amgadtewfik/atf/releases/tag/v0.5.0)
[![macOS DMG](https://img.shields.io/badge/macOS-DMG-blue)](https://github.com/amgadtewfik/atf/releases/tag/v0.5.0)
[![License](https://img.shields.io/github/license/amgadtewfik/atf)](https://github.com/amgadtewfik/atf)

Single-file `.atf` container with custom Metal kernels pushing decode close to the memory-bandwidth ceiling on Apple Silicon.

## ✨ What's new in v0.5.0

The v0.5.0 release is a **visual and UX overhaul** of the ATF Chat Electron UI. No engine changes — this release is purely about making the app look professional, feel polished, and respect the user's preferences.

- **Light, Dark, and Auto themes** — full theme system with a pre-paint preload script (no flash of the wrong theme on launch), live updates when macOS flips its appearance, and a "follows system" auto mode. Toggle from the titlebar sun/moon icon, the View menu, or **⇧⌘L**.
- **Six accent colors** — indigo, violet, teal, green, amber, rose. Each accent has a tuned palette for both light and dark so contrast stays readable across buttons, focus rings, gradients, and the dashboard hero.
- **Redesigned Settings panel** — sidebar-nav layout with five sections: Appearance, Chat & Generation, Model & Context, Shortcuts, Advanced. Settings save automatically as you adjust them.
- **New keyboard shortcuts** — **⇧⌘L** toggle theme, **⌘1–5** switch tabs.
- **Visual polish** — larger properly-svg titlebar buttons, refined tab rail with sliding active indicator, wider sidebar, cleaner composer, themed code panes, animation toggle.
- **Forward-looking: Qwen4 architecture (QSA + n-gram)** — the structural code paths for Qwen Sparse Attention and n-gram embedding are wired in and validated on synthetic data. The real-weight path is gated on a public Qwen4 GGUF landing; until then the engine correctly detects the absence of QSA weights in the existing Qwen3.5/3.8 checkpoints.
- **Bug fixes** — self-referencing CSS variables that broke every accent-based style (`--accent: var(--accent)` → empty), and a CSP violation from the inline theme-pre-paint script (extracted to `theme-preload.js`).

## 📥 Download

**Latest release: [v0.5.0](https://github.com/amgadtewfik/atf/releases/tag/v0.5.0)** (September 2026)

Grab the macOS DMG from the [v0.5.0 release page](https://github.com/amgadtewfik/atf/releases/tag/v0.5.0) →

## 🔗 Links

- 🏠 Repository: https://github.com/amgadtewfik/atf
- 🚀 Releases: https://github.com/amgadtewfik/atf/releases
- 📦 Latest (v0.5.0): https://github.com/amgadtewfik/atf/releases/tag/v0.5.0
