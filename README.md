# ATF Chat

<div align="center">

[![Latest Release](https://img.shields.io/github/v/release/amgadtewfik/atf-chat?label=v0.9.0&sort=semver)](https://github.com/amgadtewfik/atf-chat/releases/tag/v0.9.0)
[![License](https://img.shields.io/github/license/amgadtewfik/atf-chat)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-blue)](https://github.com/amgadtewfik/atf-chat)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node.js-18+-green)](https://nodejs.org/)

</div>

A native desktop chat application for running local `.atf` (GGUF-derived) models on Apple Silicon, with an optional OpenAI-compatible API server. Built with Electron and MLX for optimal performance on macOS.

---

## ✨ Features

- **Native macOS app** — Electron shell with MLX-powered inference engine
- **Local-first** — Run `.atf` models entirely offline on Apple Silicon
- **OpenAI-compatible API** — Drop-in replacement for OpenAI clients
- **Model management** — In-app model downloading and switching
- **Streaming responses** — Real-time token streaming in chat
- **Markdown rendering** — Full markdown support with syntax highlighting
- **Context persistence** — SQLite-backed conversation history

---

## 📦 Latest Release

**v0.9.0** — [Download](https://github.com/amgadtewfik/atf-chat/releases/tag/v0.9.0) | [Changelog](https://github.com/amgadtewfik/atf-chat/releases/tag/v0.9.0)

> See [GitHub Releases](https://github.com/amgadtewfik/atf-chat/releases) for full release history.

---

## 🖥 Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | macOS on Apple Silicon (M1/M2/M3/M4) |
| **RAM** | 16 GB minimum (32 GB recommended for larger models) |
| **Disk** | ~10 GB for models and dependencies |
| **Python** | 3.10+ with [uv](https://docs.astral.sh/uv/) (`brew install uv`) |
| **Node.js** | 18+ (for Electron shell) |

---

## 🚀 Quick Start

### Install from Source

```bash
# Clone the repository
git clone https://github.com/amgadtewfik/atf-chat
cd atf-chat

# 1. Python environment (engine + CLI + OpenAI-compatible server)
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-deps        # installs the `atf` package

# 2. Electron app
cd electron
npm install                          # installs deps + rebuilds native modules
npm run vendor                       # builds renderer bundles (markdown-it, highlight.js, DOMPurify)
```

### Runtime Python Environment

The Electron app uses a **separate, self-contained venv** at `electron/app-venv` at runtime (both `run.sh` and the packaged `.app`). Create it once per checkout:

```bash
uv venv electron/app-venv
uv pip install --python electron/app-venv/bin/python -e . --no-deps
uv pip install --python electron/app-venv/bin/python -r requirements.txt
```

> **Note:** `uv venv` does not install `pip` by default. Use `uv pip install --python ...` as shown above, not `electron/app-venv/bin/pip install ...`, unless you create the venv with `uv venv electron/app-venv --seed`.

### Run the App

```bash
# From repo root
./run.sh

# Or from electron/
npm start
```

Models load on demand from the in-app dropdown. Drop `.atf` files into the configured models folder (Settings → Model & Context). Model checkpoints are hosted at [`amgadtewfik/atf`](https://huggingface.co/amgadtewfik/atf) on Hugging Face.

---

## 🛠 Development

```bash
cd electron
npm install
npm run vendor      # rebuild renderer/vendor bundles
npm start           # run from source
node ../tests/test_renderer_logic.js   # renderer unit tests
```

### Native Module Build (better-sqlite3)

`electron/.npmrc` pins `node-gyp`/`prebuild-install` to build against **Electron's** headers (`runtime=electron`), not the system Node. If `npm install` fails on `better-sqlite3` after a clean `rm -rf electron/node_modules`, verify `.npmrc`'s `target` matches a current Electron 33.x version (shown in `package.json`'s `allowScripts` key).

---

## 📦 Production Build

```bash
cd electron
npm install
npm run vendor
npm run dist:dmg     # -> electron/dist/ATF Chat-<arch>.dmg
# or: npm run dist   # unsigned .app bundle only (--mac dir)
```

`electron-builder` bundles `electron/app-venv`, `atf/`, `bridge/`, and `pyproject.toml` into the app's `Resources/` (see `build.extraResources` in `electron/package.json`). **`electron/app-venv` must exist and be up to date before building**, or the packaged app ships without a Python backend.

### Verify Before Shipping

1. Open the resulting `.app` (or mount the `.dmg`)
2. Confirm it launches
3. Load a model from the dropdown
4. Send a chat message and verify streaming response

---

## 🌐 OpenAI-Compatible Server

Start/stop from the API tab in-app, or from the CLI:

```bash
python -m atf.server_openai <path-to-model.atf> --port 8000
```

Point any OpenAI-compatible client at `http://localhost:8000/v1`.

---

## 📁 Project Structure

```
atf-chat/
├── atf/                    # Python inference engine & API server
│   ├── server_openai.py    # OpenAI-compatible server
│   └── ...
├── bridge/                 # Python ↔ Electron IPC bridge
├── electron/               # Electron application
│   ├── app-venv/           # Runtime Python environment (created at build/run)
│   ├── src/                # Renderer & main process code
│   ├── package.json
│   └── ...
├── tests/                  # Unit tests
├── run.sh                  # Launcher script
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Python package config
└── update.md               # Development log & release history
```

---

## 🔗 Resources

- **Models:** [amgadtewfik/atf](https://huggingface.co/amgadtewfik/atf) on Hugging Face
- **Issues:** [GitHub Issues](https://github.com/amgadtewfik/atf-chat/issues)
- **Releases:** [GitHub Releases](https://github.com/amgadtewfik/atf-chat/releases)
- **Changelog:** [GitHub Releases](https://github.com/amgadtewfik/atf-chat/releases)

---

## 📄 License

[MIT License](LICENSE) — see LICENSE file for details.

