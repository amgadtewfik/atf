# ATF Chat — local inference UI (Electron + MLX)

Desktop chat app for running local `.atf` (GGUF-derived) models on Apple
Silicon, with an optional OpenAI-compatible API server. See `update.md` for
the changelog.

## Requirements

- macOS on Apple Silicon, 16 GB RAM, ~10 GB disk (`mlx` is Apple-Silicon-only)
- Python 3.10+ and [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- Node.js 18+ (for the Electron shell)

## Install from source

```sh
git clone https://github.com/amgadtewfik/atf-chat
cd atf-chat

# 1. Python environment (engine + CLI + OpenAI-compatible server)
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-deps        # installs the `atf` package itself

# 2. Electron app
cd electron
npm install                          # also runs postinstall (ensure-electron + native rebuild)
npm run vendor                       # builds renderer/vendor/*.min.js (markdown-it, highlight.js, DOMPurify)
```

### Two venvs, one project

`requirements.txt` / `pyproject.toml` set up the *root* Python env above. At
runtime, though, the Electron app (both `run.sh` and the packaged `.app`)
actually loads Python from a **separate, self-contained venv at
`electron/app-venv`** — see `electron/app_venv.cjs` for why. Create it once
per checkout:

```sh
uv venv electron/app-venv
uv pip install --python electron/app-venv/bin/python -e . --no-deps
uv pip install --python electron/app-venv/bin/python -r requirements.txt
```

(Note: `uv venv` does not install `pip` into the environment by default, so
`electron/app-venv/bin/pip` won't exist — use `uv pip install --python ...`
as above, not `electron/app-venv/bin/pip install ...`, unless you create the
venv with `uv venv electron/app-venv --seed`.)

`run.sh` checks for `electron/app-venv/bin/python3` and prints a similar
recipe (using the stdlib `venv` module, which does bundle pip) if it's missing.

## Run

```sh
./run.sh          # from the repo root — launches from source
# or, from electron/:
npm start
```

Models load on demand from the in-app dropdown; drop `.atf` files into the
configured models folder (see Settings → Model & Context). Model checkpoints
are hosted at [`amgadtewfik/atf`](https://huggingface.co/amgadtewfik/atf) on
Hugging Face.

## Development

```sh
cd electron
npm install
npm run vendor   # rebuild renderer/vendor bundles (markdown-it, hljs)
npm start        # run the app from source
node ../tests/test_renderer_logic.js   # renderer unit tests
```

### Native module build (better-sqlite3)

`electron/.npmrc` pins node-gyp/prebuild-install to build against **Electron's**
headers (`runtime=electron`), not whatever system Node happens to run `npm
install`. Without it, a `better-sqlite3` compile can fail outright on a newer
system Node (V8 removes APIs like `GetPrototype`/`GetIsolate`/
`PropertyCallbackInfo::This` that the addon still uses) before this project's
own `postinstall` (`scripts/ensure-electron.mjs` → `scripts/rebuild-native.mjs`)
ever gets a chance to run. If `npm install` still fails on `better-sqlite3`
after a clean `rm -rf electron/node_modules`, check that `.npmrc`'s `target`
still matches a current Electron 33.x version (the `allowScripts` key in
`package.json` shows the resolved version). Note: npm 10+ prints "Unknown
project config" warnings for these `.npmrc` keys — that's cosmetic (npm still
forwards them to node-gyp as env vars); a successful build logs
`better-sqlite3: binding already matches Electron ABI` from
`rebuild-native.mjs`.

## Production build

```sh
cd electron
npm install
npm run vendor
npm run dist:dmg     # -> electron/dist/ATF Chat-<arch>.dmg
# or: npm run dist   # unsigned .app bundle only (--mac dir)
```

`electron-builder` bundles `electron/app-venv`, `atf/`, `bridge/`, and
`pyproject.toml` into the app's `Resources/` (see `build.extraResources` in
`electron/package.json`) — so **`electron/app-venv` must exist and be up to
date before building**, or the packaged app ships without a Python backend.
Native modules (`better-sqlite3`) are rebuilt against Electron's Node ABI
automatically via `scripts/rebuild-native.mjs`.

Verify a build before shipping: open the resulting `.app` (or mount the
`.dmg`), confirm it launches, a model loads from the dropdown, and a chat
message streams a response.

## OpenAI-compatible server

Start/stop from the API tab in-app, or from the CLI:

```sh
python -m atf.server_openai <path-to-model.atf> --port 8000
```

Point any OpenAI-compatible client at `http://localhost:8000/v1`.
