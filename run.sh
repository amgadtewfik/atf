#!/bin/sh
# ATF — launch from source. Uses the bundled electron/app-venv as the Python
# interpreter (this is the same venv that ships inside the packaged .app, so
# dev and release stay in sync). The shared venv at /Volumes/SSD5/Ai/atf/.venv
# is no longer used — the per-version editable `atf` install lives inside
# electron/app-venv instead.
set -e

# Finder / LaunchServices give us a minimal PATH (/usr/bin:/bin:...);
# npm and node live in Homebrew, so add the usual locations if missing.
case ":$PATH:" in
  *":/opt/homebrew/bin:"*) ;;
  *) [ -d /opt/homebrew/bin ] && PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH" ;;
esac

cd "$(dirname "$0")"

VENV="$PWD/electron/app-venv"
[ -x "$VENV/bin/python3" ] || {
  echo "Bundled venv missing at $VENV" >&2
  echo "Create it with:" >&2
  echo "  /opt/homebrew/opt/python@3.13/bin/python3.13 -m venv electron/app-venv" >&2
  echo "  electron/app-venv/bin/pip install -e . --no-deps" >&2
  echo "  electron/app-venv/bin/pip install numpy gguf click rich mlx transformers gigatoken==0.10.0" >&2
  exit 1
}

# point imports at THIS version's atf/ package (bundled venv, per-version code)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# no model is required at launch -- pick one from the in-app dropdown.

cd electron
[ -d node_modules ] || npm install

# Self-heal: if the electron package's binary install is partial or its
# marker files are missing, repair from the local zip cache. No network.
# See scripts/ensure-electron.mjs for the full rationale.
if [ -f scripts/ensure-electron.mjs ]; then
  node scripts/ensure-electron.mjs || {
    echo "ensure-electron failed; run \`cd electron && rm -rf node_modules/electron && npm install\`" >&2
    exit 1
  }
fi

# Self-heal: if the renderer vendor bundles are missing (they are in
# .gitignore so a fresh beta tree does not have them), rebuild from
# the npm-installed deps. esbuild is the only build tool, no network.
# See build-vendor.mjs.
VENDOR_DIR=renderer/vendor
if [ ! -f "$VENDOR_DIR/markdown-it.min.js" ] || \
   [ ! -f "$VENDOR_DIR/highlight.min.js" ] || \
   [ ! -f "$VENDOR_DIR/purify.min.js" ]; then
  echo "Renderer vendor bundles missing; running \`npm run vendor\`..." >&2
  npm run vendor >/dev/null || {
    echo "npm run vendor failed; check electron/build-vendor.mjs" >&2
    exit 1
  }
fi
exec npm start
