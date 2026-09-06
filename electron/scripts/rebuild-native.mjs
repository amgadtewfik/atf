#!/usr/bin/env node
// scripts/rebuild-native.mjs
//
// Rebuilds native Node addons (currently just better-sqlite3) against the
// exact NODE_MODULE_VERSION of the Electron binary in node_modules/electron/.
//
// ─── Why this is needed ─────────────────────────────────────────────────
// better-sqlite3's own `install.js` runs node-gyp against the system Node,
// producing a binding that matches *that* Node's ABI. Electron 33 bundles
// its own Node runtime with a different NODE_MODULE_VERSION (130) than the
// system Node 26.5.0 (147), so loading the binding inside Electron throws:
//
//   Error: The module .../better_sqlite3.node was compiled against a
//   different Node.js version using NODE_MODULE_VERSION 147. This version
//   of Node.js requires NODE_MODULE_VERSION 130.
//
// better-sqlite3 catches the load error inside its own bindings loader
// and re-throws it as a generic "Database is not a constructor" — which
// looks like a bug in our main.cjs (which does `new Database(...)`) and
// silently kills every sessions-* IPC. From the renderer this surfaces
// as a chat sidebar that shows titles but no message bodies (the welcome
// screen always renders when a session has zero messages — see the v19
// SQLite migration in main.cjs and rebuildChatFromSession in app.js).
//
// @electron/rebuild knows how to invoke node-gyp with Electron's headers
// so the binding matches the Electron ABI. This script wraps it so the
// rebuild runs idempotently after every `npm install` in electron/.
//
// ─── Usage ──────────────────────────────────────────────────────────────
//   node scripts/rebuild-native.mjs                  # rebuild if needed
//   node scripts/rebuild-native.mjs --force         # always rebuild
//
// Wired into the existing self-heal pipeline: scripts/ensure-electron.mjs
// calls this after a successful electron repair, and npm's postinstall
// hook calls ensure-electron. So a fresh clone gets a working binding on
// the very first run.
//
// ─── Idempotency check ─────────────────────────────────────────────────
// Without --force, we try to load each binding under the Electron-bundled
// Node (via ELECTRON_RUN_AS_NODE). If it loads cleanly, the binding is
// already correct for Electron and we skip. A no-op runs in ~50 ms.

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename   = fileURLToPath(import.meta.url);
const SCRIPT_DIR   = path.dirname(__filename);
const ELECTRON_DIR = path.resolve(SCRIPT_DIR, '..', 'node_modules', 'electron');
const ROOT         = path.resolve(SCRIPT_DIR, '..');

// Native modules that must match the Electron ABI. Add more here if we
// ever pull in another native dep (e.g. oniguruma, keytar). Each entry
// gives the npm package name and the on-disk binding filename — which
// is NOT always the same as the package name (better-sqlite3 uses
// "better_sqlite3.node" with an underscore).
const NATIVE_MODULES = [
  { name: 'better-sqlite3', binding: 'better_sqlite3.node', reason: 'chat sessions persist in SQLite (v19+)' },
];

const log  = (...a) => console.log ('[rebuild-native]', ...a);
const warn = (...a) => console.warn('[rebuild-native]', ...a);

function getElectronExe () {
  const pathFile = path.join(ELECTRON_DIR, 'path.txt');
  if (!fs.existsSync(pathFile)) {
    throw new Error(`electron path.txt missing at ${pathFile}; run scripts/ensure-electron.mjs first`);
  }
  return path.join(ELECTRON_DIR, 'dist', fs.readFileSync(pathFile, 'utf-8').trim());
}

function bindingLoadsUnderElectron (mod) {
  // Probe: spawn Electron as plain Node (ELECTRON_RUN_AS_NODE=1) and try
  // to require the package. If it loads cleanly, the binding already
  // matches Electron's NODE_MODULE_VERSION and no rebuild is needed.
  const modPath = path.resolve(ROOT, 'node_modules', mod.name);
  const exe = getElectronExe();
  const r = spawnSync(exe, ['-e', `require(${JSON.stringify(modPath)})`], {
    env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
    encoding: 'utf-8',
  });
  return r.status === 0;
}

function needsRebuild (mod, force) {
  if (force) return { needed: true, reason: 'forced' };
  const binding = path.join(ROOT, 'node_modules', mod.name, 'build', 'Release', mod.binding);
  if (!fs.existsSync(binding)) return { needed: true, reason: `binding missing at ${binding}` };
  if (!bindingLoadsUnderElectron(mod)) {
    return { needed: true, reason: `binding fails to load under Electron (likely NODE_MODULE_VERSION mismatch)` };
  }
  return { needed: false, reason: 'binding already matches Electron ABI' };
}

function runRebuild (modules) {
  // @electron/rebuild's bin symlink sometimes points at a missing cli.js
  // (the package ships TS sources; the build step is skipped on some
  // installs). Call lib/cli.js directly so the path is stable.
  const cliJs = path.join(ROOT, 'node_modules', '@electron', 'rebuild', 'lib', 'cli.js');
  if (!fs.existsSync(cliJs)) {
    throw new Error(`@electron/rebuild CLI missing at ${cliJs}; run \`npm install\` first`);
  }
  const args = ['-f', '-w', ...modules];
  log(`invoking: node ${path.relative(ROOT, cliJs)} ${args.join(' ')}`);
  const r = spawnSync(process.execPath, [cliJs, ...args], {
    cwd: ROOT,
    stdio: 'inherit',
  });
  if (r.status !== 0) {
    throw new Error(`electron-rebuild failed (exit=${r.status})`);
  }
}

function main () {
  const force = process.argv.includes('--force');
  const installed = NATIVE_MODULES.filter((m) =>
    fs.existsSync(path.join(ROOT, 'node_modules', m.name))
  );
  if (!installed.length) {
    log('no native modules installed; nothing to do');
    return 0;
  }
  const toRebuild = [];
  for (const m of installed) {
    const verdict = needsRebuild(m, force);
    if (verdict.needed) {
      log(`${m.name}: rebuild needed (${verdict.reason}; ${m.reason})`);
      toRebuild.push(m.name);
    } else {
      log(`${m.name}: ${verdict.reason} — skipping`);
    }
  }
  if (!toRebuild.length) return 0;
  runRebuild(toRebuild);
  log('rebuild complete');
  return 0;
}

process.exit(main());
