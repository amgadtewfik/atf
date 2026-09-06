#!/usr/bin/env node
// scripts/ensure-electron.mjs
//
// PERMANENT FIX for the recurring
//   "Electron failed to install correctly, please delete node_modules/electron
//    and try installing again"
// error that hits every new beta tree.
//
// ─── Why this happens ────────────────────────────────────────────────────
// electron@33.x ships a postinstall hook (`node install.js`) that does:
//   1. download the .zip from @electron/get (cached at
//      ~/Library/Caches/electron/.../electron-v<ver>-<plat>-<arch>.zip)
//   2. extract the .zip into node_modules/electron/dist/  (Electron.app/...)
//   3. fs.writeFileSync(dist/version, <version>)
//   4. fs.promises.writeFile(path.txt, <platformPath>)
//
// Step 2 uses the `extract-zip` package (2.0.1, 2019), which silently hangs
// or aborts on newer Node versions (the failures show up as just two files
// on disk: the 50KB binary stub and the .icns icon, plus a tiny LICENSE).
// Because step 2 silently fails, steps 3-4 are also skipped.
//
// At launch time node_modules/electron/index.js then does:
//   let executablePath = fs.readFileSync('path.txt', 'utf-8');  // undefined
//   throw new Error('Electron failed to install correctly, ...');
// even though the package.json, cli.js, and the stub binary are on disk.
// Hence the misleading "delete node_modules/electron" message.
//
// Re-running `node install.js` does NOT recover, because
// (a) extract-zip still silently aborts on this Node version, and
// (b) even if it succeeded, install.js's isInstalled() short-circuits as
//     soon as path.txt + version + the stub binary are present, and never
//     re-extracts.
//
// ─── What this script does ──────────────────────────────────────────────
// Three escalating repair levels, all idempotent:
//
//   A. text markers missing (full .app on disk) → write them in place
//   B. .app is partial  → wipe dist/ + extract the cached .zip directly
//                          using the OS `unzip` tool (which is bulletproof
//                          and ~3× faster than extract-zip), then write
//                          the markers
//   C. nothing on disk  → bail with a clear message
//
// The zip is reused across every beta tree (the @electron/get cache is
// shared per-version at ~/Library/Caches/electron/...), so step B never
// hits the network. Healthy installs exit in <50 ms. Recovery from a
// partial install takes ~2-4 s on a fast SSD.
//
// ─── How to use it ──────────────────────────────────────────────────────
// Wired automatically by `run.sh` and by `package.json` scripts.prepare.
// You can also run it directly: `node scripts/ensure-electron.mjs`.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __filename   = fileURLToPath(import.meta.url);
const SCRIPT_DIR   = path.dirname(__filename);
const ELECTRON_DIR = path.resolve(SCRIPT_DIR, '..', 'node_modules', 'electron');

const log  = (...a) => console.log ('[ensure-electron]', ...a);
const warn = (...a) => console.warn('[ensure-electron]', ...a);
const readJSON = (p) => JSON.parse(fs.readFileSync(p, 'utf-8'));

function getPlatformPath () {
  const platform = process.env.npm_config_platform || os.platform();
  switch (platform) {
    case 'mas':
    case 'darwin':  return 'Electron.app/Contents/MacOS/Electron';
    case 'freebsd':
    case 'openbsd':
    case 'linux':   return 'electron';
    case 'win32':   return 'electron.exe';
    default: throw new Error('Electron builds are not available on platform: ' + platform);
  }
}

function getArchString () {
  // electron@33 uses darwin-arm64, darwin-x64, linux-x64, win32-x64, etc.
  const a = process.env.npm_config_arch || process.arch;
  const p = process.env.npm_config_platform || process.platform;
  // arm64 on darwin is "arm64" in arch but "darwin-arm64" in zip name
  return a;
}

function findCachedZip (version) {
  // @electron/get's default cacheRoot is ~/.electron on linux/darwin and
  // ~/AppData/Local/electron/Cache on win. In practice on macOS it goes
  // to ~/Library/Caches/electron/<checksum-of-args>/. Scan both.
  const candidates = [
    path.join(os.homedir(), 'Library', 'Caches', 'electron'),
    path.join(os.homedir(), '.electron'),
  ];
  const plat = process.env.npm_config_platform || process.platform;
  const arch = getArchString();
  for (const root of candidates) {
    if (!fs.existsSync(root)) continue;
    // Each subdir is a cache entry; check for the matching zip
    const subdirs = fs.readdirSync(root, { withFileTypes: true })
      .filter((d) => d.isDirectory()).map((d) => path.join(root, d.name));
    for (const sd of subdirs) {
      const name = `electron-v${version}-${plat}-${arch}.zip`;
      const candidate = path.join(sd, name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  return null;
}

function isAppBundleComplete (exe) {
  if (!fs.existsSync(exe)) return false;
  if (process.platform === 'darwin' || process.platform === 'mas') {
    const fw = path.join(path.dirname(path.dirname(exe)), 'Frameworks');
    return fs.existsSync(path.join(fw, 'Electron Framework.framework'));
  }
  return true;
}

function checkInstalled (electronPkg, platformPath) {
  const distDir  = path.join(ELECTRON_DIR, 'dist');
  const pathFile = path.join(ELECTRON_DIR, 'path.txt');
  const verFile  = path.join(distDir, 'version');
  const exe      = path.join(distDir, platformPath);

  const reasons = [];
  let ok = true;

  try {
    if (fs.readFileSync(verFile, 'utf-8').replace(/^v/, '') !== electronPkg.version) {
      ok = false; reasons.push(`dist/version mismatch (expected ${electronPkg.version})`);
    }
  } catch { ok = false; reasons.push('dist/version missing'); }

  try {
    if (fs.readFileSync(pathFile, 'utf-8') !== platformPath) {
      ok = false; reasons.push(`path.txt mismatch (expected ${platformPath})`);
    }
  } catch { ok = false; reasons.push('path.txt missing'); }

  if (!isAppBundleComplete(exe)) {
    ok = false; reasons.push('Electron.app is partial (re-extract needed)');
  }
  return { ok, reasons };
}

function writeMarkers (platformPath, version) {
  const distDir  = path.join(ELECTRON_DIR, 'dist');
  const pathFile = path.join(ELECTRON_DIR, 'path.txt');
  const verFile  = path.join(distDir, 'version');
  fs.mkdirSync(distDir, { recursive: true });
  fs.writeFileSync(verFile,  version);
  fs.writeFileSync(pathFile, platformPath);
  log(`wrote ${path.relative(process.cwd(), verFile)}  (= ${version})`);
  log(`wrote ${path.relative(process.cwd(), pathFile)} (= ${platformPath})`);
}

function rmRecursive (p) {
  try { fs.rmSync(p, { recursive: true, force: true }); return true; }
  catch (e) { warn(`rm ${p} failed: ${e.message}`); return false; }
}

function extractZip (zipPath, distDir) {
  // Prefer the OS `unzip` (always present, fast, handles symlinks + xattrs
  // correctly on macOS). Fall back to `ditto` on macOS, then to the JS
  // `extract-zip` (which we know is flaky on Node 26 but is still better
  // than nothing).
  fs.mkdirSync(distDir, { recursive: true });
  const zipAbs = path.resolve(zipPath);
  const dirAbs = path.resolve(distDir);

  if (process.platform === 'darwin' || process.platform === 'mas') {
    // ditto preserves macOS metadata and code-signing; ideal for .app bundles
    const r = spawnSync('ditto', ['-x', '-k', '--sequesterRsrc', '--rsrc',
                                  zipAbs, dirAbs], { stdio: 'inherit' });
    if (r.status === 0) return true;
    warn(`ditto failed (status=${r.status}); falling back to unzip`);
  }
  const r = spawnSync('unzip', ['-q', '-o', zipAbs, '-d', dirAbs], { stdio: 'inherit' });
  if (r.status === 0) return true;
  warn(`unzip failed (status=${r.status})`);
  return false;
}


// After a successful electron install/repair, also make sure any native
// Node addons (better-sqlite3, etc.) are built against the *Electron*
// ABI, not the system Node ABI. The npm-install pipeline builds them
// for the system Node, so a fresh `npm install` always leaves the
// bindings mismatched — see scripts/rebuild-native.mjs for details.
function runNativeRebuild () {
  const rebuildScript = path.join(SCRIPT_DIR, 'rebuild-native.mjs');
  if (!fs.existsSync(rebuildScript)) {
    warn(`rebuild-native.mjs not found at ${rebuildScript}; skipping native rebuild`);
    return 0;
  }
  const r = spawnSync(process.execPath, [rebuildScript], { stdio: 'inherit' });
  if (r.status !== 0) {
    warn(`native rebuild failed (exit=${r.status}); electron itself is OK, but native modules may not load`);
    return r.status || 1;
  }
  return 0;
}

function main () {
  if (!fs.existsSync(ELECTRON_DIR)) {
    warn(`no electron package at ${ELECTRON_DIR} — run: cd electron && npm install`);
    return 1;
  }
  const electronPkg = readJSON(path.join(ELECTRON_DIR, 'package.json'));
  const platformPath = getPlatformPath();
  let { ok, reasons } = checkInstalled(electronPkg, platformPath);
  if (ok) { log(`OK — ${electronPkg.version} at dist/${platformPath}`); return runNativeRebuild(); }

  warn('electron install is partial:');
  for (const r of reasons) warn(`  - ${r}`);

  const exe = path.join(ELECTRON_DIR, 'dist', platformPath);
  const exeOk = isAppBundleComplete(exe);
  const onlyMarkersMissing = exeOk && reasons.every(
    (r) => r.includes('path.txt') || r.includes('dist/version')
  );

  // Level A: text markers only (full .app on disk) → fast path.
  if (onlyMarkersMissing) {
    log('repairing text markers in place (no re-download, no re-extract)…');
    writeMarkers(platformPath, electronPkg.version);
    log('repaired OK');
    return runNativeRebuild();
  }

  // Level B: .app is partial → wipe dist + extract the cached zip directly.
  const zip = findCachedZip(electronPkg.version);
  if (!zip) {
    warn(`no cached electron-v${electronPkg.version}-${process.platform}-${getArchString()}.zip found.`);
    warn('searched ~/Library/Caches/electron/ and ~/.electron/.');
    warn('recovery: cd electron && rm -rf node_modules/electron && npm install');
    return 1;
  }
  log(`found cached zip: ${zip}`);
  log(`re-extracting ${path.basename(zip)} → node_modules/electron/dist/`);
  rmRecursive(path.join(ELECTRON_DIR, 'dist'));
  rmRecursive(path.join(ELECTRON_DIR, 'path.txt'));
  if (!extractZip(zip, path.join(ELECTRON_DIR, 'dist'))) {
    warn('extraction failed; recovery: cd electron && rm -rf node_modules/electron && npm install');
    return 1;
  }
  writeMarkers(platformPath, electronPkg.version);
  const post = checkInstalled(electronPkg, platformPath);
  if (post.ok) { log('repaired OK'); return runNativeRebuild(); }
  warn('after re-extract, still:');
  for (const r of post.reasons) warn(`  - ${r}`);
  return 1;
}

process.exit(main());
