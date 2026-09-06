// Self-contained Python venv + project-tree bootstrapper for the packaged .app.
//
// Responsibilities:
//   1. Locate the bundled Python venv (resources/app-venv in packaged builds,
//      electron/app-venv in dev).
//   2. In packaged mode, copy the venv out of the read-only bundle into the
//      user-data dir and patch the absolute shebangs / egg-link paths so it
//      works from its new location.
//   3. Mirror the bundled project tree (atf/, bridge/, pyproject.toml) into
//      the user-data dir with the same depth as the dev tree, so Python's
//      path math (`parents[N]`) still resolves correctly.
//   4. Resolve the user's models directory through a chain of fallbacks:
//        a. Explicit user override (renderer settings panel / IPC)
//        b. $ATF_MODELS_DIR env var
//        c. <userData>/app/models symlink if the user already created one
//        d. /Volumes/SSD5/Ai/atf/models (project's shared root)
//        e. Empty (renderer will surface a "no models found" state)
//      The resolved location is materialized as <userData>/app/models — a
//      symlink to the real dir — so model_registry.MODELS_ROOT (which is
//      computed at import time) resolves through the symlink automatically.
//   5. Log every decision loudly so the user can see in Console.app / stderr
//      exactly where the app is loading models from.
//
// Idempotent: subsequent launches reuse the cached copy unless the bundled
// venv's signature changes.

const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");

function log(...a) {
  // stderr → visible in Console.app for the packaged app, in the terminal
  // when run from source.
  console.error("[app-venv]", ...a);
}

// ── venv discovery ───────────────────────────────────────────────────

// Returns the absolute path to the bundled venv (or null if there is none).
function bundledVenvPath() {
  if (process.resourcesPath && fs.existsSync(path.join(process.resourcesPath, "app-venv", "bin", "python3"))) {
    return path.join(process.resourcesPath, "app-venv");
  }
  // Dev tree fallbacks. The first hit wins.
  const candidates = [
    path.join(__dirname, "app-venv"),                              // electron/app-venv
    path.join(__dirname, ".venv"),                                 // electron/.venv
    path.join(__dirname, "..", ".venv"),                           // beta/v5/.venv
    path.join(__dirname, "..", "..", "..", ".venv"),               // /atf/.venv
  ];
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, "bin", "python3"))) return c;
  }
  return null;
}

function readVersionFile(p) {
  try { return fs.readFileSync(p, "utf8").trim(); } catch { return null; }
}

function signatureOf(venvPath) {
  // Stable fingerprint — content hash of pyvenv.cfg + mtimes of python/pip.
  // Detects "developer bumped the venv" without hashing hundreds of MB.
  const h = crypto.createHash("sha256");
  const marker = path.join(venvPath, "pyvenv.cfg");
  if (fs.existsSync(marker)) h.update(fs.readFileSync(marker));
  for (const f of ["bin/python3", "bin/pip"]) {
    try {
      const st = fs.statSync(path.join(venvPath, f));
      h.update(String(st.mtimeMs | 0));
    } catch {}
  }
  return h.digest("hex").slice(0, 16);
}

function patchWrapper(wrapperPath, newPython) {
  const fd = fs.openSync(wrapperPath, "r");
  const buf = Buffer.alloc(512);
  const n = fs.readSync(fd, buf, 0, 512, 0);
  fs.closeSync(fd);
  const head = buf.slice(0, n).toString("utf8");
  const m = head.match(/^#!(\S+)\s*\n/);
  if (!m) return false;
  const target = m[1];
  if (!/python(\d+(\.\d+)*)?$/.test(target)) return false;
  const newShebang = `#!/usr/bin/env -S ${newPython}\n`;
  if (target === newPython) return false;
  const fd2 = fs.openSync(wrapperPath, "r+");
  try {
    fs.writeSync(fd2, newShebang, 0, "utf8");
  } finally {
    fs.closeSync(fd2);
  }
  return true;
}

function rewriteTextFiles(venvPath, replacements) {
  const skipDirs = new Set(["__pycache__", "include", "lib", "bin"]);
  const skipExt = new Set([".pyc", ".so", ".dylib", ".dylib.1", ".node", ".bin"]);
  const walk = (dir) => {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      if (skipDirs.has(ent.name)) continue;
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) { walk(p); continue; }
      if (skipExt.has(path.extname(ent.name))) continue;
      try {
        const st = fs.statSync(p);
        if (st.size === 0 || st.size > 1_000_000) continue;
        const buf = fs.readFileSync(p);
        if (buf.indexOf(0) !== -1) continue; // binary
        let text = buf.toString("utf8");
        let changed = false;
        for (const [from, to] of replacements) {
          if (text.includes(from)) { text = text.split(from).join(to); changed = true; }
        }
        if (changed) fs.writeFileSync(p, text);
      } catch {}
    }
  };
  walk(venvPath);
}

function copyVenv(src, dst) {
  const { spawnSync } = require("child_process");
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  try { fs.rmSync(dst, { recursive: true, force: true }); } catch {}
  const res = spawnSync("/bin/cp", ["-a", src + "/.", dst + "/"], { stdio: "inherit" });
  if (res.status !== 0) throw new Error(`cp -a ${src} ${dst} failed (status ${res.status})`);
}

function needsRefresh(srcVenv, dstVenv) {
  if (!fs.existsSync(dstVenv)) return true;
  const stamp = path.join(dstVenv, ".atf-bundled-signature");
  const srcSig = signatureOf(srcVenv);
  const dstSig = readVersionFile(stamp) || "";
  if (dstSig !== srcSig) return true;
  if (!fs.existsSync(path.join(dstVenv, "bin", "python3"))) return true;
  return false;
}

// ── models directory resolution ──────────────────────────────────────

// Read the user's override, preferring: explicit param → env → persistent file.
function readUserModelsDir(userDataDir) {
  if (process.env.ATF_MODELS_DIR) return process.env.ATF_MODELS_DIR;
  const settingsFile = path.join(userDataDir, "settings.json");
  try {
    const j = JSON.parse(fs.readFileSync(settingsFile, "utf8"));
    if (j && typeof j.modelsDir === "string" && j.modelsDir.trim()) {
      return j.modelsDir.trim();
    }
  } catch {}
  return null;
}

function writeUserModelsDir(userDataDir, modelsDir) {
  const settingsFile = path.join(userDataDir, "settings.json");
  let j = {};
  try { j = JSON.parse(fs.readFileSync(settingsFile, "utf8")); } catch {}
  j.modelsDir = modelsDir;
  fs.writeFileSync(settingsFile, JSON.stringify(j, null, 2));
}

// Walk a candidate list, return the first path that exists AND looks like
// a models directory (contains at least one *.atf file, recursively, OR is
// an empty directory the user is about to populate).
function firstExistingDir(candidates) {
  for (const c of candidates) {
    if (!c) continue;
    try {
      const st = fs.lstatSync(c);
      if (st.isDirectory() || st.isSymbolicLink()) return c;
    } catch {}
  }
  return null;
}

// Create/replace the <userData>/app/models symlink so model_registry's
// MODELS_ROOT (computed at import time from atf/model_registry.py's
// parents[3]) resolves through it to wherever the user's models actually live.
function ensureModelsSymlink(userDataDir, modelsDir) {
  const linkPath = path.join(userDataDir, "app", "models");
  // Remove whatever the mirror put there (probably an empty directory).
  try {
    const st = fs.lstatSync(linkPath);
    if (st.isSymbolicLink()) fs.unlinkSync(linkPath);
    else if (st.isDirectory()) fs.rmSync(linkPath, { recursive: true, force: true });
  } catch {}
  try {
    fs.symlinkSync(modelsDir, linkPath);
    log(`models symlink: ${linkPath} -> ${modelsDir}`);
    return true;
  } catch (e) {
    log(`models symlink FAILED: ${e.message}`);
    return false;
  }
}

// Detect dev mode: bundled venv lives inside the project tree (not packaged).
function isDevMode() {
  const src = bundledVenvPath();
  if (!src) return false;
  const devCandidates = [
    path.join(__dirname, "app-venv"),
    path.join(__dirname, ".venv"),
    path.join(__dirname, "..", ".venv"),
    path.join(__dirname, "..", "..", "..", ".venv"),
  ];
  return devCandidates.some(
    (c) =>
      fs.existsSync(path.join(c, "bin", "python3")) &&
      path.resolve(c) === path.resolve(src)
  );
}

const IS_DEV = isDevMode();
const DEV_MODELS_DIR = "/Volumes/SSD5/Ai/atf/models/v4";

// Resolve where models should live, in priority order, and materialize
// <userData>/app/models as a symlink so Python's path-based lookup works.
//
// The user's actual models directory lives at the top level of the user-data
// dir (~/Library/Application Support/atf-chat/models), not nested inside the
// mirrored project tree. The symlink at <userData>/app/models is purely a
// bridge for the Python code path that hardcodes parents[3]/models.
function resolveModelsDir(userDataDir) {
  const userModelsDir = path.join(userDataDir, "models");
  // 1. Explicit override (settings.json or $ATF_MODELS_DIR) — always wins.
  const override = readUserModelsDir(userDataDir);
  if (override) {
    if (!fs.existsSync(override)) {
      try { fs.mkdirSync(override, { recursive: true }); } catch {}
    }
    if (fs.existsSync(override)) {
      ensureModelsSymlink(userDataDir, override);
      log("============================================================");
      log(`[models] LOADING MODELS FROM: ${override}`);
      log(`           (custom path from settings / ATF_MODELS_DIR)`);
      log("============================================================");
      return override;
    }
  }
  // 2. Dev mode: load from the shared SSD models tree (v4) so the dev build
  //    and the production app see the same model files.
  if (IS_DEV && fs.existsSync(DEV_MODELS_DIR)) {
    ensureModelsSymlink(userDataDir, DEV_MODELS_DIR);
    log("============================================================");
    log(`[models] LOADING MODELS FROM: ${DEV_MODELS_DIR}`);
    log(`           (dev mode — shared models tree on SSD)`);
    log("============================================================");
    return DEV_MODELS_DIR;
  }
  // 3. Production: the user-data dir is the canonical default — create it if missing so
  //    the user always has a place to drop .atf files.
  try {
    fs.mkdirSync(userModelsDir, { recursive: true });
  } catch (e) {
    log("[models] could not create default dir:", e.message);
  }
  if (fs.existsSync(userModelsDir)) {
    ensureModelsSymlink(userDataDir, userModelsDir);
    log("============================================================");
    log(`[models] LOADING MODELS FROM: ${userModelsDir}`);
    log(`           (default — drop .atf files here, or change in Settings)`);
    log("============================================================");
    return userModelsDir;
  }
  // 4. Last-resort fallbacks if we can't even create the default dir.
  const candidates = [
    "/Volumes/SSD5/Ai/atf/models",
    path.join(process.resourcesPath || "", "models"),
    path.join(os.homedir(), "models"),
  ];
  const picked = firstExistingDir(candidates);
  if (picked) {
    ensureModelsSymlink(userDataDir, picked);
    log("============================================================");
    log(`[models] LOADING MODELS FROM: ${picked} (fallback)`);
    log("============================================================");
    return picked;
  }
  log("[models] WARNING: no usable models directory; in-app picker will be empty");
  return null;
}

/**
 * Returns { python, venv, sitePackages, source, projectMirror, modelsDir }
 * for the Python interpreter the app should use. Idempotent.
 *
 * `userDataDir` is Electron's app.getPath('userData') — a writable per-app
 * dir under ~/Library/Application Support/<productName>/.
 */
function resolveAppVenv(userDataDir) {
  const src = bundledVenvPath();
  if (!src) return null;

  const isPackaged = !!(process.resourcesPath && src.startsWith(process.resourcesPath));

  if (!isPackaged) {
    const py = path.join(src, "bin", "python3");
    // Dev mode resolves the models dir too (and materializes the
    // <userData>/app/models symlink) so the bridge can be told the real
    // path instead of falling back to the project tree's models/ folder.
    const modelsDir = resolveModelsDir(userDataDir);
    return {
      python: py,
      venv: src,
      sitePackages: path.join(src, "lib", "python3.13", "site-packages"),
      source: "dev-tree",
      modelsDir,
    };
  }

  // Packaged: copy venv + mirror project tree into the user-data dir.
  const dst = path.join(userDataDir, "venv");
  if (needsRefresh(src, dst)) {
    log("=== venv bootstrap ===");
    log("copying bundled venv into user data dir:", dst);
    copyVenv(src, dst);

    log("patching bin wrapper shebangs");
    const newPy = path.join(dst, "bin", "python3");
    const binDir = path.join(dst, "bin");
    let patched = 0;
    for (const ent of fs.readdirSync(binDir)) {
      const p = path.join(binDir, ent);
      try {
        const st = fs.lstatSync(p);
        if (!st.isFile() && !st.isSymbolicLink()) continue;
        if (patchWrapper(p, newPy)) patched++;
      } catch {}
    }
    log("patched", patched, "bin wrappers");

    // Mirror the project tree. Layout matches the dev tree (3 levels deep)
    // so model_registry.MODELS_ROOT and the bridge's relative lookups both
    // resolve correctly: atf/model_registry.py's parents[3] = <userData>/app.
    const projectSrc = path.join(process.resourcesPath, "app");
    const projectDst = path.join(userDataDir, "app");
    if (fs.existsSync(projectSrc)) {
      try {
        fs.mkdirSync(projectDst, { recursive: true });
        const { spawnSync } = require("child_process");
        let res = spawnSync("/usr/bin/rsync", [
          "-a", "--delete",
          "--exclude", "models",     // we replace this with a symlink below
          projectSrc + "/", projectDst + "/",
        ], { stdio: "inherit" });
        if (res.status !== 0) {
          log("rsync failed, falling back to cp -a");
          res = spawnSync("/bin/cp", ["-a", projectSrc + "/.", projectDst + "/"], { stdio: "inherit" });
        }
        rewriteTextFiles(dst, [[projectSrc, projectDst]]);
        log("mirrored project tree to", projectDst);
      } catch (e) {
        log("project mirror failed (non-fatal):", e.message);
      }
    }

    resolveModelsDir(userDataDir);

    fs.writeFileSync(path.join(dst, ".atf-bundled-signature"), signatureOf(src));
  } else {
    log("venv cache up-to-date:", dst);
    // Models symlink may have been removed (user fiddled, etc.) — re-resolve.
    resolveModelsDir(userDataDir);
  }

  const py = path.join(dst, "bin", "python3");
  const modelsLink = path.join(userDataDir, "app", "models");
  let modelsDir = null;
  try {
    if (fs.existsSync(modelsLink)) modelsDir = fs.realpathSync(modelsLink);
  } catch {}

  return {
    python: py,
    venv: dst,
    sitePackages: path.join(dst, "lib", "python3.13", "site-packages"),
    source: "user-data",
    projectMirror: path.join(userDataDir, "app"),
    modelsDir,
  };
}

// Public helpers for the IPC layer.

function getModelsDir(userDataDir) {
  const link = path.join(userDataDir, "app", "models");
  try {
    if (fs.existsSync(link)) return fs.realpathSync(link);
  } catch {}
  return null;
}

function setModelsDir(userDataDir, newDir) {
  if (!newDir || typeof newDir !== "string") {
    throw new Error("modelsDir must be a non-empty path");
  }
  // Expand ~ and resolve to absolute.
  const abs = path.resolve(newDir.replace(/^~/, os.homedir()));
  if (!fs.existsSync(abs)) {
    fs.mkdirSync(abs, { recursive: true });
    log(`created models directory: ${abs}`);
  }
  writeUserModelsDir(userDataDir, abs);
  ensureModelsSymlink(userDataDir, abs);
  log("============================================================");
  log(`[models] USER UPDATED MODELS DIR: ${abs}`);
  log("============================================================");
  return abs;
}

module.exports = {
  resolveAppVenv,
  bundledVenvPath,
  resolveModelsDir,
  getModelsDir,
  setModelsDir,
};
