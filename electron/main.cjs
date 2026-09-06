const { app, BrowserWindow, ipcMain, Menu, shell, dialog } = require("electron");
const { spawn } = require("child_process");
const net = require("net");
const http = require("http");
const https = require("https");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { resolveAppVenv } = require("./app_venv.cjs");

let win = null;
let bridge = null;
let sock = null;
// Resolved at app-startup by bootstrapAppVenv(). Always live in the user-data
// dir: the .app bundle is read-only, the source tree doesn't exist when
// packaged, and the bridge needs a writable temp dir for the unix socket.
let SOCK_PATH = null;
let APP_VENV = null;            // { python, venv, sitePackages, source, projectMirror? }
let MODELS_DIR = null;          // resolved models dir, passed to the bridge dynamically
let PROJECT_ROOT = null;        // project tree we run the bridge from (editable atf install)
let ready = false;
let currentModelId = null;   // no model loads until the user picks one
let registry = [];               // models discovered by the bridge (with loaded flag)
let registryListeners = [];      // resolve()s fired whenever the registry updates
let uiLoaded = false;
let eventLog = [];

// ── crash-restart with exponential backoff ─────────────────────────
let restartingBridge = false;
let restartDelay = 1000;
let bridgeUpSince = 0;
function scheduleRestart(code, signal) {
  sock = null;
  ready = false;
  if (restartingBridge) return;
  // if the bridge ran stably for a minute, reset the backoff ladder
  const stableFor = Date.now() - bridgeUpSince;
  if (bridgeUpSince && stableFor > 60_000) restartDelay = 1000;
  console.log(`[bridge] exited (code ${code}, signal ${signal}) -- restart in ${restartDelay}ms`);
  toUI("bridge-crashed", { code, signal, retryMs: restartDelay });
  restartingBridge = true;
  setTimeout(() => {
    restartingBridge = false;
    startBridge();
  }, restartDelay);
  restartDelay = Math.min(restartDelay * 2, 15_000);
}

function toUI(channel, payload) {
  if (uiLoaded) {
    win?.webContents.send(channel, payload);
  } else {
    eventLog.push([channel, payload]);
  }
}

function flushEventLog() {
  uiLoaded = true;
  eventLog.forEach(([ch, p]) => win?.webContents.send(ch, p));
  eventLog = [];
}

// Decide where the bridge lives, where the unix socket goes, and which Python
// interpreter to spawn. Runs once before any window is shown so the renderer
// never has to handle "venv still extracting" states.
function bootstrapAppVenv() {
  // Bridge script location. In packaged builds the project tree is mirrored
  // to <userData>/app/beta/v6/ (matching dev-tree depth) so Python's path
  // math still resolves. The .app bundle copy lives under extraResources →
  // app/beta/v6/bridge/atf_bridge.py.
  const bridgeCandidates = [
    process.resourcesPath && path.join(process.resourcesPath, "app", "beta", "v6", "bridge", "atf_bridge.py"),
    path.join(app.getPath("userData"), "app", "beta", "v6", "bridge", "atf_bridge.py"),
    path.join(__dirname, "..", "bridge", "atf_bridge.py"), // dev (beta/v6)
  ];
  let bridgeScript = null;
  for (const c of bridgeCandidates) if (c && fs.existsSync(c)) { bridgeScript = c; break; }
  if (!bridgeScript) throw new Error("Could not locate bridge/atf_bridge.py");

  // PROJECT_ROOT must match the dev tree layout (beta/v6/) so that
  // model_registry.MODELS_ROOT (parents[3]/models) and the bridge's
  // ROOT/ROOT.parent/ROOT.parent.parent walks both resolve to <mirror>/models.
  APP_VENV = resolveAppVenv(app.getPath("userData"));
  MODELS_DIR = APP_VENV?.modelsDir || null;
  console.log(`[bootstrap] modelsDir=${MODELS_DIR || "(unresolved)"}`);
  if (APP_VENV?.source === "user-data") {
    PROJECT_ROOT = path.join(APP_VENV.projectMirror, "beta", "v6");
  } else if (APP_VENV?.projectMirror) {
    PROJECT_ROOT = APP_VENV.projectMirror;
  } else {
    PROJECT_ROOT = path.join(__dirname, ".."); // beta/v6 in dev
  }

  // Ensure a writable tmp dir for the unix socket in both dev and packaged
  // modes. In dev we keep the historical <root>/tmp location; in packaged
  // mode we use <userData>/tmp so we never try to write into Resources/.
  const tmpBase = APP_VENV?.source === "user-data"
    ? path.join(app.getPath("userData"), "tmp")
    : path.join(PROJECT_ROOT, "tmp");
  fs.mkdirSync(tmpBase, { recursive: true });
  SOCK_PATH = path.join(tmpBase, "atf_bridge.sock");

  if (!APP_VENV) {
    // Fallback to system python3 if we somehow shipped without a venv.
    APP_VENV = { python: "python3", venv: null, sitePackages: null, source: "system" };
  }
  console.log(`[bootstrap] bridge=${bridgeScript}`);
  console.log(`[bootstrap] python=${APP_VENV.python} (${APP_VENV.source})`);
  console.log(`[bootstrap] projectRoot=${PROJECT_ROOT}`);
  console.log(`[bootstrap] socket=${SOCK_PATH}`);

  return { bridgeScript };
}

function resolvePython() {
  // After bootstrapAppVenv() has run, APP_VENV.python points at the right
  // interpreter (dev-tree venv or relocated packaged venv). This function is
  // safe to call from any handler once the bootstrap has finished; if it
  // somehow hasn't, we fall back to the system python3.
  return APP_VENV?.python || "python3";
}

function projectRoot() {
  return PROJECT_ROOT || path.join(__dirname, "..");
}

function pythonEnv() {
  const pyPath = projectRoot();
  const extra = APP_VENV?.sitePackages
    ? `${pyPath}:${APP_VENV.sitePackages}`
    : pyPath;
  return {
    ...process.env,
    PYTHONPATH: extra + (process.env.PYTHONPATH ? ":" + process.env.PYTHONPATH : "")
  };
}

function startBridge() {
  if (!SOCK_PATH) bootstrapAppVenv();
  const script = path.join(PROJECT_ROOT, "bridge", "atf_bridge.py");
  const py = resolvePython();
  const env = {
    ...process.env,
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    ATF_BRIDGE_SOCK: SOCK_PATH,
    // GDN_SCAN default: OFF. The chunked-parallel scan (docs/v8_gdn_scan_validation.py)
    // was only validated as an isolated function against a reference
    // implementation -- it was NOT validated through this actual bridge
    // integration (head-pairing/dtype/state-carry differences between
    // prefill chunks and the live gdn_states dict are untested here).
    // Turning it on produced garbled/incoherent output in practice
    // (mixed broken tokens, non-target-language noise) -- see
    // LOOP_ANALYSIS.md and /Users/amgad/Desktop/Ai/Claude/atf-loop-hang-v5/notes.md.
    // Do NOT flip this default without first fixing/re-validating the scan
    // path end-to-end (see "Next" in that notes.md).
    ATF_GDN_SCAN: process.env.ATF_GDN_SCAN || "0",
    // v22: MoE per-expert dequant now slices one expert instead of the
    // whole block record, so the LRU eviction loop is far less aggressive.
    // Bump to 24 GB to also cover any residual thrash during first-pass
    // warmup before the router stabilises on a consistent expert set.
    // The Electron app is the only process that spawns the bridge, so this
    // env var reaches the right place without needing a shell export.
    ATF_CACHE_GB: process.env.ATF_CACHE_GB || "24",
    // gamma/v5 hotfix: default stage-level progress logging ON (prefill/
    // decode/MoE totals -- NOT the noisy per-token/per-block trace, that's
    // level 3+). Every prior "is it hung or just quiet?" investigation this
    // session traced back to this being unset by default and the stdout-
    // drop bug above hiding console.print() too; fixing both here so a
    // future stall is visible without the user needing to relaunch with
    // ATF_VERBOSE=3 by hand first. Override with ATF_VERBOSE=0 to silence,
    // or higher for more detail.
    ATF_VERBOSE: process.env.ATF_VERBOSE || "1",
    // gamma/v7: half the default prefill chunk size (1024 -> 512) to reduce
    // the chance of a macOS GPU watchdog timeout during long-context prefill.
    // A smaller chunk = shorter Metal command buffer = more likely to complete
    // within the ~30s watchdog window. Override with ATF_PREFILL_CHUNK env.
    ATF_PREFILL_CHUNK: process.env.ATF_PREFILL_CHUNK || "512",
    PYTHONPATH: (APP_VENV?.sitePackages ? APP_VENV.sitePackages + ":" : "") + PROJECT_ROOT +
      (process.env.PYTHONPATH ? ":" + process.env.PYTHONPATH : ""),
  };
  // Dynamic models dir: the bridge scans wherever Electron resolved, not the
  // compile-time parents[3]/models guess. Covers bridge restarts too.
  if (MODELS_DIR) env.ATF_MODELS_DIR = MODELS_DIR;
  // v19: Paged SSD KV cache settings from the Settings panel, surfaced as
  // env vars. Python (atf/engine.py) already reads ATF_KV_SSD /
  // ATF_KV_SSD_PATH / ATF_KV_SSD_HOT_PAGES via os.environ on first use of
  // GenConfig -- no Python change required. We re-resolve every spawn so
  // mid-session changes in the Settings panel take effect on the next
  // bridge start (which the app already triggers on model load / reload).
  const _kvssdSettings = readKvSsdSettings();
  if (_kvssdSettings.enabled === false) {
    env.ATF_KV_SSD = "0";
  } else {
    env.ATF_KV_SSD = "1";
  }
  // gamma/v7 persistence: OFF by default, no Settings-panel toggle yet --
  // opt in for testing with `ATF_KV_SSD_PERSIST=1 ./run.sh` (or export it
  // before launching the packaged app). See the atf-gamma-v7-ssd-cache-ui
  // notes for why this isn't defaulted on yet (needs an on-device
  // coherence check first -- a reuse bug here is garbled output, not a
  // crash, so it shouldn't ship silently-on).
  env.ATF_KV_SSD_PERSIST = process.env.ATF_KV_SSD_PERSIST || "0";
  if (_kvssdSettings.path) {
    env.ATF_KV_SSD_PATH = _kvssdSettings.path;
  }
  if (_kvssdSettings.hotPages) {
    env.ATF_KV_SSD_HOT_PAGES = String(_kvssdSettings.hotPages);
  }
  if (APP_VENV?.venv) {
    // mlx, huggingface_hub, and others look at VIRTUAL_ENV to locate the
    // python executable; set it so subprocesses (e.g. tokenizer downloads)
    // resolve into our relocated copy.
    env.VIRTUAL_ENV = APP_VENV.venv;
  }
  bridge = spawn(py, ["-u", script], { cwd: PROJECT_ROOT, env });
  bridgeUpSince = Date.now();

  bridge.on("exit", (code, signal) => scheduleRestart(code, signal));

  bridge.stderr.on("data", (d) => {
    const s = d.toString();
    process.stderr.write(s);
    // Mirror to the in-app API log so Python tracebacks are visible
    // without opening a terminal.
    for (const line of s.split(/\r?\n/)) {
      if (!line) continue;
      const tag = line.startsWith("[bridge-trace]") ? "[trace]" : "[bridge]";
      apiAppend(`${tag} ${line}\n`);
    }
  });

  // ONE stdout parser. The registry is emitted BEFORE the socket exists,
  // so this channel must stay live for the whole process lifetime.
  let socketSeen = false;
  let sbuf = "";
  bridge.stdout.on("data", (d) => {
    if (d.toString().includes("socket-listening") && !socketSeen) {
      socketSeen = true;
      connectSocket();
    }
    sbuf += d.toString();
    let idx;
    while ((idx = sbuf.indexOf("\n")) !== -1) {
      const line = sbuf.slice(0, idx).trim();
      sbuf = sbuf.slice(idx + 1);
      if (!line) continue;
      let msg;
      try { msg = JSON.parse(line); } catch {
        // Not a JSON status event -- this is engine.py's rich Console
        // status output (the "[step] ..." prefill/load progress lines),
        // which was previously silently dropped here, leaving the app
        // showing nothing while the engine was actively working. Mirror
        // it the same way stderr lines are mirrored, so a stuck-looking
        // load/prefill and a genuinely silent one are distinguishable
        // without needing ATF_VERBOSE set by hand every time.
        console.log("[bridge-stdout-raw]", line);
        apiAppend(`[engine] ${line}\n`);
        continue;
      }
      if (process.env.ATF_DEBUG && msg.type !== "progress") {
        console.log("[bridge-stdout]", msg.type, (msg.text || "").slice(0, 60));
      }
      handleEngineEvent(msg);
    }
  });
}

// ── single event router (used by BOTH channels) ────────────────────
// stdout carries: models / model-loaded / ready / status / progress /
// error (engine-level). The socket carries per-request streams: token,
// think, tier, usage, done + everything above for the requesting client.
function handleEngineEvent(msg) {
  switch (msg.type) {
    case "error": toUI("error", msg); break;
    case "status": toUI("status", msg.text); break;
    case "progress": toUI("progress", msg); break;
    case "models":
      registry = (msg.items || []).map((m) => ({
        ...m, loaded: !!(currentModelId && m.id === currentModelId),
      }));
      toUI("bridge-models", registry);
      // wake any awaiter of the next registry refresh (e.g. /v1/models handler)
      const waiters = registryListeners; registryListeners = [];
      for (const r of waiters) r(registry);
      break;
    case "model-loaded":
      currentModelId = msg.id;
      for (const m of registry) m.loaded = (m.id === currentModelId);
      toUI("model-loaded", msg);
      break;
    case "ready":
      ready = true;
      toUI("bridge-ready");
      break;
    default: break;   // socket-only types are handled there
  }
}

function connectSocket() {
  if (!SOCK_PATH) bootstrapAppVenv();
  sock = net.createConnection(SOCK_PATH);
  let sbuf = "";
  sock.on("connect", () => {
    console.log("[socket] connected");
    ready = true;
    bridgeUpSince = Date.now();
    restartDelay = 1000;
    toUI("bridge-ready");
    toUI("status", "engine connected");
    flushPending();
  });
  sock.on("data", (d) => {
    sbuf += d.toString();
    let idx;
    while ((idx = sbuf.indexOf("\n")) !== -1) {
      const line = sbuf.slice(0, idx).trim();
      sbuf = sbuf.slice(idx + 1);
      if (!line) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }
      if (process.env.ATF_DEBUG && msg.type !== "token") {
        console.log("[bridge]", msg.type, (msg.text || "").slice(0, 60));
      }
      switch (msg.type) {
        case "token": toUI("token", msg); break;
        case "tier": toUI("tier", msg); break;
        case "think": toUI("think", msg.open); break;
        case "usage": toUI("usage", msg); break;
        case "progress": toUI("progress", msg); break;
        case "done": toUI("done", msg); break;
        case "mem": toUI("mem", msg.text); break;
        case "unloaded":
          currentModelId = null;
          for (const m of registry) m.loaded = false;
          toUI("model-unloaded");
          break;
        case "error": toUI("error", msg); break;
        default: handleEngineEvent(msg);
      }
    }
  });
  sock.on("error", (e) => {
    console.log("[socket] error:", e.message);
    toUI("status", `socket error: ${e.message}`);
  });
}

const pendingQueue = [];
function flushPending() {
  while (pendingQueue.length && sock && !sock.destroyed && ready) {
    sendNow(pendingQueue.shift());
  }
}

function sendNow(obj) {
  sock.write(JSON.stringify(obj) + "\n");
}

function bridgeRequest(obj) {
  if (obj.type === "generate") obj.id ??= `r${Date.now()}${Math.random().toString(36).slice(2, 7)}`;
  if (sock && !sock.destroyed && (ready || obj.type !== "generate")) {
    sendNow(obj);
    return obj.id || true;
  }
  pendingQueue.push(obj);
  toUI("status", "queued (waiting for engine)…");
  return obj.id || false;
}

// ── IPC: engine control ─────────────────────────────────────────────
ipcMain.handle("list-models", () => bridgeWrite({ type: "list_models" }));

ipcMain.handle("load-model", (_e, id) => {
  console.log("[load-model]", id);
  ready = false;
  return bridgeRequest({ type: "load", model: id });
});

ipcMain.handle("unload-model", async () => {
  return new Promise((resolve) => {
    if (!sock || !ready) { resolve({ ok: true }); return; }
    bridgeRequest({ type: "unload" });
    // the socket router already broadcasts "model-unloaded"; resolve fast
    const t0 = Date.now();
    const iv = setInterval(() => {
      if (currentModelId === null || Date.now() - t0 > 30_000) {
        clearInterval(iv);
        resolve({ ok: currentModelId === null });
      }
    }, 200);
  });
});

ipcMain.handle("delete-model", async (_e, modelId) => {
  if (!modelId) {
    return { success: false, error: "No model ID provided" };
  }

  try {
    // If the model to delete is currently loaded, unload it first
    if (currentModelId === modelId) {
      await new Promise((resolve) => {
        bridgeRequest({ type: "unload" });
        const t0 = Date.now();
        const iv = setInterval(() => {
          if (currentModelId === null || Date.now() - t0 > 30_000) {
            clearInterval(iv);
            resolve({ ok: currentModelId === null });
          }
        }, 200);
      });
    }

    // Get the models directory - ask the bridge for registry which includes paths
    const model = registry.find((m) => m.id === modelId);
    if (!model || !model.path) {
      return { success: false, error: "Model not found or path unknown" };
    }

    // Delete the model file/directory
    if (fs.existsSync(model.path)) {
      fs.rmSync(model.path, { recursive: true, force: true });
      console.log(`[delete-model] Deleted ${modelId} at ${model.path}`);

      // Refresh the model list
      bridgeWrite({ type: "list_models" });

      return { success: true };
    } else {
      return { success: false, error: "Model file/directory not found" };
    }
  } catch (err) {
    console.error("[delete-model] Error:", err);
    return { success: false, error: err.message };
  }
});

ipcMain.handle("generate", (_e, req) => {
  if (currentModelId) req.model = currentModelId;
  return bridgeRequest({ type: "generate", ...req });
});

ipcMain.handle("stop", () => {
  if (sock && !sock.destroyed) sendNow({ type: "stop" });
});

function bridgeWrite(obj) {
  if (sock && !sock.destroyed) {
    sendNow(obj);
    return true;
  }
  toUI("status", "bridge not connected");
  return false;
}

// full engine reset: kill + relaunch the python bridge process
ipcMain.handle("reload-engine", () => {
  console.log("[reload] restarting bridge");
  ready = false;
  uiLoaded = true;
  eventLog = [];
  if (sock) { try { sock.destroy(); } catch { } sock = null; }
  if (bridge) {
    restartingBridge = true;             // suppress the crash handler
    bridge.removeAllListeners("exit");
    bridge.kill();
    setTimeout(() => { try { bridge.kill("SIGKILL"); } catch { } }, 3000);
  }
  toUI("reloading");
  setTimeout(startBridge, 500);
});

// ── GGUF → ATF conversion (Convert tab) ────────────────────────────
// Runs `python -m atf.cli convert <gguf> -o <out> [opts]` as a child process
// and streams its stdout/stderr to the renderer line by line.
let convertProc = null;

ipcMain.handle("convert-pick-gguf", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Select a GGUF model",
    filters: [{ name: "GGUF models", extensions: ["gguf"] },
    { name: "All files", extensions: ["*"] }],
    properties: ["openFile"],
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle("convert-pick-mlx", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Select an MLX model folder (config.json + safetensors)",
    properties: ["openDirectory"],
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle("convert-pick-output", async (_e, suggested) => {
  const r = await dialog.showSaveDialog(win, {
    title: "Save ATF model as",
    defaultPath: suggested || undefined,
    filters: [{ name: "ATF model", extensions: ["atf"] }],
  });
  return r.canceled ? null : r.filePath;
});

ipcMain.handle("convert-start", (_e, req) => {
  if (convertProc) return { ok: false, error: "A conversion is already running." };
  const { gguf, output, experts, topK, expertStorage, denseFp4, raw } = req || {};
  if (!gguf || !fs.existsSync(gguf)) return { ok: false, error: "Select a valid .gguf file first." };
  if (!output || !String(output).trim()) return { ok: false, error: "Choose an output .atf path first." };
  if (fs.existsSync(output)) return { ok: false, error: `Output already exists: ${output} — pick another name.` };

  let args;
  if (req.source === "mlx") {
    // MLX safetensors folder -> dense merged-FFN ATF (convert_mlx path)
    args = ["-m", "atf.cli", "convert-mlx", String(gguf), "-o", String(output),
      "--bits", String(parseInt(req.targetBits, 10) || 4)];
  } else {
    args = ["-m", "atf.cli", "convert", String(gguf), "-o", String(output),
      "--experts", String(parseInt(experts, 10) || 8),
      "--top-k", String(parseInt(topK, 10) || 4),
      "--expert-storage", expertStorage || "auto"];
    if (denseFp4) args.push("--dense-fp4");
    if (raw) args.push("--raw");
  }

  console.log("[convert]", args.join(" "));
  convertProc = spawn(resolvePython(), args, { cwd: projectRoot(), env: pythonEnv() });
  toUI("convert-started", { gguf, output });

  const feed = (d) => {
    for (const line of d.toString().split(/\r?\n/)) {
      if (line.trim()) toUI("convert-log", line);
    }
  };
  convertProc.stdout.on("data", feed);
  convertProc.stderr.on("data", feed);
  convertProc.on("error", (err) => {
    toUI("convert-log", `[error] ${err.message}`);
    convertProc = null;
    toUI("convert-done", { code: -1, cancelled: false });
  });
  convertProc.on("close", (code, signal) => {
    const cancelled = signal === "SIGTERM";
    convertProc = null;
    toUI("convert-done", { code, cancelled });
  });
  return { ok: true };
});

ipcMain.handle("convert-cancel", () => {
  if (!convertProc) return { ok: false };
  convertProc.kill();
  setTimeout(() => { try { convertProc?.kill("SIGKILL"); } catch { } }, 3000);
  return { ok: true };
});

// ── Hugging Face model search / download (Models tab) ──────────────
// Lets the Models tab download .atf models from
// https://huggingface.co/amgadtewfik/atf/tree/main/models straight into
// the local models folder. Runs entirely in the main process (Node's
// https) since the renderer's CSP is default-src 'self' and cannot reach
// the network itself.
const HF_API = "https://huggingface.co";
const HF_REPO = "amgadtewfik/atf";
const HF_REPO_DIR = "models";

function hfEncodePath(p) {
  return String(p).split("/").map(encodeURIComponent).join("/");
}

function httpsJson(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { "User-Agent": "ATF-Chat/1.0", Accept: "application/json" } }, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
        res.resume();
        if (redirects >= 8) return reject(new Error("too many redirects"));
        return resolve(httpsJson(res.headers.location, redirects + 1));
      }
      let body = "";
      res.on("data", (d) => (body += d));
      res.on("end", () => {
        if (res.statusCode !== 200) {
          return reject(new Error(`HF API ${res.statusCode}: ${body.slice(0, 200)}`));
        }
        try { resolve(JSON.parse(body)); } catch (e) { reject(e); }
      });
    });
    req.on("error", reject);
  });
}

// Streams a file from Hugging Face to disk, reporting (received, total)
// bytes via onProgress. Writes to a .part file and renames on success so a
// cancelled/failed download never leaves a half-written model file behind.
function hfDownloadFile(url, destPath, { onProgress, signal, redirects = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: { "User-Agent": "ATF-Chat/1.0" },
      signal,
    }, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
        res.resume();
        if (redirects >= 8) return reject(new Error("too many redirects"));
        return resolve(hfDownloadFile(res.headers.location, destPath, { onProgress, signal, redirects: redirects + 1 }));
      }
      if (res.statusCode !== 200) {
        let body = "";
        res.on("data", (d) => (body += d));
        res.on("end", () => reject(new Error(`HTTP ${res.statusCode} for ${url}: ${body.slice(0, 200)}`)));
        return;
      }
      const total = parseInt(res.headers["content-length"], 10) || 0;
      let received = 0;
      fs.mkdirSync(path.dirname(destPath), { recursive: true });
      const tmpPath = destPath + ".part";
      const out = fs.createWriteStream(tmpPath);
      res.on("data", (chunk) => {
        received += chunk.length;
        onProgress?.(received, total);
      });
      res.pipe(out);
      out.on("finish", () => {
        out.close(() => {
          try {
            if (fs.existsSync(destPath)) fs.rmSync(destPath, { force: true });
            fs.renameSync(tmpPath, destPath);
            resolve();
          } catch (e) { reject(e); }
        });
      });
      out.on("error", reject);
      res.on("error", reject);
    });
    req.on("error", (e) => {
      try { fs.unlinkSync(destPath + ".part"); } catch { }
      reject(e);
    });
  });
}

ipcMain.handle("hf-search-models", async (_e, opts) => {
  const repoId = ((opts && opts.author) || `${HF_REPO}`).trim();
  const subdir = ((opts && opts.subdir) || HF_REPO_DIR).trim();
  const query = ((opts && opts.query) || "").trim().toLowerCase();
  try {
    // Fetch the repo metadata plus the files inside `subdir` only.
    const metaUrl = `${HF_API}/api/models/${hfEncodePath(repoId)}`;
    const meta = await httpsJson(metaUrl);
    const treeUrl = `${HF_API}/api/models/${hfEncodePath(repoId)}/tree/main/${hfEncodePath(subdir)}?recursive=true`;
    const items = await httpsJson(treeUrl);
    const files = (items || []).filter((it) => it.type === "file" && /\.atf(\.|$)/i.test(it.path || ""));
    const filtered = files.filter((it) => {
      if (!query) return true;
      return (it.path || "").toLowerCase().includes(query);
    });
    if (!filtered.length) {
      return { ok: true, items: [] };
    }
    // Return one card per .atf file so each model shows up individually in
    // the Models tab. The repo metadata is shared across all cards.
    return {
      ok: true,
      items: filtered.map((it) => ({
        // Use "<repoId>::<file path>" as a unique card id; the renderer
        // splits on "::" to recover the file path.
        id: `${repoId}::${it.path}`,
        repoId,
        path: it.path,
        size: it.size || 0,
        likes: meta.likes || 0,
        downloads: meta.downloads || 0,
        lastModified: meta.lastModified || null,
        tags: meta.tags || [],
      })),
    };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("hf-list-files", async (_e, args) => {
  const repoId = (args && args.repoId) || HF_REPO;
  const subdir = (args && args.subdir) || HF_REPO_DIR;
  if (!repoId) return { ok: false, error: "no repo id" };
  try {
    const url = `${HF_API}/api/models/${hfEncodePath(repoId)}/tree/main/${hfEncodePath(subdir)}?recursive=true`;
    const items = await httpsJson(url);
    const files = (items || [])
      .filter((it) => it.type === "file")
      .map((it) => ({ path: it.path, size: it.size || 0 }));
    return { ok: true, files };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

const hfDownloads = new Map(); // downloadId -> AbortController

ipcMain.handle("hf-download-files", async (_e, req) => {
  const repoId = (req && req.repoId) || HF_REPO;
  const downloadId = req && req.downloadId;
  const subdir = (req && req.subdir) || HF_REPO_DIR;
  // Default: every .atf file inside the repo's models/ folder.
  if (!Array.isArray(req?.files) || !req.files.length) {
    try {
      const treeUrl = `${HF_API}/api/models/${hfEncodePath(repoId)}/tree/main/${hfEncodePath(subdir)}?recursive=true`;
      const items = await httpsJson(treeUrl);
      const atf = (items || [])
        .filter((it) => it.type === "file" && /\.atf(\.|$)/i.test(it.path || ""))
        .map((it) => ({ path: it.path, size: it.size || 0 }));
      req = { ...req, repoId, subdir, files: atf };
    } catch (e) {
      // Fall back to the legacy guess: the repo name IS the model file.
      req = { ...req, repoId, subdir, files: [{ path: repoId.split("/").pop() }] };
    }
  }
  let files = Array.isArray(req?.files) && req.files.length
    ? req.files
    : [{ path: repoId.split("/").pop() }];
  if (!repoId || !files[0]?.path) {
    return { ok: false, error: "no files specified" };
  }
  if (!downloadId || hfDownloads.has(downloadId)) {
    return { ok: false, error: "invalid or duplicate download id" };
  }
  const modelsDir = getModelsDirImpl(app.getPath("userData"))
    || MODELS_DIR
    || path.join(app.getPath("userData"), "models");
  const controller = new AbortController();
  hfDownloads.set(downloadId, controller);

  (async () => {
    const saved = [];
    try {
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        const url = `${HF_API}/${hfEncodePath(repoId)}/resolve/main/${hfEncodePath(f.path)}`;
        const destPath = path.join(modelsDir, path.basename(f.path));
        toUI("hf-download-log", `[${repoId}] downloading ${f.path} → ${destPath}`);
        await hfDownloadFile(url, destPath, {
          signal: controller.signal,
          onProgress: (received, total) => {
            toUI("hf-download-progress", {
              downloadId, repoId, file: f.path,
              fileIndex: i, fileCount: files.length,
              receivedBytes: received, totalBytes: total || f.size || 0,
            });
          },
        });
        saved.push(destPath);
        toUI("hf-download-log", `[${repoId}] done: ${f.path}`);
      }
      bridgeWrite({ type: "list_models" });
      toUI("hf-download-done", { downloadId, repoId, files, ok: true, savedPaths: saved });
    } catch (e) {
      toUI("hf-download-log", `[${repoId}] failed: ${e.message}`);
      toUI("hf-download-done", { downloadId, repoId, files, ok: false, error: e.message, savedPaths: saved });
    } finally {
      hfDownloads.delete(downloadId);
    }
  })();

  return { ok: true, downloadId };
});

ipcMain.handle("hf-cancel-download", (_e, downloadId) => {
  const c = hfDownloads.get(downloadId);
  if (!c) return { ok: false };
  c.abort();
  hfDownloads.delete(downloadId);
  return { ok: true };
});

// ── OpenAI-compatible server control ───────────────────────────────
let apiServer = null;
let apiPort = 8000;
let apiLog = "";
let apiRunningState = false;
// v10: per-request start time for elapsed logging in apiVerbose
let requestT0 = 0;

function apiBroadcast(running, stateText) {
  apiRunningState = running;
  toUI("api-server-status", { running, port: apiPort, state: stateText });
}
// Verbose server-log helper for debugging the OpenAI-compatible endpoint.
// Off by default; set ATF_API_VERBOSE=1 (or ATF_DEBUG=1) to turn it on.
// Mirrors every line to BOTH the in-app API log tab (via apiAppend) AND the
// terminal (console.log), so you can read the same trace from either side.
function apiVerbose(line) {
  // Verbose by default so every API request leaves a trace in the API tab
  // and the terminal. Set ATF_API_QUIET=1 to silence.
  if (process.env.ATF_API_QUIET === "1") return;
  // v10: prepend wall-clock timestamp + request-elapsed time
  const ts = new Date().toISOString().slice(11, 23);  // HH:MM:SS.mmm
  const elapsed = requestT0 ? ` | ${((Date.now() - requestT0) / 1000).toFixed(2)}s` : "";
  const prefix = ts + elapsed;
  const full = `${prefix} | ${line}`;
  console.log("[api]", full.replace(/\n$/, ""));
  apiAppend(`[api] ${full}`);
}
function apiAppend(line) {
  apiLog += line;
  if (apiLog.length > 512 * 1024) apiLog = apiLog.slice(-512 * 1024);
  // INCREMENTAL: send only the new bytes (was: whole buffer per line)
  toUI("api-log-append", line);
}

// v16: the API server no longer spawns a second engine process. It is a
// thin HTTP layer that proxies OpenAI-style requests to the SAME bridge
// process that powers the chat UI, so the already-loaded model is reused
// and starting the server costs zero extra RAM.
function json(res, status, obj) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (d) => {
      body += d;
      if (body.length > 32 * 1024 * 1024) {
        reject(new Error("request body too large"));
        try { req.destroy(); } catch { }
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

// OpenAI messages[] -> bridge {system, history[], message}.
// Qwen emits Hermes-style tool markup, so the tool schema and protocol must
// be part of the prompt; the bridge intentionally only transports strings.
function toBridgeMessages(messages, tools = null, toolChoice = null) {
  const textOf = (c) => {
    if (typeof c === "string") return c;
    if (Array.isArray(c)) {
      return c.filter((p) => p && typeof p === "object" && p.type === "text")
        .map((p) => p.text || "").join("");
    }
    return "";
  };
  const system = [];
  const seq = [];
  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    if (m.role === "system") { system.push(textOf(m.content)); continue; }
    if (m.role === "tool") {
      // v20 fix: tool results were silently dropped here -- there was no
      // branch for role=="tool" at all, so the model's history showed it
      // emitting a <tool_call> but NEVER receiving the result. From the
      // model's perspective the call was never answered, so it reissued
      // the same call every turn (the "calls ls -la again and again" loop).
      // Render as a Hermes <tool_response> block starting a new user turn,
      // matching the convention atf/toolcall.py's render_chatml_messages
      // already uses elsewhere in this codebase and that the "# Tools"
      // system directive below tells the model to expect.
      seq.push({ role: "user", content: `<tool_response>\n${textOf(m.content)}\n</tool_response>` });
      continue;
    }
    if (m.role === "user" || m.role === "assistant") {
      let content = textOf(m.content);
      if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
        for (const call of m.tool_calls) {
          const fn = call && call.function;
          if (!fn || typeof fn.name !== "string") continue;
          let args = fn.arguments ?? {};
          if (typeof args === "string") {
            try { args = JSON.parse(args); } catch { }
          }
          content += `<tool_call>\n${JSON.stringify({
            name: fn.name, arguments: args,
          })}\n</tool_call>`;
        }
      }
      seq.push({ role: m.role, content });
    }
  }
  let lastUser = -1;
  for (let k = seq.length - 1; k >= 0; k--) {
    if (seq[k].role === "user") { lastUser = k; break; }
  }
  if (lastUser === -1) {
    throw new Error("messages[] must contain at least one user message");
  }
  const history = [];
  let message = "";
  for (let k = 0; k < seq.length; k++) {
    if (k === lastUser) { message = seq[k].content; continue; }
    if (seq[k].role === "user") history.push({ user: seq[k].content, assistant: "" });
    else if (history.length) history[history.length - 1].assistant += seq[k].content;
  }
  let systemText = system.join("\n");
  if (Array.isArray(tools) && tools.length) {
    systemText += `\n\n# Tools\n\n` +
      `You may call one or more functions to assist with the user query.\n\n` +
      `You are provided with function signatures within <tools></tools> XML tags:\n` +
      `<tools>\n${JSON.stringify(tools)}\n</tools>\n\n` +
      `For each function call, return a json object with function name and ` +
      `arguments within <tool_call></tool_call> XML tags:\n` +
      `<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n` +
      `</tool_call>\n` +
      (toolChoice === "required" || (toolChoice && toolChoice.type === "function")
        ? `When a supplied function is appropriate, you MUST call it and must not answer with prose or code representing the call.`
        : `When a supplied function is appropriate, emit a tool call instead of describing the command in prose.`);
  }
  return { system: systemText, history, message };
}

// One dedicated bridge socket connection per API request; events come back
// tagged with the request id on that same connection. Generation itself is
// serialized inside the bridge, so concurrent callers get a busy error.
function bridgeApiCall(reqObj, onEvent) {
  // v16: no artificial watchdog — long generations are normal and the API
  // waits as long as the bridge takes. Only REAL bridge messages resolve
  // this promise: done/error events, socket errors, or a closed socket.
  return new Promise((resolve, reject) => {
    let settled = false;
    let buf = "";
    let lineCount = 0;
    const t0 = Date.now();
    const conn = net.createConnection(SOCK_PATH);
    const finish = (ok, val, reason) => {
      if (settled) return;
      settled = true;
      try { conn.destroy(); } catch { }
      if (reason !== "done") {
        apiVerbose(`bridgeApiCall id=${reqObj.id} settle ok=${ok}  dt=${Date.now() - t0}ms  lines=${lineCount}  reason=${reason}`);
      }
      if (ok) resolve(val); else reject(val);
    };
    conn.on("connect", () => {
      apiVerbose(`bridgeApiCall id=${reqObj.id} connected  type=${reqObj.type}`);
      try { conn.write(JSON.stringify(reqObj) + "\n"); }
      catch (e) { finish(false, e, "write_failed"); }
    });
    conn.on("data", (d) => {
      buf += d.toString();
      let idx;
      while ((idx = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        lineCount++;
        let msg;
        try { msg = JSON.parse(line); } catch { continue; }
        try { if (onEvent) onEvent(msg); } catch { }
        if (msg.type === "done") return finish(true, { ok: true, event: msg }, "done");
        if (msg.type === "error") return finish(true, { ok: false, event: msg }, "error");
      }
    });
    conn.on("error", (e) => {
      apiVerbose(`bridgeApiCall id=${reqObj.id} socket error after ${Date.now() - t0}ms: ${e.message}`);
      finish(false, e, "socket_error");
    });
    conn.on("close", () => {
      apiVerbose(`bridgeApiCall id=${reqObj.id} socket close (settled=${settled})  dt=${Date.now() - t0}ms  lines=${lineCount}`);
      finish(false, new Error("bridge socket closed before done"), "socket_closed");
    });
    conn.setTimeout(0);   // long generations are normal; no socket-level timeout
  });
}

let apiReqCounter = 0;

// ── v17: model tool-call markup -> OpenAI tool_calls ─────────────────────
// Coding models emit tool calls as literal text in the token stream, e.g.
//   <tool_call>
//   <function=bash>
//   <parameter=command>ls -la</parameter>
//   </function>
//   </tool_call>
// or the JSON-object form  <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
// Clients must receive structured OpenAI tool_calls, never the raw markup.
const TOOL_OPEN = "<tool_call>", TOOL_CLOSE = "</tool_call>";

function parseToolCallBody(body) {
  const t = (body || "").trim();
  if (!t) return null;
  if (t.startsWith("{")) {
    try {
      const o = JSON.parse(t);
      const name = o.name || o.function || (o.function && o.function.name);
      let args = o.arguments ?? o.parameters ?? (o.function && o.function.arguments) ?? {};
      if (typeof args === "string") { try { args = JSON.parse(args); } catch { } }
      if (name) return { name: String(name), arguments: args };
    } catch { /* fall through to tag form */ }
    return null;
  }
  const fm = t.match(/<function\s*=\s*([^>]+)>/i);
  if (!fm) return null;
  const name = fm[1].trim();
  const args = {};
  const pre = /<parameter\s*=\s*([^>]+)>([\s\S]*?)<\/parameter>/gi;
  let m;
  while ((m = pre.exec(t)) !== null) {
    const key = m[1].trim();
    let val = m[2];
    try { val = JSON.parse(val); } catch { /* keep as string */ }
    args[key] = val;
  }
  return { name, arguments: args };
}

function partialPrefixLen(buf, tag) {
  // length of the longest suffix of `buf` that is a proper prefix of `tag`
  const max = Math.min(buf.length, tag.length - 1);
  for (let k = max; k > 0; k--) {
    if (buf.endsWith(tag.slice(0, k))) return k;
  }
  return 0;
}

// Streaming filter: push() plain model text; emits clean content via
// onContent and parsed calls via onToolCall. flush() releases any tail.
//
// `opts.disabled` is for requests that did NOT pass an OpenAI `tools`
// list: the model may still hallucinate <tool_call> blocks, but
// honoring them as real tool calls would cause the client (e.g. pi) to
// execute the phantom tool, feed the result back, and loop forever.  In
// disabled mode every byte of model output is passed through verbatim
// to onContent, the <tool_call> markup reaches the user as visible
// text, and no tool_calls are ever emitted.
function makeToolCallFilter(onContent, onToolCall, opts = {}) {
  const disabled = !!opts.disabled;
  let buf = "";          // plain-text buffer (tag-boundary holdback)
  let inTool = false;    // capturing a <tool_call> block
  let toolBuf = "";      // accumulation inside an open block
  let count = 0;
  let phantomSeen = false;     // warn-once per stream if a phantom tag appeared

  function notePhantom() {
    if (phantomSeen || disabled) return;
    phantomSeen = true;
    apiVerbose(`!! phantom <tool_call> with no tools in request; passing through as text`);
  }

  function endTool() {
    const i = toolBuf.indexOf(TOOL_CLOSE);
    if (i < 0) return false;
    const body = toolBuf.slice(0, i);
    toolBuf = toolBuf.slice(i + TOOL_CLOSE.length);
    const call = parseToolCallBody(body);
    if (call) {
      if (disabled) {
        // Don't surface the phantom call. Re-emit the entire block,
        // including the markup, as content so the user can see what the
        // model produced and so the model gets a stable turn shape.
        notePhantom();
        onContent(TOOL_OPEN + body + TOOL_CLOSE);
      } else {
        count += 1;
        onToolCall(call);
      }
    } else {
      // Unparseable body -- always pass through as text so nothing is lost.
      onContent(TOOL_OPEN + body + TOOL_CLOSE);
    }
    inTool = false;
    buf = toolBuf;        // text after </tool_call> resumes normal flow
    toolBuf = "";
    return true;
  }

  function drainPlain() {
    if (inTool) return;
    // hold back a possible partial "<tool_call>" prefix at the tail
    const keep = partialPrefixLen(buf, TOOL_OPEN);
    if (buf.length > keep) {
      onContent(buf.slice(0, buf.length - keep));
      buf = buf.slice(buf.length - keep);
    }
  }

  function step() {
    if (inTool) {
      while (endTool()) { /* drain chained blocks */ }
      if (!inTool) step();          // resume scanning post-block remainder
      return;
    }
    const i = buf.indexOf(TOOL_OPEN);
    if (i < 0) { drainPlain(); return; }
    const plain = buf.slice(0, i);
    buf = buf.slice(i + TOOL_OPEN.length);
    if (plain) onContent(plain);
    inTool = true;
    toolBuf = buf;                  // EVERYTHING after the opener is block body
    buf = "";
    step();
  }

  return {
    push(text) {
      if (!text) return;
      if (disabled) {
        // Bypass the tag scanner entirely: every byte is content, even
        // literal <tool_call>...</tool_call> markup from a hallucinating
        // model. This prevents the client from seeing a phantom tool_call
        // and the agent harness from looping.
        onContent(text);
        return;
      }
      if (inTool) toolBuf += text;
      else buf += text;
      step();
    },
    flush() {
      if (disabled) { buf = ""; toolBuf = ""; return count; }
      if (inTool) {
        // unterminated block at EOS: surface it as content so nothing is lost
        onContent(TOOL_OPEN + toolBuf);
        inTool = false; toolBuf = "";
      } else if (buf) {
        onContent(buf);
      }
      buf = "";
      return count;
    },
    get count() { return count; },
  };
}


async function handleChatCompletions(req, res, body) {
  let payload;
  try { payload = JSON.parse(body); } catch {
    return json(res, 400, { error: { message: "invalid JSON body" } });
  }
  const messages = payload.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    apiVerbose(`!! 400 messages[] missing or empty`);
    return json(res, 400, { error: { message: "messages[] is required" } });
  }
  // Accept the legacy OpenAI `functions` field used by older harnesses.
  const requestedTools = Array.isArray(payload.tools) ? payload.tools :
    (Array.isArray(payload.functions)
      ? payload.functions.map((fn) => ({ type: "function", function: fn }))
      : null);
  apiVerbose(`chat parsed: messages=${messages.length}  roles=[${messages.map((m) => m?.role).join(",")}]  stream=${!!payload.stream}  requested_model=${JSON.stringify(payload.model)}`);
  const requestedModel = (typeof payload.model === "string" && payload.model.trim()) || null;
  // Decide which model to serve. If the request names one that's registered,
  // auto-load it; otherwise fall back to whatever is currently resident.
  let activeModel = currentModelId;
  if (requestedModel && requestedModel !== currentModelId) {
    const known = registry.find((m) => m.id === requestedModel);
    if (!known) {
      apiVerbose(`!! 404 unknown model ${requestedModel}  registered=${registry.map((m) => m.id).join(",") || "(empty)"}`);
      return json(res, 404, {
        error: {
          message: `unknown model: ${JSON.stringify(requestedModel)}; registered: ${registry.map((m) => m.id).join(", ") || "(none — wait for the registry to load)"}`,
          type: "model_not_found",
        },
      });
    }
    apiAppend(`[chat] auto-loading ${requestedModel} for request\n`);
    apiVerbose(`auto-load requested: ${requestedModel}  (current=${currentModelId || "(none)"})`);
    try {
      const tLoad = Date.now();
      await loadModelAndWait(requestedModel);
      apiVerbose(`auto-load done: ${requestedModel}  dt=${Date.now() - tLoad}ms`);
      activeModel = currentModelId;
    } catch (e) {
      apiVerbose(`!! 503 auto-load failed: ${e.message}`);
      return json(res, 503, { error: { message: `could not load ${requestedModel}: ${e.message}`, type: "load_failed" } });
    }
  } else if (!activeModel) {
    apiVerbose(`!! 503 no model loaded  registered=${registry.map((m) => m.id).join(",") || "(empty)"}`);
    return json(res, 503, {
      error: {
        message: `no model loaded — either pick one in ATF Chat first or pass "model" in the request body; registered: ${registry.map((m) => m.id).join(", ") || "(none — wait for the registry to load)"}`,
        type: "no_model",
      },
    });
  }
  apiVerbose(`active model: ${activeModel}  (sock=${sock && !sock.destroyed ? "up" : "DOWN"}  ready=${ready})`);

  let conv;
  try { conv = toBridgeMessages(messages, requestedTools, payload.tool_choice); }
  catch (e) { return json(res, 400, { error: { message: e.message } }); }

  const rid = `api-${Date.now().toString(36)}-${apiReqCounter++}`;
  const genReq = {
    type: "generate",
    id: rid,
    message: conv.message,
    history: conv.history,
    // absent max_tokens -> let the adaptive router tier decide the budget
    max_tokens: Math.min(Math.max(parseInt(payload.max_tokens, 10) || 8192, 1), 8192),
    thinking: typeof payload.thinking === "string" ? payload.thinking : "auto",
  };
  if (conv.system) genReq.system = conv.system;
  if (payload.temperature != null) genReq.temperature = payload.temperature;
  if (payload.top_p != null) genReq.top_p = payload.top_p;

  const cid = `chatcmpl-${rid}`;
  const created = Math.floor(Date.now() / 1000);
  // The chat request was validated to either auto-load requestedModel or fall
  // back to the currently loaded one; `activeModel` reflects that decision.
  const model = activeModel;
  apiAppend(`[chat] request → model=${model} stream=${!!payload.stream} messages=${messages.length}\n`);
  apiVerbose(`genReq id=${rid}  model=${model}  max_tokens=${genReq.max_tokens}  thinking=${genReq.thinking}  msg="${conv.message.slice(0, 80).replace(/\n/g, " ")}${conv.message.length > 80 ? "…" : ""}"  history=${conv.history.length}  sys=${conv.system ? `"${conv.system.slice(0, 60).replace(/\n/g, " ")}…"` : "(none)"}`);
  const stream = !!payload.stream;
  // event counters for the verbose trace
  const evt = { tokens: 0, thinkTokens: 0, thinks: 0, usage: null, done: null };

  let reasoning = "";
  let inThink = false;
  let usageEvent = null;
  let doneEvent = null;     // done event, or {__error: errEvent}
  let clientGone = false;
  const toolCalls = [];
  let lastProgressLine = null;     // v17: parsed out of the model's markup

  const sseChunk = (delta, finish) => JSON.stringify({
    id: cid, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta, finish_reason: finish ?? null }],
  });

  // v17: route non-think text through the tool-call filter. Clean text
  // streams as content deltas; <tool_call> blocks become OpenAI tool_calls.
  let toolIndex = 0;
  let content = "";
  const emitContent = (text) => {
    content += text;
    if (stream && !clientGone) res.write(`data: ${sseChunk({ content: text })}\n\n`);
  };
  const emitToolCall = (call) => {
    toolCalls.push(call);
    if (!stream || clientGone) return;
    res.write(`data: ${sseChunk({
      tool_calls: [{
        index: toolIndex,
        id: `call_${cid}-${toolIndex}`,
        type: "function",
        function: { name: call.name, arguments: JSON.stringify(call.arguments ?? {}) },
      }],
    })}\n\n`);
    toolIndex += 1;
  };
  // Disable tool-call parsing when the client didn't ask for any tools.
  // A hallucinated <tool_call> with no `tools` in the request would
  // otherwise be surfaced as a real tool_call, the client (pi) would
  // execute the phantom tool, feed the result back, and the model
  // would hallucinate the same call again -- an infinite loop.
  const toolFilter = makeToolCallFilter(emitContent, emitToolCall, {
    disabled: !requestedTools || requestedTools.length === 0,
  });

  const onEvent = (msg) => {
    if (msg.id !== rid) return;
    switch (msg.type) {
      case "think":
        inThink = !!msg.open;
        evt.thinks += 1;
        apiVerbose(`event think open=${msg.open}`);
        break;
      case "usage":
        usageEvent = msg;
        evt.usage = msg;
        apiVerbose(`event usage prompt=${msg.prompt_tokens} gen=${msg.gen_tokens ?? "?"} tps=${msg.tps ?? "?"}`);
        break;
      case "progress":
        if (msg.stage === "prefill") {
          const line = `prefilling ${msg.done}/${msg.total} tokens`;
          if (line !== lastProgressLine) {
            lastProgressLine = line;
            apiAppend(`[chat] ${line}\n`);
            apiVerbose(`event progress ${line}`);
            // SSE comment: legal inside any event stream, ignored by parsers,
            // keeps proxies from timing the connection out during long prompts
            if (stream && !clientGone) res.write(`: ${line}\n\n`);
          }
        }
        break;
      case "token": {
        if (inThink) {
          reasoning += msg.text;
          evt.thinkTokens += (msg.text || "").length;
          if (stream && !clientGone) res.write(`data: ${sseChunk({ reasoning_content: msg.text })}\n\n`);
        } else {
          evt.tokens += 1;
          toolFilter.push(msg.text || "");
        }
        // throttle the per-token trace (verbose gets noisy otherwise)
        if (process.env.ATF_API_VERBOSE && (evt.tokens <= 3 || evt.tokens % 25 === 0)) {
          apiVerbose(`event token #${evt.tokens} think=${inThink} text=${JSON.stringify((msg.text || "").slice(0, 40))}${msg.text && msg.text.length > 40 ? "…" : ""}`);
        }
        break;
      }
      case "done":
        doneEvent = msg;
        evt.done = msg;
        apiVerbose(`event done stop_reason=${msg.stop_reason || "?"} gen_tokens=${msg.gen_tokens ?? "?"} prompt_tokens=${msg.prompt_tokens ?? "?"}`);
        break;
      case "error":
        if (!doneEvent) doneEvent = { __error: msg };
        apiVerbose(`!! event error: ${msg.message || JSON.stringify(msg)}`);
        break;
      default: break;
    }
  };

  res.on("close", () => {
    clientGone = true;
    // v20: pass this request's id so the bridge only cancels IT, not
    // whatever else may be running by the time this stop is processed --
    // see the rtype == "stop" handler in atf_bridge.py.
    if (!doneEvent) { try { sendNow({ type: "stop", id: rid }); } catch { } }
  });

  if (stream) {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    });
    res.write(`data: ${sseChunk({ role: "assistant" })}\n\n`);
  }

  const tBridge = Date.now();
  apiVerbose(`bridge -> send id=${rid}  size=${JSON.stringify(genReq).length}B  sock=${sock && !sock.destroyed ? "up" : "DOWN"}`);
  let result;
  try {
    result = await bridgeApiCall(genReq, onEvent);
  } catch (e) {
    apiVerbose(`!! bridge call failed id=${rid}  dt=${Date.now() - tBridge}ms  err="${e.message}"`);
    if (clientGone) return;
    if (stream) {
      res.write(`data: ${sseChunk({ content: `\n[error: ${e.message}]` })}\n\n`);
      res.write(`data: ${sseChunk({}, "stop")}\n\n`);
      res.write("data: [DONE]\n\n");
      return res.end();
    }
    return json(res, 502, { error: { message: `bridge connection failed: ${e.message}` } });
  }
  apiVerbose(`bridge <- done id=${rid}  dt=${Date.now() - tBridge}ms  result.ok=${result?.ok}  events: tokens=${evt.tokens} think_tokens=${evt.thinkTokens} thinks=${evt.thinks}  content=${content.length}B reasoning=${reasoning.length}B`);
  if (clientGone) {
    apiVerbose(`client disconnected before finalize id=${rid}`);
    return;
  }

  const err = (doneEvent && doneEvent.__error) || (!result.ok && result.event) || null;
  if (err) {
    apiVerbose(`!! returning error id=${rid}  busy=${!!err.busy}  msg="${err.message || "generation failed"}"`);
    if (stream) {
      res.write(`data: ${sseChunk({ content: `\n[error: ${err.message}]` })}\n\n`);
      res.write(`data: ${sseChunk({}, "stop")}\n\n`);
      res.write("data: [DONE]\n\n");
      return res.end();
    }
    return json(res, err.busy ? 429 : 500, {
      error: {
        message: err.message || "generation failed",
        type: err.busy ? "server_busy" : "internal_error",
      },
    });
  }

  const promptTokens = (doneEvent && doneEvent.prompt_tokens)
    || (usageEvent && usageEvent.prompt_tokens) || 0;
  const completionTokens = (doneEvent && doneEvent.gen_tokens)
    || Math.round(content.length / 4);

  if (stream) {
    toolFilter.flush();
    const finish = toolCalls.length ? "tool_calls" : "stop";
    res.write(`data: ${sseChunk({}, finish)}\n\n`);
    res.write("data: [DONE]\n\n");
    apiVerbose(`SSE finalize id=${rid}  prompt=${promptTokens} gen=${completionTokens} tool_calls=${toolCalls.length}`);
    return res.end();
  }

  toolFilter.flush();
  const msg = { role: "assistant", content };
  if (reasoning) msg.reasoning_content = reasoning;
  if (toolCalls.length) {
    msg.tool_calls = toolCalls.map((c, i) => ({
      id: `call_${cid}-${i}`, type: "function",
      function: { name: c.name, arguments: JSON.stringify(c.arguments ?? {}) },
    }));
  }
  const finishReason = toolCalls.length ? "tool_calls" : "stop";
  apiVerbose(`JSON finalize id=${rid}  prompt=${promptTokens} gen=${completionTokens}  content_len=${content.length}`);
  return json(res, 200, {
    id: cid, object: "chat.completion", created, model,
    choices: [{ index: 0, message: msg, finish_reason: finishReason }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  });
}

// Ask the bridge for the freshest registry; resolve with whatever the
// bridge reports (or with the cached list if the socket is down).
function refreshRegistry(timeoutMs = 1500) {
  return new Promise((resolve) => {
    if (!sock || sock.destroyed) return resolve(registry);
    registryListeners.push(resolve);
    const timer = setTimeout(() => {
      // drop ourselves from the listener list and fall back to cache
      registryListeners = registryListeners.filter((r) => r !== resolve);
      resolve(registry);
    }, timeoutMs);
    const prevResolve = resolve;
    // patch resolve so the timer is cleared when we actually receive the update
    resolve = ((val) => { clearTimeout(timer); prevResolve(val); });
    registryListeners[registryListeners.length - 1] = resolve;
    try { sendNow({ type: "list_models" }); } catch { resolve(registry); }
  });
}

function listModelsHandler(_req, res) {
  const build = () => {
    const data = registry.length
      ? registry.map((m) => ({
        id: m.id, object: "model", owned_by: "atf",
        format: m.format, size_gb: m.size_gb, loaded: !!m.loaded,
      }))
      : [];
    apiAppend(`[models] ${data.length} registered (${data.filter((m) => m.loaded).length} loaded)
`);
    return json(res, 200, { object: "list", data });
  };
  if (registry.length === 0) {
    refreshRegistry().then(build);
  } else {
    build();
  }
}

// Load a specific model over the bridge socket and resolve once the bridge
// confirms it is loaded (or rejects after `timeoutMs`). Mirrors the lazy
// loading the standalone python server does.
// Load a specific model over the bridge and resolve when the bridge confirms
// it is loaded. Mirrors the lazy-load behavior of the standalone python
// server. Sets apiAppend() log lines so the API tab reflects progress.
function loadModelAndWait(modelId, timeoutMs = 180_000) {
  if (!sock || sock.destroyed) {
    return Promise.reject(new Error("bridge not connected"));
  }
  if (currentModelId === modelId) return Promise.resolve();

  // dedupe concurrent loads for the same id
  if (loadModelAndWait._pending && loadModelAndWait._pending.id === modelId) {
    return new Promise((resolve, reject) => {
      loadModelAndWait._pending.waiters.push({ resolve, reject });
    });
  }

  let resolveOuter, rejectOuter;
  const promise = new Promise((res, rej) => { resolveOuter = res; rejectOuter = rej; });
  loadModelAndWait._pending = {
    id: modelId,
    waiters: [{ resolve: resolveOuter, reject: rejectOuter }],
  };

  const cleanup = (err) => {
    if (loadModelAndWait._pending && loadModelAndWait._pending.id === modelId) {
      const waiters = loadModelAndWait._pending.waiters;
      loadModelAndWait._pending = null;
      if (err) for (const w of waiters) w.reject(err);
    }
  };

  apiAppend(`[load] requesting model "${modelId}" from the engine\n`);
  const onLoaded = (msg) => {
    if (!loadModelAndWait._pending || msg.id !== modelId) return;
    bridge?.stdout?.off("data", onStdout);
    sock?.off("data", onSockData);
    clearTimeout(timer);
    apiAppend(`[load] model "${modelId}" ready\n`);
    const waiters = loadModelAndWait._pending.waiters;
    loadModelAndWait._pending = null;
    for (const w of waiters) w.resolve();
  };
  const onError = (msg) => {
    if (!loadModelAndWait._pending) return;
    bridge?.stdout?.off("data", onStdout);
    sock?.off("data", onSockData);
    clearTimeout(timer);
    cleanup(new Error(msg.message || "load failed"));
    apiAppend(`[load] failed: ${msg.message || "unknown error"}\n`);
  };
  // per-call parser buffers so concurrent loads don't interleave
  const bufOut = { text: "" };
  const bufSock = { text: "" };
  const onStdout = (d) => parseChunk(bufOut, d.toString(), onLoaded, onError);
  const onSockData = (d) => parseChunk(bufSock, d.toString(), onLoaded, onError);

  const timer = setTimeout(() => {
    bridge?.stdout?.off("data", onStdout);
    sock?.off("data", onSockData);
    cleanup(new Error(`load ${modelId} timed out after ${Math.round(timeoutMs / 1000)}s`));
    apiAppend(`[load] timed out waiting for "${modelId}"\n`);
  }, timeoutMs);

  bridge?.stdout?.on("data", onStdout);
  sock.on("data", onSockData);
  try {
    sendNow({ type: "load", model: modelId });
  } catch (e) {
    bridge?.stdout?.off("data", onStdout);
    sock.off("data", onSockData);
    clearTimeout(timer);
    cleanup(e);
  }
  return promise;
}

// Tiny line parser; each caller gets its own buffer to stay race-free
// across concurrent loadModelAndWait invocations.
function parseChunk(buf, text, onLoaded, onError) {
  buf.text += text;
  let idx;
  while ((idx = buf.text.indexOf("\n")) !== -1) {
    const line = buf.text.slice(0, idx).trim();
    buf.text = buf.text.slice(idx + 1);
    if (!line) continue;
    let m; try { m = JSON.parse(line); } catch { continue; }
    if (!m) continue;
    if (m.type === "model-loaded") onLoaded(m);
    else if (m.type === "error") onError(m);
  }
}

async function handleApiRequest(req, res) {
  const url = (req.url || "").split("?")[0];
  const t0 = Date.now();
  // v10: set module-level requestT0 for elapsed timing in subsequent apiVerbose calls
  requestT0 = t0;
  apiVerbose(`-> ${req.method} ${url}  remote=${req.socket?.remoteAddress}:${req.socket?.remotePort}  ua="${(req.headers["user-agent"] || "").slice(0, 60)}"`);
  const onDone = (status) => () => {
    apiVerbose(`<- ${req.method} ${url}  status=${status}  dt=${Date.now() - t0}ms`);
  };
  res.on("close", onDone(res.statusCode));
  res.on("finish", onDone(res.statusCode));

  if (req.method === "GET" && (url === "/health" || url === "/v1/health")) {
    apiVerbose(`health: ok model=${currentModelId || "(none)"}`);
    return json(res, 200, { ok: true, model: currentModelId });
  }
  if (req.method === "GET" && (url === "/v1/models" || url === "/models")) {
    return listModelsHandler(req, res);
  }
  if (req.method === "POST" && (url === "/v1/chat/completions" || url === "/chat/completions")) {
    const body = await readBody(req);
    apiVerbose(`chat body: ${body.length}B  preview=${JSON.stringify(body.slice(0, 200))}${body.length > 200 ? "…" : ""}`);
    return handleChatCompletions(req, res, body);
  }
  apiVerbose(`!! 404 ${req.method} ${url}`);
  return json(res, 404, { error: { message: `not found: ${req.method} ${url}` } });
}

function probeAtf(port, timeoutMs, cb) {
  // a real ATF/OpenAI server answers GET /v1/models with JSON
  const req = http.get(
    { host: "127.0.0.1", port, path: "/v1/models", timeout: timeoutMs },
    (res) => {
      let body = "";
      res.on("data", (d) => (body += d));
      res.on("end", () => cb(res.statusCode === 200 && body.trim().startsWith("{")));
    });
  req.on("timeout", () => { req.destroy(); cb(false); });
  req.on("error", () => cb(false));
}

ipcMain.handle("api-server-start", (_e, opts) => {
  const port = Math.min(65535, Math.max(1024, parseInt(opts?.port, 10) || 8000));
  if (apiServer && apiPort === port) { apiBroadcast(true, "running"); return; }

  probeAtf(port, 1500, (isAtf) => {
    if (isAtf) {
      apiLog += `[detect] found an existing ATF server on port ${port} — adopting it\n`;
      apiBroadcast(true, "running (already active)");
    } else {
      // distinguish "nothing there" from "occupied by something else"
      const probe = net.createConnection(port, "127.0.0.1");
      probe.setTimeout(1200);
      probe.on("error", () => actuallyStart(port));
      probe.on("timeout", () => { try { probe.destroy(); } catch { } actuallyStart(port); });
      probe.on("connect", () => {
        try { probe.destroy(); } catch { }
        apiLog += `[error] port ${port} is occupied by another service (not an OpenAI-compatible server)\n`;
        apiBroadcast(false, `port ${port} busy — pick another`);
      });
    }
  });
});

function actuallyStart(port) {
  if (apiServer) {
    try { apiServer.close(); apiServer.closeAllConnections?.(); } catch { }
    apiServer = null;
  }

  apiPort = port;
  apiLog = "";
  apiAppend(`[launch] OpenAI-compatible server on http://127.0.0.1:${apiPort}/v1\n`);
  apiAppend(`[reuse] serving the model already loaded in the engine (${currentModelId || "none yet"}) — no second copy\n`);
  const verboseOn = process.env.ATF_API_QUIET !== "1";
  apiAppend(`[verbose] detailed request traces are ${verboseOn ? "ON by default" : "silenced"} (set ATF_API_QUIET=1 to disable)\n`);
  apiBroadcast(true, "starting…");

  apiServer = http.createServer((req, res) => {
    handleApiRequest(req, res).catch((e) => {
      try {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: String((e && e.message) || e) } }));
      } catch { }
    });
  });
  apiServer.on("error", (e) => {
    apiAppend(`[error] ${e.message}\n`);
    apiServer = null;
    apiBroadcast(false, `server error: ${e.message}`);
  });
  apiServer.listen(port, "127.0.0.1", () => {
    apiAppend(`[ready] listening on http://127.0.0.1:${apiPort}/v1\n`);
    apiBroadcast(true, "running");
  });
}

ipcMain.handle("api-server-stop", () => {
  stopApiServer("requested from UI");
});

// Programmatic shutdown used when the engine drops the model underneath the
// server (explicit unload, bridge crash, etc.). `reason` is appended to the
// log so the user can see why the server stopped in the API tab.
function stopApiServer(reason) {
  if (!apiServer) {
    if (apiRunningState) {
      apiLog += `[stop] ${reason} — adopted an external server cannot be stopped from the app\n`;
      apiBroadcast(false, "stopped (external)");
    }
    return;
  }
  apiLog += `[stop] ${reason}\n`;
  try { apiServer.close(); apiServer.closeAllConnections?.(); } catch { }
  apiServer = null;
  apiBroadcast(false, "stopped");
}

ipcMain.handle("get-state", () => ({
  engineReady: ready,
  currentModelId,
  apiServer: { running: !!apiServer || apiRunningState, owned: !!apiServer, port: apiPort },
  apiLog,
}));

// ── persistent store: sessions + settings + window state ───────────
// Simple JSON-per-key store under userData. No schema, no dependencies.
const storeDir = () => {
  const dir = path.join(app.getPath("userData"), "store");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
};
const storePath = (key) => path.join(storeDir(), `${key}.json`);

ipcMain.handle("store-get", (_e, key) => {
  try {
    return JSON.parse(fs.readFileSync(storePath(String(key)), "utf8"));
  } catch {
    return null;
  }
});

ipcMain.handle("store-set", (_e, key, value) => {
  const tmp = storePath(String(key)) + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(value));
  fs.renameSync(tmp, storePath(String(key)));
  return true;
});

ipcMain.handle("store-delete", (_e, key) => {
  try { fs.unlinkSync(storePath(String(key))); } catch { }
  return true;
});

// ── v19: chat sessions — SQLite (better-sqlite3) ────────────────
//
// The previous design wrote all sessions into a single JSON file under the
// "sessions" key. That had two failure modes that bit us: a stale empty
// in-memory array would atomically overwrite the whole list on save (the
// cause of the v19 loss of 6 sessions), and the whole list had to be
// re-serialized on every chat update. SQLite gives us per-row writes, a
// typed schema, and free CASCADE deletes.
//
// The schema preserves every field from the JSON shape so nothing is lost
// in migration. We keep the JSON store IPCs above for "settings" and
// "kvssd" — only the chat-session path moves to SQLite.
//
// better-sqlite3 is added as a dep and rebuilt against the Electron Node
// ABI via @electron/rebuild. The rebuilt native module lives in
// node_modules/better-sqlite3/build/Release/better_sqlite3.node.

let _sessionDb = null;

function sessionDbPath() {
  return path.join(storeDir(), "sessions.sqlite3");
}

function openSessionDb() {
  if (_sessionDb) return _sessionDb;
  // node_modules/better-sqlite3 — only require if we get here so a missing
  // native module does not blow up the whole app on boot. The first call
  // to any session IPC is the earliest it can fail; we surface a clear
  // error to the renderer.
  let Database;
  try {
    // better-sqlite3 exports the Database class as the module's
    // default export — not as a named property. The destructure
    // ({ Database } = ...) yields undefined and then `new Database`
    // throws. Capture the module and pick the constructor we want.
    const mod = require("better-sqlite3");
    Database = typeof mod === "function" ? mod : (mod.Database || mod.default);
    if (typeof Database !== "function") {
      throw new Error("better-sqlite3 did not export a Database constructor");
    }
  } catch (e) {
    throw new Error("better-sqlite3 is not installed or was not rebuilt against this Electron version. " +
      "Fix: cd electron && node scripts/rebuild-native.mjs --force (" + (e && e.message) + ")");
  }
  const dbPath = sessionDbPath();
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);
  // WAL gives us concurrent readers (the UI scans the list on every
  // dashSync tick) without blocking the writer; foreign keys are off
  // by default in SQLite, we want CASCADE.
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.pragma("synchronous = NORMAL");
  db.exec(SCHEMA_SQL);
  // Migrate from the legacy JSON file exactly once. We only do it when
  // the messages table is empty so re-running the app is safe.
  const msgCount = db.prepare("SELECT COUNT(*) AS n FROM messages").get().n;
  if (msgCount === 0) {
    migrateJsonToSqlite(db);
  }
  _sessionDb = db;
  return db;
}

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS sessions (
  id      TEXT PRIMARY KEY,
  title   TEXT NOT NULL,
  created INTEGER NOT NULL,
  updated INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  position   INTEGER NOT NULL,
  role       TEXT NOT NULL,
  content    TEXT NOT NULL,
  model      TEXT,
  thinking   TEXT,
  truncated  INTEGER,
  tier       TEXT,    -- JSON blob (badge, name, score)
  stats      TEXT,    -- JSON blob (tokens, seconds, tps, thinkSeconds, stopReason)
  created    INTEGER NOT NULL,
  UNIQUE (session_id, position)
);
CREATE INDEX IF NOT EXISTS idx_messages_session
  ON messages(session_id, position);
`;

// One-time migration: legacy store/sessions.json (list of session
// objects) -> sessions + messages tables. Atomic in a single
// transaction; on success we rename the JSON to .migrated so a
// re-run won't double-insert.
function migrateJsonToSqlite(db) {
  const jsonPath = storePath("sessions");
  if (!fs.existsSync(jsonPath)) return;
  let list;
  try {
    list = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  } catch (e) {
    console.warn("[sessions] legacy sessions.json could not be parsed; skipping migration:", e && e.message);
    return;
  }
  if (!Array.isArray(list) || list.length === 0) {
    // Empty / non-array — rename anyway so we don't keep re-attempting.
    try { fs.renameSync(jsonPath, jsonPath + ".migrated"); } catch { }
    return;
  }
  const insSess = db.prepare(
    "INSERT OR REPLACE INTO sessions (id, title, created, updated) VALUES (?, ?, ?, ?)"
  );
  const insMsg = db.prepare(
    `INSERT OR REPLACE INTO messages
       (session_id, position, role, content, model, thinking, truncated, tier, stats, created)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  );
  const txn = db.transaction((rows) => {
    for (const s of rows) {
      if (!s || typeof s.id !== "string") continue;
      insSess.run(
        s.id,
        typeof s.title === "string" ? s.title : "New chat",
        Number.isFinite(s.created) ? s.created : Date.now(),
        Number.isFinite(s.updated) ? s.updated : Date.now(),
      );
      const msgs = Array.isArray(s.messages) ? s.messages : [];
      msgs.forEach((m, i) => {
        if (!m || typeof m.role !== "string") return;
        insMsg.run(
          s.id,
          i,
          m.role,
          typeof m.content === "string" ? m.content : "",
          m.model || null,
          m.thinking || null,
          m.truncated ? 1 : 0,
          m.tier ? JSON.stringify(m.tier) : null,
          m.stats ? JSON.stringify(m.stats) : null,
          Number.isFinite(m.created) ? m.created : Date.now(),
        );
      });
    }
  });
  try {
    txn(list);
    // Only rename after a successful txn so a crash mid-migration leaves
    // the JSON intact for the next launch to retry.
    fs.renameSync(jsonPath, jsonPath + ".migrated");
    console.log(`[sessions] migrated ${list.length} session(s) from sessions.json to sessions.sqlite3`);
  } catch (e) {
    console.error("[sessions] migration failed; leaving sessions.json in place:", e);
  }
}

// Convert a SQLite row into the same JS object shape the renderer has
// always used, so the rest of the app can keep treating sessions as
// { id, title, created, updated, messages: [...] }.
function rowToSession(row, msgRows) {
  return {
    id: row.id,
    title: row.title,
    created: row.created,
    updated: row.updated,
    messages: msgRows.map((m) => {
      const out = { role: m.role, content: m.content };
      if (m.model) out.model = m.model;
      if (m.thinking) out.thinking = m.thinking;
      if (m.truncated) out.truncated = true;
      if (m.tier) {
        try { out.tier = JSON.parse(m.tier); } catch { /* ignore corrupt blob */ }
      }
      if (m.stats) {
        try { out.stats = JSON.parse(m.stats); } catch { /* ignore */ }
      }
      return out;
    }),
  };
}

ipcMain.handle("sessions-list", () => {
  const db = openSessionDb();
  // Metadata only — no messages. The renderer keeps a list of session
  // summaries in the sidebar; the full message list is fetched on
  // demand when a session is opened.
  return db.prepare(
    "SELECT id, title, created, updated FROM sessions ORDER BY updated DESC"
  ).all();
});

ipcMain.handle("sessions-active-get", (_e, activeId) => {
  const db = openSessionDb();
  const id = typeof activeId === "string" ? activeId : null;
  if (!id) return null;
  const row = db.prepare("SELECT id, title, created, updated FROM sessions WHERE id = ?").get(id);
  if (!row) return null;
  const msgs = db.prepare(
    "SELECT role, content, model, thinking, truncated, tier, stats FROM messages WHERE session_id = ? ORDER BY position ASC"
  ).all(id);
  const result = rowToSession(row, msgs);
  // v17+ DEBUG: surface the raw SQLite read in the main-process stdout so
  // the user can see exactly what messages were recovered (or not) for a
  // session. Truncated to keep the log readable.
  const n = result && result.messages ? result.messages.length : 0;
  console.log(`[sessions-active-get] id=${id} title=${JSON.stringify(result && result.title)} msg_count=${n}`);
  if (n > 0) {
    for (const m of result.messages) {
      console.log(`  [${m.role}] len=${(m.content || "").length} content[:120]=${JSON.stringify((m.content || "").slice(0, 120))}`);
    }
  }
  return result;
});

ipcMain.handle("sessions-save", (_e, sess) => {
  if (!sess || typeof sess.id !== "string") {
    throw new Error("sessions-save: invalid session");
  }
  // v17+ DEBUG: dump the incoming session so we can see what the renderer
  // is sending to persist. Truncated to keep the log readable.
  console.log(`[sessions-save] id=${sess.id} title=${JSON.stringify(sess.title)} msg_count=${Array.isArray(sess.messages) ? sess.messages.length : 0}`);
  if (Array.isArray(sess.messages)) {
    for (let ii = 0; ii < sess.messages.length; ii++) {
      const m = sess.messages[ii];
      console.log(`  in[${ii}] role=${m && m.role} len=${m && m.content ? m.content.length : 0} content[:120]=${JSON.stringify(m && m.content ? m.content.slice(0, 120) : null)}`);
    }
  }
  const db = openSessionDb();
  const now = Date.now();
  const title = typeof sess.title === "string" && sess.title.trim() ? sess.title.trim() : "New chat";
  const created = Number.isFinite(sess.created) ? sess.created : now;
  const updated = Number.isFinite(sess.updated) ? sess.updated : now;
  const msgs = Array.isArray(sess.messages) ? sess.messages : [];

  const txn = db.transaction(() => {
    db.prepare(
      "INSERT INTO sessions (id, title, created, updated) VALUES (?, ?, ?, ?) " +
      "ON CONFLICT(id) DO UPDATE SET title = excluded.title, updated = excluded.updated"
    ).run(sess.id, title, created, updated);

    // Replace the message list wholesale. Cheaper than diffing in
    // practice (chat messages are short, sessions are capped at a few
    // hundred) and avoids any positional-shift bugs.
    db.prepare("DELETE FROM messages WHERE session_id = ?").run(sess.id);
    const ins = db.prepare(
      `INSERT INTO messages
         (session_id, position, role, content, model, thinking, truncated, tier, stats, created)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );
    msgs.forEach((m, i) => {
      if (!m || typeof m.role !== "string") return;
      ins.run(
        sess.id,
        i,
        m.role,
        typeof m.content === "string" ? m.content : "",
        m.model || null,
        m.thinking || null,
        m.truncated ? 1 : 0,
        m.tier ? JSON.stringify(m.tier) : null,
        m.stats ? JSON.stringify(m.stats) : null,
        Number.isFinite(m.created) ? m.created : now,
      );
    });
  });
  txn();
  // v17+ DEBUG: confirm the persist round-trip. The read-back should
  // match the in-memory msgs array length.
  try {
    const dbCheck = openSessionDb();
    const cnt = dbCheck.prepare("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?").get(sess.id).n;
    console.log(`[sessions-save] DONE id=${sess.id} wrote=${msgs.length} read_back=${cnt}`);
  } catch (e) {
    console.log(`[sessions-save] DONE id=${sess.id} wrote=${msgs.length} read_back=ERR ${e.message}`);
  }
  return { ok: true, id: sess.id, updated };
});

ipcMain.handle("sessions-delete", (_e, id) => {
  if (typeof id !== "string") return { ok: false };
  const db = openSessionDb();
  db.prepare("DELETE FROM sessions WHERE id = ?").run(id);
  return { ok: true };
});

ipcMain.handle("sessions-rename", (_e, id, title) => {
  if (typeof id !== "string" || typeof title !== "string") return { ok: false };
  const db = openSessionDb();
  db.prepare("UPDATE sessions SET title = ?, updated = ? WHERE id = ?")
    .run(title.trim() || "New chat", Date.now(), id);
  return { ok: true };
});

// ── models directory (user-configurable) ──────────────────────────

const { setModelsDir: setModelsDirImpl, getModelsDir: getModelsDirImpl } = require("./app_venv.cjs");

ipcMain.handle("get-models-dir", () => {
  return getModelsDirImpl(app.getPath("userData"));
});

ipcMain.handle("set-models-dir", async (_e, newDir) => {
  const abs = setModelsDirImpl(app.getPath("userData"), newDir);
  MODELS_DIR = abs;
  // Tell the bridge directly where to scan from now on, then let it re-emit
  // the registry so the UI picker updates without a model reload or restart.
  try {
    bridgeWrite({ type: "set_models_dir", path: abs });
  } catch { }
  // Notify the renderer of the change so any open Settings panel can refresh.
  toUI("models-dir-changed", { dir: abs });
  return abs;
});

ipcMain.handle("pick-models-dir", async () => {
  const res = await dialog.showOpenDialog(win, {
    title: "Choose models folder",
    properties: ["openDirectory", "createDirectory"],
    defaultPath: getModelsDirImpl(app.getPath("userData")) || app.getPath("userData"),
  });
  if (res.canceled || !res.filePaths || !res.filePaths[0]) return null;
  return res.filePaths[0];
});

// ── paged SSD KV cache (v19) ─────────────────────────────────────
// Settings live in the same store as the rest of the app, under a single
// JSON object so an atomic `store-set` updates all three fields. The
// Python engine reads them via ATF_KV_SSD* env vars at bridge spawn.
// The live "how much is on disk right now" stat is computed here so the
// renderer can show the size in the right pane without round-tripping
// through Python. The Python engine would otherwise have to publish its
// own per-request event, which would couple the display to the request
// lifecycle (and miss the idle state entirely).

const KVSSD_DEFAULTS = Object.freeze({
  enabled: true,
  path: null,       // null -> ~/.cache/atf/kvpages
  hotPages: 256,
});

function readKvSsdSettings() {
  try {
    const saved = JSON.parse(fs.readFileSync(storePath("kvssd"), "utf8"));
    if (saved && typeof saved === "object") {
      return {
        enabled: saved.enabled !== false,                // default true
        path: typeof saved.path === "string" ? saved.path : null,
        hotPages: Number.isFinite(saved.hotPages) ? saved.hotPages : 256,
      };
    }
  } catch { }
  return { ...KVSSD_DEFAULTS };
}

function writeKvSsdSettings(next) {
  const tmp = storePath("kvssd") + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(next));
  fs.renameSync(tmp, storePath("kvssd"));
  return next;
}

function resolveKvSsdPath(p) {
  if (!p) {
    return path.join(os.homedir(), ".cache", "atf", "kvpages");
  }
  if (p.startsWith("~")) {
    return path.join(os.homedir(), p.slice(1));
  }
  return p;
}

function kvSsdWalkStats() {
  // Sum sizes + count files in the kvpages dir. Tolerates a missing dir
  // (returns zeros) and ignores unrelated files defensively.
  // gamma/v7 persistence (ATF_KV_SSD_PERSIST): sessions saved with
  // persist=True also drop a small "<key>.meta.npz" sidecar (fed_ids +
  // GDN state) next to their "<key>_b{N}.kvmm" page files -- counted here
  // too so the Runtime Cache panel's file count/size reflects everything
  // "Clear cache" is about to remove, and cleared below so a stale
  // manifest can never outlive the .kvmm files it points at.
  const root = resolveKvSsdPath(readKvSsdSettings().path);
  let totalBytes = 0, fileCount = 0, oldestMtime = 0, newestMtime = 0;
  try {
    const entries = fs.readdirSync(root, { withFileTypes: true });
    for (const e of entries) {
      if (!e.isFile()) continue;
      if (!e.name.endsWith(".kvmm") && !e.name.endsWith(".meta.npz")) continue;
      const full = path.join(root, e.name);
      try {
        const st = fs.statSync(full);
        totalBytes += st.size;
        fileCount += 1;
        if (st.mtimeMs > newestMtime) newestMtime = st.mtimeMs;
        if (oldestMtime === 0 || st.mtimeMs < oldestMtime) oldestMtime = st.mtimeMs;
      } catch { }
    }
  } catch { /* dir missing */ }
  return {
    path: root,
    exists: fileCount > 0,
    totalBytes,
    fileCount,
    oldestMtime,
    newestMtime,
  };
}

ipcMain.handle("get-kvssd-path", () => resolveKvSsdPath(readKvSsdSettings().path));

ipcMain.handle("set-kvssd-path", (_e, newPath) => {
  const s = readKvSsdSettings();
  s.path = newPath && newPath.trim() ? newPath.trim() : null;
  writeKvSsdSettings(s);
  return resolveKvSsdPath(s.path);
});

ipcMain.handle("pick-kvssd-path", async () => {
  const cur = resolveKvSsdPath(readKvSsdSettings().path);
  const res = await dialog.showOpenDialog(win, {
    title: "Choose paged SSD KV cache folder",
    properties: ["openDirectory", "createDirectory"],
    defaultPath: cur,
  });
  if (res.canceled || !res.filePaths || !res.filePaths[0]) return null;
  return res.filePaths[0];
});

ipcMain.handle("get-kvssd-stats", () => kvSsdWalkStats());

// v19: clear the on-disk paged-SSD KV cache. Called from the right-pane
// "Clear SSD cache" button. The Python side will recreate the directory
// and start spilling fresh pages on the next long context, so we don't
// need to mkdir it back here. We tolerate the dir not existing.
ipcMain.handle("clear-kvssd-cache", () => {
  const root = resolveKvSsdPath(readKvSsdSettings().path);
  let removed = 0;
  let bytes = 0;
  try {
    const entries = fs.readdirSync(root, { withFileTypes: true });
    for (const e of entries) {
      // gamma/v7 persistence: remove manifest sidecars alongside the page
      // files, or a cleared .kvmm set would leave an orphaned .meta.npz
      // that find_reusable() could still match against on the next run.
      if (!e.isFile() || (!e.name.endsWith(".kvmm") && !e.name.endsWith(".meta.npz"))) continue;
      const full = path.join(root, e.name);
      try {
        const st = fs.statSync(full);
        fs.unlinkSync(full);
        removed += 1;
        bytes += st.size;
      } catch { /* file vanished between readdir and unlink; skip */ }
    }
  } catch { /* dir missing is fine */ }
  return { removed, bytes, ...kvSsdWalkStats() };
});

// ── window state persistence ───────────────────────────────────────
function savedWindowState() {
  try {
    return JSON.parse(fs.readFileSync(storePath("window-state"), "utf8"));
  } catch {
    return null;
  }
}
function saveWindowState() {
  if (!win) return;
  try {
    const b = win.getNormalBounds();
    b.maximized = win.isMaximized();
    store_set_direct("window-state", b);
  } catch { }
}
function store_set_direct(key, value) {
  try {
    fs.writeFileSync(storePath(key), JSON.stringify(value));
  } catch { }
}

// ── window + menu ──────────────────────────────────────────────────
function sendMenu(action) {
  win?.webContents.send("menu-action", action);
}

function buildMenu() {
  const template = [
    {
      label: app.name,
      submenu: [
        { role: "about" },
        { type: "separator" },
        { label: "Settings…", accelerator: "Cmd+,", click: () => sendMenu("settings") },
        { type: "separator" },
        { role: "hide" }, { role: "hideOthers" }, { role: "unhide" },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "File",
      submenu: [
        { label: "New Chat", accelerator: "Cmd+N", click: () => sendMenu("new-chat") },
        { label: "Clear Conversation", accelerator: "Cmd+K", click: () => sendMenu("clear-chat") },
        { type: "separator" },
        { role: "close" },
      ],
    },
    { role: "editMenu" },
    {
      label: "View",
      submenu: [
        { label: "Toggle Theme", accelerator: "CmdOrCtrl+Shift+L", click: () => sendMenu("toggle-theme") },
        { type: "separator" },
        { role: "reload" }, { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" }, { role: "zoomIn" }, { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    { role: "windowMenu" },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  const saved = savedWindowState();
  const opts = {
    width: saved?.width ?? 1440,
    height: saved?.height ?? 900,
    minWidth: 640,
    minHeight: 480,
    maximizable: true,
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 18, y: 20 },
    backgroundColor: "#101014",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  };
  if (saved?.x != null && saved?.y != null) { opts.x = saved.x; opts.y = saved.y; }
  win = new BrowserWindow(opts);
  if (saved?.maximized) win.maximize();
  else if (!saved) win.maximize();

  win.loadFile(path.join(__dirname, "renderer", "index.html"));

  // security guards: never let model content navigate or open windows
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith("file://")) {
      e.preventDefault();
      if (/^https?:/i.test(url)) shell.openExternal(url);
    }
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    // v4: SVG preview button creates blob: URLs. The default Electron
    // handler denies them with action: "deny", so the preview was
    // silently failing. Open a new BrowserWindow with the blob URL
    // for the SVG content (or any http(s): URL via the OS browser).
    if (/^https?:/i.test(url)) {
      shell.openExternal(url);
      return { action: "deny" };
    }
    if (/^blob:/i.test(url)) {
      const preview = new BrowserWindow({
        width: 832, height: 632,
        title: "SVG preview",
        autoHideMenuBar: true,
        webPreferences: { contextIsolation: true, sandbox: true },
      });
      preview.loadURL(url);
      preview.on("closed", () => { /* no-op */ });
      return { action: "deny" };
    }
    return { action: "deny" };
  });

  win.webContents.on("did-finish-load", () => {
    console.log("[main] renderer did-finish-load; replaying",
      eventLog.length, "buffered events");
    flushEventLog();
    // CI/smoke hook: ATF_SMOKE=<ms> exits shortly after first render
    if (process.env.ATF_SMOKE) {
      setTimeout(() => { saveWindowState(); app.exit(0); },
        parseInt(process.env.ATF_SMOKE, 10) || 3000);
    }
  });
  win.webContents.on("console-message", (_e, level, message) => {
    console.log("[renderer]", message);
  });
}

app.whenReady().then(() => {
  // Dock icon at runtime: `run.sh` launches via `npm start`, so the real
  // process belongs to the Electron runtime bundle, not "ATF Chat.app" --
  // macOS would show the runtime's icon. Set ours explicitly (same icns
  // that is wired into the app bundle).
  if (process.platform === "darwin" && app.dock) {
    try {
      const { nativeImage } = require("electron");
      const iconPath = path.join(__dirname, "renderer", "icon.png");
      if (fs.existsSync(iconPath)) app.dock.setIcon(nativeImage.createFromPath(iconPath));
    } catch (e) { console.log("[icon]", e.message); }
  }
  buildMenu();
  // Bootstrap the venv + bridge script location before anything else so the
  // very first bridge spawn already has a writable socket path and a relocated
  // python interpreter.
  bootstrapAppVenv();
  startBridge();
  createWindow();
});

app.on("window-all-closed", () => app.quit());

app.on("before-quit", () => saveWindowState());

app.on("quit", () => {
  try { sock?.destroy(); } catch { }
  if (bridge) {
    try { bridge.removeAllListeners("exit"); bridge.kill(); } catch { }
    setTimeout(() => { try { bridge.kill("SIGKILL"); } catch { } }, 2000);
  }
});
