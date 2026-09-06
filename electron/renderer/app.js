/* ATF Chat — renderer application (v15)
 *
 * Structure:
 *   §1  dom refs & helpers          §6  sessions (persist, sidebar, actions)
 *   §2  markdown pipeline           §7  chat rendering + streaming
 *   §3  settings store              §8  message actions
 *   §4  engine lifecycle            §9  API server tab
 *   §5  model selection             §10 keyboard, menu, boot
 */
"use strict";

/* ── §1 dom refs & helpers ─────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const chat = $("chat"), input = $("input"), send = $("send"), stopBtn = $("stop");
const statusBadge = $("status"), meta = $("meta"), thinkingSel = $("thinking");
const modelSelect = $("model-select");
const welcomeEl = $("welcome"), loadingEl = $("loading");
const barFill = $("bar-fill"), loadLabel = $("load-label");
const paneThink = $("pane-think"), paneThinkLevel = $("pane-think-level"),
  ctxUsedEl = $("ctx-used"), ctxMeter = $("ctx-meter-fill");
const tierInfoEl = $("tier-info");
const sbConn = $("sb-conn"), sbModel = $("sb-model"), sbPhase = $("sb-phase"),
  sbGpu = $("sb-gpu"), sbTokens = $("sb-tokens"), sbPrefill = $("sb-prefill"),
  sbDecode = $("sb-decode"), sbTps = $("sb-tps"), sbTotal = $("sb-total"),
  sbTier = $("sb-tier"), sbCtx = $("sb-ctx");
const statusLog = $("status-log");
const apiToggle = $("api-toggle"), apiPortInput = $("api-port"), apiDot = $("api-dot"),
  apiState = $("api-state"), apiUrl = $("api-url"), apiLogEl = $("api-log");

const STATUS = { IDLE: "idle", LOADING: "loading", READY: "ready", ERROR: "error" };
const STATUS_LABELS = { idle: "Idle", loading: "Loading…", ready: "Ready", error: "Error" };

function h(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}

function logStatus(msg) {
  if (!statusLog) return;
  const lines = statusLog.textContent.split("\n");
  if (lines.length > 600) statusLog.textContent = lines.slice(-500).join("\n") + "\n";
  statusLog.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  statusLog.scrollTop = statusLog.scrollHeight;
}

let toastCount = 0;
function toast(msg, kind = "info", ms = 4200) {
  const t = h("div", `toast ${kind}`, msg);
  $("toasts").appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => t.remove(), 300);
  }, ms);
}

/* ── §2 markdown pipeline ──────────────────────────────────────── */
const md = window.markdownit({
  html: false, linkify: true, breaks: true,
});
// auto-highlight fenced blocks (with or without a language tag)
md.renderer.rules.fence = (tokens, idx) => {
  const token = tokens[idx];
  const lang = (token.info || "").trim().split(/\s+/)[0];
  let code = "";
  try {
    code = window.hljs
      ? (lang && window.hljs.getLanguage(lang)
        ? window.hljs.highlight(token.content, { language: lang }).value
        : window.hljs.highlightAuto(token.content).value)
      : md.utils.escapeHtml(token.content);
  } catch { code = md.utils.escapeHtml(token.content); }
  const label = lang || "code";
  // gamma/v4: SVG fenced blocks get a "Preview" button next to Copy
  // that opens the SVG in a new Electron window at native size. The
  // raw SVG text is stashed in svgPreviewStore (a Map keyed by a
  // unique id) so we don't have to base64-encode it into the HTML
  // attribute or risk DOMPurify stripping the markup.
  // gamma/v4: trigger the preview button on `svg` OR `xml`
  // (the model often tags SVG as ```xml because SVG is XML) and
  // additionally verify the content actually starts with <svg so
  // a random XML block doesn't get a preview button.
  const isSvg = (label.toLowerCase() === "svg" || label.toLowerCase() === "xml")
    && /<svg\b/i.test(token.content);
  let svgBtn = "";
  if (isSvg) {
    const sid = "svg-" + Math.random().toString(36).slice(2, 10);
    svgPreviewStore.set(sid, token.content);
    svgBtn = `<button class="copy-btn svg-preview-btn" data-svg-preview="${sid}" ` +
      `title="Open SVG in browser" aria-label="Open SVG preview">▶</button>`;
  }
  // gamma/v4: also offer a Preview button on HTML fenced blocks so the
  // rendered page can be opened in a new window (mirrors the SVG flow).
  // Trigger when the fence tag is html/htm, OR when no language was given
  // but the body clearly looks like HTML (starts with `<` and is not
  // `<svg`). The svg branch above always wins on overlap.
  const llabel = label.toLowerCase();
  const trimmed = (token.content || "").trimStart();
  const isHtml = !isSvg
    && (llabel === "html" || llabel === "htm"
      || ((!llabel || llabel === "code") && /^<[a-z!/]/i.test(trimmed) && !/^<svg\b/i.test(trimmed)));
  let htmlBtn = "";
  if (isHtml) {
    const hid = "html-" + Math.random().toString(36).slice(2, 10);
    htmlPreviewStore.set(hid, token.content);
    htmlBtn = `<button class="copy-btn html-preview-btn" data-html-preview="${hid}" ` +
      `title="Open HTML in browser" aria-label="Open HTML preview">▶</button>`;
  }
  return `<div class="code-pane${isSvg ? " svg-pane" : ""}${isHtml ? " html-pane" : ""}"><div class="code-header">` +
    `<span class="code-lang">${md.utils.escapeHtml(label)}</span>` +
    `<span class="code-actions">` +
    `<button class="copy-btn" data-copy title="Copy code">⎘</button>` +
    svgBtn + htmlBtn +
    `</span>` +
    `</div>` +
    `<pre class="code-pre"><code>${code}</code></pre></div>`;
};
function renderMarkdown(text) {
  return window.DOMPurify ? window.DOMPurify.sanitize(md.render(text || ""))
    : `<p>${escapeHtml(text || "")}</p>`;
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
// one delegated listener — copy buttons never need re-wiring per token
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-copy]");
  if (!btn) return;
  const code = btn.closest(".code-pane")?.querySelector("code");
  navigator.clipboard.writeText(code ? code.textContent : "").then(() => {
    btn.textContent = "✓";
    setTimeout(() => (btn.textContent = "⎘"), 1200);
  });
});

// SVG preview store: maps the unique data-svg-preview id to the raw
// SVG markup, so the click handler can open it without base64-encoding
// it into the HTML attribute (which DOMPurify would also strip).
const svgPreviewStore = new Map();
const htmlPreviewStore = new Map();
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-svg-preview]");
  if (!btn) return;
  const sid = btn.getAttribute("data-svg-preview");
  const svg = svgPreviewStore.get(sid);
  if (!svg) return;
  openSvgPreview(svg);
});
// gamma/v4: HTML preview click handler — mirrors the SVG flow above.
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-html-preview]");
  if (!btn) return;
  const hid = btn.getAttribute("data-html-preview");
  const html = htmlPreviewStore.get(hid);
  if (!html) return;
  openHtmlPreview(html);
});

// gamma/v4: open an SVG block in a new Electron BrowserWindow at
// the SVG's native viewBox size. Uses a blob URL (CSP-friendly: no
// data: scheme, no inline-script) and a minimal HTML wrapper that
// sets the SVG as the body content with sensible defaults.
function openSvgPreview(svgText) {
  // Pull viewBox or width/height from the SVG so the preview window
  // is sized to the artwork instead of a tiny 300x150 default.
  const m = svgText.match(/<svg[^>]*\bviewBox=["']([^"']+)["']/i);
  let w = 800, h = 600;
  if (m) {
    const parts = m[1].trim().split(/\s+/).map(Number);
    if (parts.length === 4 && parts.every(x => !isNaN(x))) {
      w = parts[2]; h = parts[3];
    }
  } else {
    const wm = svgText.match(/<svg[^>]*\bwidth=["'](\d+)/i);
    const hm = svgText.match(/<svg[^>]*\bheight=["'](\d+)/i);
    if (wm) w = parseInt(wm[1], 10);
    if (hm) h = parseInt(hm[1], 10);
  }
  // Clamp to something sensible so a 5000x5000 viewBox doesn't open
  // a giant window. Cap at 90% of the display.
  const cap = Math.max(320, Math.min(window.screen.availWidth - 80, 1600));
  if (w > cap) { h = Math.round(h * cap / w); w = cap; }
  const capH = Math.max(240, Math.min(window.screen.availHeight - 120, 1200));
  if (h > capH) { w = Math.round(w * capH / h); h = capH; }

  // Inline the SVG into a minimal HTML document. We escape the
  // closing </script> just in case the SVG contains that string
  // (it's not a valid SVG element, but defense in depth).
  const safeSvg = svgText.replace(/<\/script>/gi, "<\\/script>");
  const html = `<!doctype html><html><head><meta charset="utf-8">` +
    `<title>SVG preview</title><style>` +
    `html,body{margin:0;height:100%;background:#1a1a1f;display:flex;` +
    `align-items:center;justify-content:center}` +
    `svg{max-width:100%;max-height:100%;background:#fff;` +
    `box-shadow:0 0 24px rgba(0,0,0,0.5)}` +
    `</style></head><body>${safeSvg}</body></html>`;

  // Use a blob URL so the CSP allows it (the existing CSP has
  // default-src 'self', and blob: URLs created from same-origin
  // JS are considered same-origin for CSP purposes in Electron).
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  // Open in a new Electron window. The 'webPreferences' block is
  // intentionally minimal: we want the SVG to be able to load any
  // external references it contains (e.g. <image href="...">), so
  // we don't tighten the webSecurity default here.
  const wObj = window.open(url, "_blank",
    `width=${w + 32},height=${h + 32},menubar=no,toolbar=no,location=no,status=no`);
  // Revoke the blob URL after the new window has had time to load.
  // 60s is generous; most SVGs are <1s.
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  // If window.open was blocked (popup blocker), fall back to a
  // download hint -- the user can paste the URL into a new tab.
  if (!wObj) {
    console.warn("SVG preview: window.open was blocked; blob URL =", url);
  }
}

// gamma/v4: open an HTML fenced block in a new Electron window at a
// comfortable default size. The model's output may be a full document
// (starts with <!doctype / <html>) or just a fragment; either way we
// hand it to the browser verbatim. We only inject a minimal stylesheet
// for the chrome (a subtle dark backdrop so white pages don't strobe)
// and otherwise let the user's HTML own the page.
function openHtmlPreview(htmlText) {
  let w = 1024, h = 720;
  const capW = Math.max(320, Math.min(window.screen.availWidth - 80, 1600));
  const capH = Math.max(240, Math.min(window.screen.availHeight - 120, 1200));
  if (w > capW) w = capW;
  if (h > capH) h = capH;
  // If the model emitted a full document we render it as-is; otherwise
  // we wrap the fragment in a minimal HTML shell with a dark backdrop.
  const trimmed = (htmlText || "").trimStart();
  const isFullDoc = /^<!doctype\b/i.test(trimmed) || /^<html\b/i.test(trimmed);
  // Defang </script> in case the model emitted one inside <script>.
  const safe = (htmlText || "").replace(/<\/script>/gi, "<\\/script>");
  const shell = `<!doctype html><html><head><meta charset="utf-8">` +
    `<title>HTML preview</title>` +
    `<style>html,body{margin:0;min-height:100%;background:#1a1a1f;` +
    `color-scheme:light dark}</style></head><body>` +
    safe + `</body></html>`;
  const finalHtml = isFullDoc ? safe : shell;
  const blob = new Blob([finalHtml], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const features = `width=${w},height=${h},menubar=no,toolbar=no,location=no,status=no`;
  const wObj = window.open(url, "_blank", features);
  setTimeout(() => URL.revokeObjectURL(url), 30000);
  if (!wObj) {
    console.warn("HTML preview: window.open was blocked; blob URL =", url);
  }
}
$("copy-log-btn").addEventListener("click", () => {
  navigator.clipboard.writeText(statusLog.textContent).then(() => {
    $("copy-log-btn").textContent = "✓";
    setTimeout(() => ($("copy-log-btn").textContent = "⎘"), 1200);
  });
});
$("copy-api-log-btn").addEventListener("click", () => {
  navigator.clipboard.writeText(apiLogEl.textContent).then(() => {
    $("copy-api-log-btn").textContent = "✓";
    setTimeout(() => ($("copy-api-log-btn").textContent = "⎘"), 1200);
  });
});

/* ── §3 settings store ─────────────────────────────────────────── */
const SETTINGS_KEY = "settings";
const DEFAULTS = Object.freeze({
  system: "",
  // gamma/v4 (2026-09-02): maxTokens now tracks contextTokens
  // (no separate cap) so the generation can run to the full context
  // budget without an artificial truncation. Users can still lower
  // it via the Settings panel if they want a shorter reply.
  maxTokens: 65536,
  topP: 0.9,
  repeatPenalty: 1.15,
  // gamma/v4 (2026-09-02): default flipped to greedy (temperature=0)
  // for the fastest, most deterministic, and most reproducible decode.
  // Users can still pick any value in [0, 2] via the Settings panel.
  temperature: 0.0,
  contextTokens: 65536,
  thinking: "auto",
  mirrorThink: true,
  showTokens: true,
  confirmUnload: false,
  autoResume: false,
  lastModelId: null,
  activeSessionId: null,
  // v19: Paged SSD KV cache (gamma/v7). Default ON. Mirrors the
  // GenConfig.kv_ssd_* knobs in Python; the values are sent to the
  // bridge as ATF_KV_SSD* env vars by main.cjs at spawn time. The path
  // is null here (meaning "~/.cache/atf/kvpages" resolved in main) so
  // the placeholder is the same on every machine.
  kvSsdEnabled: true,
  kvSsdPath: null,         // null -> ~/.cache/atf/kvpages
  kvSsdHotPages: 256,      // ~16k tokens resident on GPU
  // appearance
  theme: "dark",            // "light" | "dark" | "auto"
  accent: "indigo",         // "indigo" | "violet" | "teal" | "green" | "amber" | "rose"
  fontSize: "comfortable",  // "compact" | "comfortable" | "large"
  density: "comfortable",   // "compact" | "comfortable" | "spacious"
  animations: true,
  // privacy / housekeeping
  telemetry: false,
  clearOnExit: false,
});
let settings = { ...DEFAULTS };
async function loadSettings() {
  const saved = await window.atf.storeGet(SETTINGS_KEY);
  if (saved && typeof saved === "object") {
    settings = { ...DEFAULTS, ...saved };
    // gamma/v4 (2026-09-02): one-time migration. Users who installed
    // a prior version (maxTokens default was 2048) have that value
    // persisted in their store. Bump it to the new default so the
    // upgrade is invisible -- no manual Settings change required.
    if (settings.maxTokens === 2048) {
      settings.maxTokens = DEFAULTS.maxTokens;
      saveSettingsSoon();
    }
    // v19: explicit migration so users who installed v18 or earlier
    // (when kvSsdEnabled did not exist) get a real boolean. The spread
    // above handles "field missing", but stored `null` from a partial
    // save would otherwise leak through.
    if (typeof settings.kvSsdEnabled !== "boolean") {
      settings.kvSsdEnabled = true;
      saveSettingsSoon();
    }
  }
}
const saveSettingsSoon = (() => {
  let t = null;
  return () => {
    clearTimeout(t);
    t = setTimeout(() => window.atf.storeSet(SETTINGS_KEY, settings), 250);
  };
})();


// Persist the visual theme under a dedicated localStorage key so the inline
// pre-paint script (in index.html) can apply it instantly on next launch.
function persistThemeCache() {
  try {
    localStorage.setItem(
      "atf-theme",
      JSON.stringify({
        mode: settings.theme,
        accent: settings.accent,
        fontSize: settings.fontSize,
        density: settings.density,
      })
    );
  } catch (e) { /* localStorage may be unavailable; fall through */ }
}

function resolveThemeMode(mode) {
  if (mode === "auto") {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light" : "dark";
  }
  return mode === "light" ? "light" : "dark";
}

function applyAppearance() {
  const root = document.documentElement;
  const resolved = resolveThemeMode(settings.theme);
  root.setAttribute("data-theme", resolved);
  root.setAttribute("data-accent", settings.accent || "indigo");
  root.setAttribute("data-density", settings.density || "comfortable");
  root.setAttribute("data-fontsize", settings.fontSize || "comfortable");
  root.setAttribute("data-animations", settings.animations ? "on" : "off");
  persistThemeCache();
}

// React to system theme changes when in "auto" mode.
if (window.matchMedia) {
  const mql = window.matchMedia("(prefers-color-scheme: light)");
  const onChange = () => { if (settings.theme === "auto") applyAppearance(); };
  if (mql.addEventListener) mql.addEventListener("change", onChange);
  else if (mql.addListener) mql.addListener(onChange);
}



/* ── §4 engine lifecycle ───────────────────────────────────────── */
let currentStatus = null;
let busy = false;            // a generation is in flight
let ready = false;           // a usable model is loaded
let currentModelId = null;
let loading = false;

function setStatus(next) {
  if (next === currentStatus) return;
  currentStatus = next;
  statusBadge.textContent = STATUS_LABELS[next] ?? next;
  statusBadge.className = `status ${next}`;
  if (sbPhase && !busy) sbPhase.textContent = next === STATUS.READY ? "ready" : next;
  logStatus(`status: ${next}`);
}

window.atf.onStatus((t) => {
  if (typeof t === "string" && t.includes("Engine ready")) setStatus(STATUS.READY);
  else if (currentStatus !== STATUS.READY && !ready) {
    if (loadLabel) loadLabel.textContent = t;
    if (t.includes("engine: loading")) {
      setStatus(STATUS.LOADING);
      loading = true;
    }
    if (!ready && !loading && t.includes("engine: idle"))
      setStatus(STATUS.IDLE);
  }
  logStatus(`engine: ${t}`);
});

window.atf.onProgress((p) => {
  if (currentStatus === STATUS.READY || ready) return;
  barFill.style.width = p.pct + "%";
  loadLabel.textContent = `${p.label} — ${Math.round(p.pct)}%`;
});

window.atf.onReady(() => {
  // bridge sends ready at startup AND after each model load; never demote READY
  if (currentStatus !== STATUS.READY) setStatus(STATUS.IDLE);
  hideCrashBanner();
  sbConn.textContent = "connected";
  sbConn.className = "sb-item on";
});

window.atf.onReloading(() => {
  ready = false;
  busy = false;
  if (_elapsedRaf) { cancelAnimationFrame(_elapsedRaf); _elapsedRaf = 0; }
  send.disabled = true;
  setStatus(STATUS.LOADING);
  meta.textContent = "";
});

window.atf.onCrashed(({ retryMs }) => {
  showCrashBanner(retryMs);
  sbConn.textContent = "restarting…";
  sbConn.className = "sb-item";
});

function showCrashBanner(retryMs) {
  const b = $("crash-banner");
  b.classList.remove("hidden");
  $("crash-text").textContent =
    `Engine crashed — restarting automatically${retryMs ? ` in ~${Math.round(retryMs / 1000)}s` : ""}. Your conversations are saved.`;
}
function hideCrashBanner() { $("crash-banner").classList.add("hidden"); }

window.atf.onError((m) => {
  finishMessage();
  const msg = typeof m === "string" ? m : m.message || "unknown error";
  if (m && m.busy) { toast("Engine busy — wait for the current task to finish.", "warn"); return; }
  meta.textContent = msg;
  setStatus(STATUS.ERROR);
  toast(msg, "error", 6500);
  logStatus(`error: ${msg}`);
  busy = false;
  if (_elapsedRaf) { cancelAnimationFrame(_elapsedRaf); _elapsedRaf = 0; }
  stopBtn.style.display = "none";
  send.disabled = !ready;
});

/* ── §5 model selection ────────────────────────────────────────── */
let modelsLoaded = false;
let selectedModelId = null;

window.atf.onModels((items) => {
  modelsLoaded = true;
  modelSelect.innerHTML = "";
  const ph = h("option");
  ph.value = ""; ph.disabled = true; ph.selected = true;
  ph.textContent = items.length ? "— select a model —" : "— no models found —";
  modelSelect.appendChild(ph);
  for (const m of items) {
    const o = h("option");
    o.value = m.id;
    o.textContent = `${m.id} · ${m.format} · ${m.size_gb} GB`;
    modelSelect.appendChild(o);
  }
  // v16: nothing auto-loads at startup. Remember the last model in the
  // placeholder text; it loads only when the user explicitly picks it.
  // (Restoring modelSelect.value would swallow the change event if the
  // user re-picks the same model, so the placeholder carries the hint.)
  const want = settings.lastModelId;
  if (want && [...modelSelect.options].some((o) => o.value === want)) {
    ph.textContent = `— select a model · last used: ${want} —`;
  }
});

modelSelect.addEventListener("change", () => selectModel(modelSelect.value));

function selectModel(id) {
  if (!id || id === selectedModelId) return;
  selectedModelId = id;
  settings.lastModelId = id;
  saveSettingsSoon();
  modelSelect.disabled = true;
  setStatus(STATUS.LOADING);
  loadingEl?.classList.remove("hidden");
  welcomeEl.style.display = "none";
  window.atf.loadModel(id);
}

window.atf.onModelLoaded((m) => {
  currentModelId = m.id;
  modelSelect.disabled = false;
  setUnloadEnabled(true);
  ready = true;
  loadingEl?.classList.add("hidden");
  welcomeEl.style.display = "none";
  setStatus(STATUS.READY);
  statusBadge.textContent = `Ready · ${m.id}`;
  sbModel.textContent = `model: ${m.id} (${m.format})`;
  send.disabled = false;
  input.focus();
  // v10: re-evaluate apiToggle (a model is now loaded → can serve)
  if (typeof setApiUI === "function") setApiUI(apiRunning);
  toast(`${m.id} loaded (${m.format})`, "ok", 2600);
});

window.atf.onModelUnloaded(() => {
  currentModelId = null;
  ready = false;
  modelSelect.disabled = false;
  modelSelect.value = "";          // allow re-selecting the same model
  selectedModelId = null;
  setUnloadEnabled(false);
  send.disabled = true;
  sbConn.className = "sb-item on";
  sbModel.textContent = "model: —";
  setStatus(STATUS.IDLE);
  // v10: disable Start Server button (no model = nothing to serve)
  if (typeof setApiUI === "function") setApiUI(apiRunning);
  toast("Model unloaded — GPU freed.", "ok");
});

const unloadBtn = $("unload-btn");

function setUnloadEnabled(on) { unloadBtn.disabled = !on; }

unloadBtn.addEventListener("click", async () => {
  if (!ready && !currentModelId) return;
  setUnloadEnabled(false);
  try {
    await window.atf.unloadModel();
  } finally {
    // re-enabled by onModelUnloaded / onModelLoaded if a model is present
    if (!currentModelId && !ready) setTimeout(() => setUnloadEnabled(false), 0);
  }
});

/* ── models management ─────────────────────────────────────────── */
let allModels = [];
let selectedModelsForDelete = new Set();

window.atf.onModels((items) => {
  allModels = items || [];
  renderModelsTable();
  refreshHfOnDiskStates();
});

function formatDate(dateStr) {
  if (!dateStr) return "—";
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (mins < 1) return "now";
    if (mins < 60) return `${mins}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 30) return `${days}d ago`;

    return date.toLocaleDateString();
  } catch {
    return dateStr;
  }
}

function renderModelsTable() {
  const tbody = $("models-tbody");
  tbody.innerHTML = "";

  if (!allModels || allModels.length === 0) {
    const row = tbody.insertRow();
    row.innerHTML = `<td colspan="6" class="models-loading">(no models found)</td>`;
    $("models-delete").classList.add("hidden");
    $("models-select-all").checked = false;
    $("models-right").classList.remove("active");
    selectedModelsForDelete.clear();
    return;
  }

  for (const model of allModels) {
    const row = tbody.insertRow();
    row.className = "model-row";
    row.dataset.modelId = model.id;

    const checkbox = h("input");
    checkbox.type = "checkbox";
    checkbox.className = "model-checkbox";
    checkbox.dataset.modelId = model.id;
    checkbox.addEventListener("change", (e) => {
      if (e.target.checked) {
        selectedModelsForDelete.add(model.id);
      } else {
        selectedModelsForDelete.delete(model.id);
      }
      updateDeleteButton();
      // Update select-all checkbox state
      const allChecked = document.querySelectorAll(".model-checkbox:checked").length === allModels.length;
      const someChecked = document.querySelectorAll(".model-checkbox:checked").length > 0;
      $("models-select-all").checked = allChecked;
      $("models-select-all").indeterminate = someChecked && !allChecked;
    });

    // v19: cells use CSS classes instead of inline style writes. The
    // base .models-table td { padding: 8px 10px } rule in style.css
    // already covers the padding; the row hover transition is on
    // .models-table tbody tr. We only add classes for the per-cell
    // tweaks (centered columns, clickable name, dimmed metadata).
    const checkCell = row.insertCell();
    checkCell.className = "models-cell-center";
    checkCell.appendChild(checkbox);

    const nameCell = row.insertCell();
    nameCell.className = "models-cell-name";
    nameCell.textContent = model.id;
    nameCell.title = model.id;

    const formatCell = row.insertCell();
    formatCell.textContent = model.format || "—";

    const sizeCell = row.insertCell();
    sizeCell.textContent = `${model.size_gb} GB`;

    const modifiedCell = row.insertCell();
    modifiedCell.className = "models-cell-meta";
    modifiedCell.textContent = formatDate(model.modified);

    const actionCell = row.insertCell();
    actionCell.className = "models-cell-center";

    const deleteBtn = h("button");
    deleteBtn.className = "model-delete-btn";
    deleteBtn.innerHTML = "🗑";
    deleteBtn.title = `Delete ${model.id}`;
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      selectedModelsForDelete.clear();
      selectedModelsForDelete.add(model.id);
      checkbox.checked = true;
      updateDeleteButton();
      deleteSelectedModels();
    });

    actionCell.appendChild(deleteBtn);

    // Click row to select/show details
    row.addEventListener("click", (e) => {
      if (e.target === deleteBtn || e.target.closest("input")) return;
      showModelDetails(model);
    });
  }

  // Select all checkbox  // Select all checkbox
  $("models-select-all").checked = false;
  $("models-select-all").indeterminate = false;
  $("models-select-all").addEventListener("change", (e) => {
    document.querySelectorAll(".model-checkbox").forEach((cb) => {
      cb.checked = e.target.checked;
      if (e.target.checked) {
        selectedModelsForDelete.add(cb.dataset.modelId);
      } else {
        selectedModelsForDelete.delete(cb.dataset.modelId);
      }
    });
    updateDeleteButton();
  });
}

function showModelDetails(model) {
  if (!model) {
    $("models-right").classList.remove("active");
    return;
  }

  $("detail-name").textContent = model.id;
  $("detail-format").textContent = model.format || "—";
  $("detail-size").textContent = `${model.size_gb} GB`;
  $("detail-path").textContent = model.path || "—";

  // Check if model is currently loaded
  if (currentModelId === model.id) {
    $("models-detail-loaded").classList.add("active");
  } else {
    $("models-detail-loaded").classList.remove("active");
  }

  $("models-right").classList.add("active");
}

function updateDeleteButton() {
  if (selectedModelsForDelete.size > 0) {
    $("models-delete").classList.remove("hidden");
    $("models-delete").textContent = `🗑 Delete (${selectedModelsForDelete.size})`;
  } else {
    $("models-delete").classList.add("hidden");
  }
}

$("models-refresh").addEventListener("click", () => {
  $("models-status").textContent = "Refreshing…";
  window.atf.getModels?.();
  setTimeout(() => {
    $("models-status").textContent = "";
  }, 1500);
});

/* ── models folder picker (Models tab) ───────────────────────────── */
const modelsFolderPath = $("models-folder-path"), modelsFolderPickBtn = $("models-folder-pick");

async function refreshModelsFolderDisplay() {
  if (!modelsFolderPath || !window.atf?.getModelsDir) return;
  try {
    const dir = await window.atf.getModelsDir();
    modelsFolderPath.value = dir || "(not set)";
    modelsFolderPath.title = dir || "";
  } catch {
    modelsFolderPath.value = "(unavailable)";
  }
}
refreshModelsFolderDisplay();

modelsFolderPickBtn?.addEventListener("click", async () => {
  const picked = await window.atf.pickModelsDir();
  if (!picked) return;
  modelsFolderPickBtn.disabled = true;
  try {
    const abs = await window.atf.setModelsDir(picked);
    modelsFolderPath.value = abs;
    modelsFolderPath.title = abs;
    toast(`Models folder set to ${abs}`, "ok", 2500);
  } catch (e) {
    toast(`Could not set models folder: ${e.message || e}`, "error", 4000);
  } finally {
    modelsFolderPickBtn.disabled = false;
  }
});

if (window.atf?.onModelsDirChanged) {
  window.atf.onModelsDirChanged((p) => {
    if (modelsFolderPath && p?.dir) {
      modelsFolderPath.value = p.dir;
      modelsFolderPath.title = p.dir;
    }
  });
}

$("models-delete").addEventListener("click", deleteSelectedModels);

async function deleteSelectedModels() {
  if (selectedModelsForDelete.size === 0) {
    toast("No models selected", "error");
    return;
  }

  const modelIds = Array.from(selectedModelsForDelete);
  const confirmed = confirm(
    `Delete ${modelIds.length} model${modelIds.length > 1 ? "s" : ""}? This cannot be undone.\n\n${modelIds.join("\n")}`
  );
  if (!confirmed) return;

  $("models-delete").disabled = true;
  $("models-status").textContent = "Deleting…";

  let deleted = 0;
  let failed = 0;

  for (const modelId of modelIds) {
    try {
      const result = await window.atf.deleteModel(modelId);
      if (result && result.success) {
        deleted++;
      } else {
        failed++;
      }
    } catch (err) {
      failed++;
    }
  }

  if (deleted > 0) {
    toast(`Deleted ${deleted} model${deleted > 1 ? "s" : ""}`, "ok");
    $("models-status").textContent = "Refreshing list…";
    selectedModelsForDelete.clear();
    setTimeout(() => {
      window.atf.getModels?.();
    }, 500);
  }

  if (failed > 0) {
    toast(`Failed to delete ${failed} model${failed > 1 ? "s" : ""}`, "error");
  }

  $("models-delete").disabled = false;
  setTimeout(() => {
    $("models-status").textContent = "";
  }, 1500);
}

/* ── Hugging Face model browser / downloader (Models tab) ───────── */
const hfQueryInput = $("hf-query"), hfSearchBtn = $("hf-search-btn"),
  hfResultsEl = $("hf-results"), hfStatusEl = $("hf-status");
const HF_AUTHOR = "amgadtewfik/atf";
const hfActiveDownloads = new Map();   // downloadId -> {btn, prog, fill, label, origText}
const hfRepoFilesCache = new Map();    // repoId -> files[]
const hfCardRefs = new Map();          // repoId -> {btn, fname}

// Local model ids (from atf.model_registry) are the path relative to the
// models root with the .atf suffix stripped, e.g. "Qwen3.5-9B-BF16" or
// "v3/Qwen3.5-9B-mlx4bit". HF repo files land flat at the models root
// (main.cjs hf-download-files), so comparing basenames (extension-
// stripped, case-insensitive) tells us whether a HF file is already local.
function hfLocalBasenames() {
  return new Set((allModels || []).map((m) => (m.id.split("/").pop() || "").toLowerCase()));
}
function hfIsOnDisk(filename) {
  if (!filename) return false;
  const base = filename.replace(/\.atf$/i, "").toLowerCase();
  return hfLocalBasenames().has(base);
}
// Re-check on-disk state for already-rendered HF cards (e.g. after a
// download completes or a model is deleted) without re-running the search.
function refreshHfOnDiskStates() {
  for (const [repoId, ref] of hfCardRefs) {
    if (ref.downloading) continue; // mid-download, leave its UI alone
    const onDisk = hfIsOnDisk(ref.fname);
    if (onDisk && !ref.btn.classList.contains("hf-on-disk")) {
      ref.btn.disabled = true;
      ref.btn.textContent = "✓ On disk";
      ref.btn.classList.add("hf-on-disk");
      ref.btn.title = "Already downloaded";
    } else if (!onDisk && ref.btn.classList.contains("hf-on-disk")) {
      ref.btn.disabled = false;
      ref.btn.textContent = ref.origText;
      ref.btn.classList.remove("hf-on-disk");
      ref.btn.title = "";
    }
  }
}

function humanBytes(n) {
  if (!n) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

async function hfSearch() {
  if (!hfResultsEl) return;
  const query = hfQueryInput.value.trim();
  hfResultsEl.innerHTML = `<div class="hf-loading">Searching huggingface.co/${HF_AUTHOR}…</div>`;
  hfStatusEl.textContent = "";
  hfSearchBtn.disabled = true;
  try {
    const res = await window.atf.hfSearchModels({ author: HF_AUTHOR, query });
    if (!res || !res.ok) {
      hfResultsEl.innerHTML = `<div class="hf-loading">Search failed: ${escapeHtml((res && res.error) || "unknown error")}</div>`;
      return;
    }
    renderHfResults(res.items);
  } catch (e) {
    hfResultsEl.innerHTML = `<div class="hf-loading">Search failed: ${escapeHtml(e.message || String(e))}</div>`;
  } finally {
    hfSearchBtn.disabled = false;
  }
}

function renderHfResults(items) {
  hfResultsEl.innerHTML = "";
  if (!items || !items.length) {
    hfResultsEl.innerHTML = `<div class="hf-loading">No models found.</div>`;
    return;
  }
  for (const m of items) {
    const card = h("div", "hf-model-card");
    const head = h("div", "hf-model-head");
    // Per-file items expose `path` and `repoId`; legacy items only had an
    // `id` equal to the repo id.
    const repoId = m.repoId || (m.id.includes("/") ? m.id.split("::")[0] : m.id);
    const filePath = m.path || null;
    const displayName = filePath ? filePath.split("/").pop() : m.id;
    const fname = filePath ? filePath.split("/").pop() : m.id.split("/").pop();
    const name = h("span", "hf-model-name", displayName);
    const sizeText = m.size ? ` · ${humanBytes(m.size)}` : "";
    const meta = h("span", "hf-model-meta", `${m.downloads ?? 0} downloads · ${m.likes ?? 0} likes${sizeText}`);
    const expandBtn = h("button", "hf-expand-btn", "Show files ▾");
    const origDlText = "⭳ Download Model";
    const dlAllBtn = h("button", "hf-download-btn hf-download-all", origDlText);
    const onDiskNow = hfIsOnDisk(fname);
    if (onDiskNow) {
      dlAllBtn.disabled = true;
      dlAllBtn.textContent = "✓ On disk";
      dlAllBtn.classList.add("hf-on-disk");
      dlAllBtn.title = "Already downloaded";
    }
    // Key hfCardRefs by repoId+file so per-file cards don't collide.
    const cardKey = filePath ? `${repoId}::${filePath}` : repoId;
    hfCardRefs.set(cardKey, { btn: dlAllBtn, fname, origText: origDlText, downloading: false });
    head.append(name, meta, expandBtn, dlAllBtn);

    const progRow = h("div", "hf-card-progress-row hidden");
    const prog = h("div", "hf-progress");
    const fill = h("div", "hf-progress-fill");
    prog.appendChild(fill);
    const label = h("span", "hf-progress-label", "");
    progRow.append(prog, label);

    const filesWrap = h("div", "hf-files-wrap hidden");
    card.append(head, progRow, filesWrap);
    hfResultsEl.appendChild(card);

    async function ensureFiles() {
      let files = hfRepoFilesCache.get(repoId);
      if (files) return files;
      const res = await window.atf.hfListFiles({ repoId });
      if (!res.ok) throw new Error(res.error || "could not list files");
      hfRepoFilesCache.set(repoId, res.files);
      return res.files;
    }

    let filesLoaded = false;
    expandBtn.addEventListener("click", async () => {
      const willShow = filesWrap.classList.contains("hidden");
      filesWrap.classList.toggle("hidden");
      expandBtn.textContent = willShow ? "Hide files ▴" : "Show files ▾";
      if (willShow && !filesLoaded) {
        filesLoaded = true;
        filesWrap.innerHTML = `<div class="hf-loading">Loading files…</div>`;
        try {
          const files = await ensureFiles();
          renderHfFiles(filesWrap, repoId, files);
        } catch (e) {
          filesWrap.innerHTML = `<div class="hf-loading">Could not list files: ${escapeHtml(e.message)}</div>`;
        }
      }
    });

    dlAllBtn.addEventListener("click", () => {
      if (dlAllBtn.disabled) return;
      dlAllBtn.disabled = true;
      const ref = hfCardRefs.get(cardKey);
      if (ref) ref.downloading = true;
      // Download straight from
      // https://huggingface.co/amgadtewfik/atf/resolve/main/<path> instead
      // of round-tripping through the file-list API.
      progRow.classList.remove("hidden");
      // Per-file card already knows which path to grab; legacy items fall
      // back to the basename of the repo id.
      const dlPath = filePath || fname;
      startHfDownload(repoId, [{ path: dlPath, size: m.size || 0 }], dlAllBtn, prog, fill, label);
    });
  }
}

function renderHfFiles(container, repoId, files) {
  container.innerHTML = "";
  if (!files || !files.length) {
    container.innerHTML = `<div class="hf-loading">(no files)</div>`;
    return;
  }
  for (const f of files) {
    const row = h("div", "hf-file-row");
    const fname = h("span", "hf-file-name", f.path);
    fname.title = f.path;
    const fsize = h("span", "hf-file-size", humanBytes(f.size));
    const dlBtn = h("button", "hf-download-btn", "⭳ Download");
    if (hfIsOnDisk(f.path)) {
      dlBtn.disabled = true;
      dlBtn.textContent = "✓ On disk";
      dlBtn.classList.add("hf-on-disk");
      dlBtn.title = "Already downloaded";
    }
    const prog = h("div", "hf-progress hidden");
    const fill = h("div", "hf-progress-fill");
    prog.appendChild(fill);
    const label = h("span", "hf-progress-label", "");
    dlBtn.addEventListener("click", () => {
      if (dlBtn.disabled) return;
      prog.classList.remove("hidden");
      startHfDownload(repoId, f, dlBtn, prog, fill, label);
    });
    row.append(fname, fsize, dlBtn, prog, label);
    container.appendChild(row);
  }
}

function startHfDownload(repoId, files, btn, prog, fill, label) {
  const fileList = Array.isArray(files) ? files : [files];
  const downloadId = `hf${Date.now()}${Math.random().toString(36).slice(2, 6)}`;
  const origText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Downloading…";
  fill.style.width = "0%";
  label.textContent = "";
  hfActiveDownloads.set(downloadId, { btn, prog, fill, label, origText });
  window.atf.hfDownloadFiles({ repoId, files: fileList, downloadId }).then((res) => {
    if (!res || !res.ok) {
      hfActiveDownloads.delete(downloadId);
      btn.disabled = false;
      btn.textContent = origText;
      toast(`Download failed: ${(res && res.error) || "unknown error"}`, "error", 4500);
    }
  });
}

window.atf.onHfDownloadProgress?.((p) => {
  const d = hfActiveDownloads.get(p.downloadId);
  if (!d) return;
  const fraction = ((p.fileIndex || 0) + (p.totalBytes ? p.receivedBytes / p.totalBytes : 0)) / (p.fileCount || 1);
  const pct = Math.max(0, Math.min(100, fraction * 100));
  d.fill.style.width = `${pct}%`;
  const bytesText = p.totalBytes ? `${humanBytes(p.receivedBytes)} / ${humanBytes(p.totalBytes)}` : humanBytes(p.receivedBytes);
  d.label.textContent = p.fileCount > 1 ? `file ${(p.fileIndex || 0) + 1}/${p.fileCount} · ${bytesText}` : bytesText;
});

window.atf.onHfDownloadLog?.((line) => {
  if (hfStatusEl) hfStatusEl.textContent = line;
});

window.atf.onHfDownloadDone?.((r) => {
  const d = hfActiveDownloads.get(r.downloadId);
  if (!d) return;
  hfActiveDownloads.delete(r.downloadId);
  // Mark every card that was part of this download as no-longer-downloading.
  if (Array.isArray(r.files)) {
    for (const f of r.files) {
      const key = `${r.repoId}::${f.path}`;
      const ref = hfCardRefs.get(key);
      if (ref) ref.downloading = false;
    }
  }
  d.btn.disabled = false;
  if (r.ok) {
    d.btn.textContent = "✓ Downloaded";
    d.label.textContent = "done";
    const fname = (Array.isArray(r.files) && r.files[0]?.path || r.repoId).split("/").pop();
    toast(`Downloaded ${fname}`, "ok", 3000);
    // getModels() -> onModels -> refreshHfOnDiskStates() will flip this
    // card to the persistent disabled "✓ On disk" state once the registry
    // scan confirms the file; this timeout just clears the transient label
    // if that round-trip is slow.
    window.atf.getModels?.();
    setTimeout(() => { if (!d.btn.classList.contains("hf-on-disk")) d.btn.textContent = d.origText; }, 3000);
  } else {
    d.btn.textContent = d.origText;
    toast(`Download failed: ${r.error || "unknown error"}`, "error", 4500);
  }
});

if (hfSearchBtn) {
  hfSearchBtn.addEventListener("click", hfSearch);
  hfQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); hfSearch(); }
  });
  // run an initial search the first time the Models tab is opened
  let hfAutoSearched = false;
  document.querySelector('.tab[data-tab="models"]')?.addEventListener("click", () => {
    if (!hfAutoSearched) { hfAutoSearched = true; hfSearch(); }
  });
}

/* ── §6 sessions ───────────────────────────────────────────────── */
// Session shape: {id,title,created,updated,messages:[{role,content,thinking,tier,truncated,stats}]}
let sessions = [];
let activeSessionId = null;

// v19: chat sessions live in a SQLite DB on the main process (see
// main.cjs openSessionDb). The renderer still keeps a JS array of
// sessions for the sidebar and the streaming flow, but persistence
// goes through per-session IPCs instead of one giant storeSet.
// per-session saves mean a stale empty array can never nuke the DB
// (the cause of the v19 loss of six sessions from the JSON store).
async function loadSessions() {
  // Pull the metadata list first; the sidebar can render right away.
  // We lazy-load the messages for the active session so a user with
  // hundreds of stored chats does not pay for them all on boot.
  //
  // v19+ sessions live in SQLite (see main.cjs openSessionDb). If the
  // better-sqlite3 native binding wasn't rebuilt for Electron, this
  // IPC throws "Database is not a constructor" — and silently dropping
  // the error here makes every chat appear empty (the sidebar shows
  // titles from memory, but the body has zero messages and renders the
  // welcome screen). Surface a loud toast instead so the user can run
  // `cd electron && node scripts/rebuild-native.mjs --force` to fix.
  let list = [];
  try {
    list = await window.atf.sessions.list();
  } catch (e) {
    console.error("[sessions] list() failed:", e);
    toast(
      "Chat sessions could not be loaded — better-sqlite3 was built for the wrong Node version. " +
      "Run: cd electron && node scripts/rebuild-native.mjs --force",
      "error",
      15000
    );
  }
  list = list || [];
  sessions = list.map((row) => ({
    id: row.id,
    title: row.title,
    created: row.created,
    updated: row.updated,
    // messages are filled in on openSession() when the row is opened.
    // The `messagesLoaded` flag tells persistSessions() it is safe to
    // overwrite this session's messages on disk — without it, the save
    // would DELETE every persisted message and INSERT zero, wiping the
    // session's history. See the v17+ openSession fix and the persistSessions
    // guard below.
    messages: [],
    messagesLoaded: false,
  }));
  // Pull the active session's full history (if any). If it does not
  // exist in the DB, fall through to creating a new one below.
  activeSessionId = settings.activeSessionId;
  if (activeSessionId && sessions.some((s) => s.id === activeSessionId)) {
    const full = await window.atf.sessions.get(activeSessionId);
    // v17+ DEBUG: surface the JSON we just got from SQLite so the
    // user can see what messages survived the round-trip.
    console.log(`[renderer] loadSessions: get(${activeSessionId}) ->`, JSON.stringify({
      id: full && full.id,
      title: full && full.title,
      msg_count: full && full.messages ? full.messages.length : 0,
      first_msg: full && full.messages && full.messages[0] ? {
        role: full.messages[0].role,
        content_preview: (full.messages[0].content || "").slice(0, 120),
      } : null,
    }, null, 2));
    if (full) {
      const idx = sessions.findIndex((s) => s.id === activeSessionId);
      sessions[idx] = { ...full, messagesLoaded: true };
    }
  } else {
    activeSessionId = null;
  }
  if (!sessions.length) createSession({ silent: true });
  if (!activeSessionId) activeSessionId = sessions[0].id;
  renderSessionList();
  openSession(activeSessionId, { silent: true });
}

function persistSessions() {
  // Per-session upsert. We only save the session objects that are in
  // memory; nothing on disk is overwritten from a stale empty list.
  // Errors are caught and logged so a single bad write does not stop
  // the streaming flow.
  settings.activeSessionId = activeSessionId;
  saveSettingsSoon();
  for (const s of sessions) {
    if (!s || !s.id) continue;
    // CRITICAL: skip sessions whose message history has not been
    // loaded yet from SQLite. The save path in main.cjs does
    // DELETE+INSERT on the messages table — saving a session with
    // an empty in-memory `messages` array wipes the persisted history.
    // v17+ openSession sets messagesLoaded=true after the lazy
    // fetch, and createSession / clearConversation / onDone all leave
    // it true. The only place a session has messagesLoaded=false is
    // when it's a sidebar entry that has never been opened this run.
    if (s.messagesLoaded === false) {
      console.log(`[renderer] persistSessions: SKIP save(${s.id}) — messages not loaded yet`);
      continue;
    }
    // v17+ DEBUG: surface the in-memory state right before we save so
    // we can see exactly what the renderer is sending to main.cjs.
    console.log(`[renderer] persistSessions: save(${s.id}) title=${JSON.stringify(s.title)} msg_count=${Array.isArray(s.messages) ? s.messages.length : 0}`, s.messages && s.messages.map((m) => ({ role: m.role, len: (m.content || "").length, preview: (m.content || "").slice(0, 60) })));
    window.atf.sessions.save(s)
      .then(() => { persistSessions._warned = false; })
      .catch((e) => {
        console.warn(`[sessions] save(${s.id}) failed:`, e);
        // Same SQLite-NMV failure mode as loadSessions() / openSession().
        // A silent console.warn here is exactly what hid the "chats
        // never persist, every restart starts empty" data-loss bug for
        // so long. Surface it once (suppress further toasts until the
        // next successful save) so the user can see why their chats
        // vanished and run scripts/rebuild-native.mjs --force.
        if (!persistSessions._warned) {
          persistSessions._warned = true;
          toast(
            "Chat persistence is broken — better-sqlite3 was built for the wrong Node version. " +
            "New messages will be lost on restart. Run: cd electron && node scripts/rebuild-native.mjs --force",
            "error",
            15000
          );
        }
      });
  }
}

function activeSession() { return sessions.find((s) => s.id === activeSessionId); }

function createSession({ silent } = {}) {
  const s = {
    id: `s${Date.now()}${Math.random().toString(36).slice(2, 6)}`,
    title: "New chat",
    created: Date.now(),
    updated: Date.now(),
    messages: [],
  };
  sessions.unshift(s);
  activeSessionId = s.id;
  renderSessionList();
  openSession(s.id, { silent: true });
  if (!silent) { persistSessions(); input.focus(); }
  return s;
}

function deleteSession(id) {
  const idx = sessions.findIndex((s) => s.id === id);
  if (idx < 0) return;
  if (!confirm("Delete this conversation? This cannot be undone.")) return;
  sessions.splice(idx, 1);
  // v19: tell main to drop the row immediately. persistSessions()
  // below also runs, but only saves in-memory sessions — without
  // this explicit delete the deleted row would linger in SQLite.
  window.atf.sessions.delete(id).catch((e) => {
    console.warn(`[sessions] delete(${id}) failed:`, e);
  });
  if (activeSessionId === id) {
    if (sessions.length) openSession(sessions[Math.max(0, idx - 1)].id);
    else createSession({ silent: true });
  }
  persistSessions();
  renderSessionList();
}

function renameSession(id) {
  const s = sessions.find((x) => x.id === id);
  if (!s) return;
  const name = prompt("Conversation name:", s.title);
  if (name == null) return;
  s.title = name.trim() || s.title;
  persistSessions();
  renderSessionList();
}

function exportSession(id, fmt) {
  const s = sessions.find((x) => x.id === id);
  if (!s) return;
  let body;
  if (fmt === "json") body = JSON.stringify(s, null, 2);
  else body = s.messages.map((m) =>
    m.role === "user"
      ? `## You\n\n${m.content}\n`
      : `## ATF${m.tier ? ` (${m.tier.badge} ${m.tier.name})` : ""}\n\n${m.content}\n`
  ).join("\n---\n\n");
  const blob = new Blob([body], { type: fmt === "json" ? "application/json" : "text/markdown" });
  const a = h("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${s.title.replace(/[^\w\- ]+/g, "").slice(0, 40) || "chat"}.${fmt}`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function renderSessionList() {
  const list = $("session-list");
  list.innerHTML = "";
  for (const s of sessions) {
    const item = h("div", `session-item ${s.id === activeSessionId ? "active" : ""}`);
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(s.id === activeSessionId));
    const title = h("span", "session-title", s.title);
    title.title = s.title;
    const del = h("button", "session-btn", "✕");
    del.title = "Delete";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(s.id); });
    const ren = h("button", "session-btn", "✎");
    ren.title = "Rename";
    ren.addEventListener("click", (e) => { e.stopPropagation(); renameSession(s.id); });
    item.append(title, ren, del);
    item.tabIndex = 0;
    item.addEventListener("click", () => openSession(s.id));
    list.appendChild(item);
  }
  // right-click exports the session (Markdown / JSON)
  list.oncontextmenu = (e) => {
    const it = e.target.closest(".session-item");
    if (!it) return;
    e.preventDefault();
    const idx = [...list.children].indexOf(it);
    const s = sessions[idx];
    exportSession(s.id, confirm("Export as JSON?\n(Cancel = Markdown)") ? "json" : "md");
  };
}

async function openSession(id, { silent } = {}) {
  // v19: refuse to switch sessions while a generation is in flight.
  // rebuildChatFromSession() wipes #chat and rebuilds it from the
  // target session's saved messages, which destroys the in-DOM
  // streaming bubble (the model keeps writing into the original
  // session's `turn`, but the user can no longer see it). Same
  // pattern as the tab guard: stop / wait, then switch.
  if (busy && id !== activeSessionId) {
    toast("Generation in progress — stop it (Esc) or wait, then switch sessions.", "warn");
    return;
  }
  // v19 (SQLite): sessions that were never opened during this app
  // run may not have their message history loaded yet (loadSessions
  // only pulls the full history for the initially-active session).
  // Fetch on demand so opening an old chat from the sidebar works
  // without a restart.
  //
  // CRITICAL: await the fetch BEFORE calling activeSessionId = id or
  // persistSessionsSoon2(). The previous non-awaited version was a
  // data-loss bug: it scheduled a save 400ms later while the in-memory
  // session still had an empty messages[]. The save would DELETE every
  // persisted message for that session and INSERT zero in their place,
  // wiping real chat history. The await guarantees the save sees the
  // fully-populated messages[].
  const sess = sessions.find((s) => s.id === id);
  if (sess && (!Array.isArray(sess.messages) || sess.messages.length === 0)) {
    try {
      const full = await window.atf.sessions.get(id);
      if (full) {
        const idx = sessions.findIndex((s) => s.id === id);
        if (idx >= 0) sessions[idx] = { ...full, messagesLoaded: true };
      }
    } catch (e) {
      // Same SQLite-NMV failure mode as loadSessions(). Surface it;
      // a silent console.warn here is exactly what hid the "click a
      // chat, see the welcome screen" bug for so long.
      console.error(`[sessions] get(${id}) failed:`, e);
      toast(
        "Could not open this chat — better-sqlite3 was built for the wrong Node version. " +
        "Run: cd electron && node scripts/rebuild-native.mjs --force",
        "error",
        12000
      );
    }
  }
  activeSessionId = id;
  renderSessionList();
  rebuildChatFromSession();
  updateCtxUsage();
  if (!silent) persistSessionsSoon2();
}
const persistSessionsSoon2 = (() => {
  let t = null;
  return () => { clearTimeout(t); t = setTimeout(persistSessions, 400); };
})();

/* ── §7 chat rendering + streaming ─────────────────────────────── */
let curReqId = null;        // correlation id of the in-flight generate
let turn = null;            // live turn view-model while generating
let sessionTokens = 0;      // real prompt tokens reported by the engine
let rafPending = false;
let genStart = 0, firstTokenAt = 0, turnTokens = 0;
let thinkStartedAt = 0, thinkEndedAt = 0;
let _elapsedRaf = 0;  // requestAnimationFrame id for the elapsed-time ticker

function newTurnView() {
  welcomeEl.remove?.();
  welcomeEl.style.display = "none";
  const msgEl = h("div", "msg assistant");
  // Show the active model name in place of the static "ATF" label.
  const modelLabel = currentModelId || "ATF";
  msgEl.appendChild(h("div", "who", modelLabel));

  // live phase indicator: prefilling -> thinking -> generating, hidden once
  // real content (thinking text or the answer) starts appearing on screen
  const phase = h("div", "phase-indicator prefill");
  const phaseDots = h("div", "phase-dots");
  phaseDots.append(h("span"), h("span"), h("span"));
  const phaseLabel = h("span", "phase-label", "Prefilling…");
  const phaseBar = h("div", "phase-bar");
  const phaseBarFill = h("div", "phase-bar-fill");
  phaseBar.appendChild(phaseBarFill);
  // small meta line: tok/s + elapsed, fades in during prefill so the user
  // sees the engine is actually working between progress events
  const phaseMeta = h("span", "phase-meta", "— tok/s · 0.0s");
  phase.append(phaseDots, phaseLabel, phaseBar, phaseMeta);
  msgEl.appendChild(phase);

  // Per-turn metrics row: pill chips (tok/s · tokens · seconds · stop reason).
  // Hidden while streaming, populated in onDone() and persisted with the message.
  const turnStats = h("div", "turn-stats hidden");

  const wrap = h("details", "think-wrap");
  wrap.open = false;
  wrap.style.display = "none";
  const thinkSummary = h("summary", "thinking-summary");
  thinkSummary.append(h("span", "think-chev", "›"), h("span", "think-text", "Thinking for 0.00 seconds"));
  wrap.appendChild(thinkSummary);
  const thinkPre = h("pre", "think-pre");
  wrap.appendChild(thinkPre);
  const answer = h("div", "bubble answer");
  const actions = h("div", "msg-actions");
  actions.style.display = "none";
  msgEl.append(wrap, answer, turnStats, actions);
  chat.appendChild(msgEl);
  scrollChat(true);
  return {
    msgEl, wrap, thinkPre, thinkSummary, answer, actions,
    phase, phaseLabel, phaseBar, phaseBarFill, phaseMeta,
    turnStats, modelId: modelLabel,
    thinkBuf: "", ansBuf: "", thinkOpen: false, answerStarted: false
  };
}

function scrollChat(force) {
  const nearBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
  if (nearBottom || force) chat.scrollTop = chat.scrollHeight;
}

function scheduleRender() {
  if (rafPending || !turn) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; renderTurn(); });
}

function renderTurn() {
  if (!turn) return;
  const v = computeVisibility(turn);
  turn.thinkPre.textContent = v.thinkText;
  turn.wrap.style.display = v.showThink ? "" : "none";
  // during the stream use the streaming-safe renderer; final pass gets full
  // markdown once the fence structure is stable
  turn.answer.innerHTML = busy ? renderMarkdownStream(v.answerText)
    : renderMarkdown(v.answerText);
  if (settings.mirrorThink && v.thinkText && paneThink) {
    const stick = paneThink.scrollTop + paneThink.clientHeight >= paneThink.scrollHeight - 24;
    paneThink.textContent = turn.thinkBuf;
    if (stick) paneThink.scrollTop = paneThink.scrollHeight;
  }
  scrollChat(false);
}

// streaming-safe markdown: an unclosed fence becomes a live code pane so
// users can read/copy code as it arrives
function renderMarkdownStream(text) {
  const openFence = /```([\w-]+)?\s*\n([\s\S]*)$/.exec(text);
  let base = text, tailHtml = "";
  if (openFence) {
    base = text.slice(0, openFence.index);
    const lang = (openFence[1] || "").trim().toLowerCase() || "code";
    const code = openFence[2] || "";
    // gamma/v4: same content-aware check as the closed-fence path.
    // Trigger on svg OR xml (models often mis-tag SVG as xml) and
    // verify the body actually starts with <svg.
    const isSvg = (lang === "svg" || lang === "xml")
      && /<svg\b/i.test(code);
    let svgBtn = "";
    if (isSvg) {
      const sid = "svg-" + Math.random().toString(36).slice(2, 10);
      svgPreviewStore.set(sid, code);
      svgBtn = `<button class="copy-btn svg-preview-btn" data-svg-preview="${sid}" ` +
        `title="Open SVG in browser" aria-label="Open SVG preview">▶</button>`;
    }
    // gamma/v4: HTML preview mirror of the SVG flow. Same content-aware
    // rule as the closed-fence path so the two never disagree.
    const ll = (lang || "").toLowerCase();
    const trimmed = (code || "").trimStart();
    const isHtml = !isSvg
      && (ll === "html" || ll === "htm"
        || ((!ll || ll === "code") && /^<[a-z!/]/i.test(trimmed) && !/^<svg\b/i.test(trimmed)));
    let htmlBtn = "";
    if (isHtml) {
      const hid = "html-" + Math.random().toString(36).slice(2, 10);
      htmlPreviewStore.set(hid, code);
      htmlBtn = `<button class="copy-btn html-preview-btn" data-html-preview="${hid}" ` +
        `title="Open HTML in browser" aria-label="Open HTML preview">▶</button>`;
    }
    tailHtml =
      `<div class="code-pane streaming${isSvg ? " svg-pane" : ""}${isHtml ? " html-pane" : ""}"><div class="code-header">` +
      `<span class="code-lang">${escapeHtml(lang)}</span>` +
      `<span class="code-actions">` +
      `<button class="copy-btn" data-copy>⎘</button>` +
      svgBtn + htmlBtn +
      `</span>` +
      `</div>` +
      `<pre class="code-pre"><code>${escapeHtml(code)}</code></pre></div>`;
  }
  return window.DOMPurify
    ? DOMPurify.sanitize(md.render(base) + tailHtml,
      { ADD_ATTR: ["data-copy", "data-svg-preview", "data-html-preview"] })
    : escapeHtml(base) + tailHtml;
}

/* generation flow */
function submit(textOverride) {
  const text = (textOverride ?? input.value).trim();
  if (!text || busy) return;
  if (!ready) { toast("Load a model first.", "warn"); return; }

  const s = activeSession() || createSession({ silent: true });
  s.messages.push({ role: "user", content: text });
  if (s.title === "New chat") { s.title = sessionTitle(text); renderSessionList(); }
  appendUserBubble(text);

  startGeneration();
  input.value = "";
  input.style.height = "auto";
}

function startGeneration() {
  const s = activeSession();
  // v19: re-assert the Chat tab so the user always sees the response they
  // just asked for, even if they were on the Dashboard / Models / API
  // tabs when they hit Submit. Without this, the streaming bubble lives
  // inside #tab-inference and would render into a hidden tab.
  activateTab("inference");
  busy = true;
  send.disabled = true;
  stopBtn.style.display = "inline-block";
  meta.textContent = "";
  if (paneThink) paneThink.textContent = "(thinking…)";

  turn = newTurnView();
  // v17+ thinking selector value: capture here so the turn stats pill
  // (and the persisted message stats) can show which reasoning level
  // the user asked for, even if they change the dropdown after submit.
  // Read from the <select> directly — settings.thinking is the
  // user-visible default but may lag a tick behind during a rapid
  // dropdown change.
  if (turn) turn.thinking = (typeof thinkingSel !== "undefined" && thinkingSel) ? thinkingSel.value : "auto";
  // Highlight the right-pane pill so the user can see at a glance
  // that a generation is using this reasoning effort right now.
  if (paneThinkLevel) paneThinkLevel.classList.add("active");
  genStart = performance.now();
  firstTokenAt = 0;
  turnTokens = 0;
  thinkStartedAt = 0;
  thinkEndedAt = 0;
  setPhase("prefill…");
  // gamma/v4: start the elapsed-time ticker so the meta line
  // counts up from the moment the user clicks Submit, not from
  // the first onProgress/onToken event. tickElapsed self-cancels
  // when busy flips false in finishMessage() / error / reload.
  if (_elapsedRaf) cancelAnimationFrame(_elapsedRaf);
  _elapsedRaf = requestAnimationFrame(tickElapsed);
  // reset prefill animation bookkeeping so the new turn starts from a
  // clean indeterminate state instead of inheriting the last turn's
  // smoothing / completion-pulse flags
  _prefillLastRender = 0;
  _prefillRenderQueued = false;
  _prefillLastDone = 0;
  _prefillLastDoneAt = 0;
  _prefillSmoothingFrom = 0;
  _prefillSmoothingTo = 0;
  _prefillCompleteFlashed = false;
  if (turn && turn.phaseBar) {
    turn.phaseBar.classList.remove("indeterminate", "complete");
    turn.phaseBarFill.style.width = "0%";
    if (turn.phaseMeta) turn.phaseMeta.textContent = "";
  }

  const history = [];
  for (let i = 0; i < s.messages.length - 1; i++) {
    if (s.messages[i].role === "user" && s.messages[i + 1].role === "assistant") {
      history.push({ user: s.messages[i].content, assistant: s.messages[i + 1].content });
    }
  }

  window.atf.generate({
    message: s.messages[s.messages.length - 1].content,
    history,
    system: settings.system || undefined,
    max_tokens: settings.maxTokens,
    temperature: settings.temperature,
    top_p: settings.topP,
    repeat_penalty: settings.repeatPenalty,
    thinking: thinkingSel.value,
    context_tokens: settings.contextTokens,
  }).then((id) => { curReqId = id || null; });
  persistSessionsSoon2();
}

window.atf.onToken((t) => {
  if (!turn || !matchesCurrent(t)) return;
  if (turnTokens === 0) {
    firstTokenAt = performance.now();
    if (turn.phase) turn.phase.classList.add("hidden");
  }
  if (turn.thinkOpen) turn.thinkBuf += t.text; else turn.ansBuf += t.text;
  turnTokens++;
  setPhaseIfBusy("decoding…");
  scheduleRender();
  const el = (performance.now() - genStart) / 1000;
  const decodeEl = Math.max(0.001, (performance.now() - firstTokenAt) / 1000);
  const tps = turnTokens / decodeEl;
  meta.textContent = `${turnTokens} tokens · ${el.toFixed(1)}s · ${tps.toFixed(2)} tok/s`;
  sbTokens.textContent = String(turnTokens);
  sbTps.textContent = `${tps.toFixed(2)} tok/s`;
  sbDecode.textContent = `${decodeEl.toFixed(1)}s`;
  sbTotal.textContent = `${el.toFixed(1)}s`;
});

function matchesCurrent(ev) {
  return curReqId == null || ev.id == null || ev.id === curReqId;
}

function setPhaseIfBusy(t) { if (busy) sbPhase.textContent = t; }
function setPhase(t) { sbPhase.textContent = t; }

// gamma/v4 (2026-09-02): tick the chat-message elapsed-time
// display every animation frame from submit onward. Before this,
// the meta line (#meta, below the assistant's bubble) only updated
// onProgress (during prefill) and onToken (during decode), so the
// user saw a frozen "0.0s" until the first event arrived. Now the
// elapsed clock in the CHAT starts the instant the user clicks
// Submit. The top status-bar cells (sb-tps, sb-total, etc.) are
// only updated on real events -- this ticker is the per-turn timer
// the user actually watches.
function tickElapsed() {
  if (!genStart) return;
  const el = (performance.now() - genStart) / 1000;
  if (turn && thinkStartedAt > 0 && turn.thinkSummary) {
    const thinkNow = thinkEndedAt > 0 ? thinkEndedAt : performance.now();
    const thinkSecs = Math.max(0, (thinkNow - thinkStartedAt) / 1000);
    const thinkText = turn.thinkSummary.querySelector(".think-text");
    if (thinkText) {
      const label = turn.thinkOpen && !thinkEndedAt ? "Thinking" : "Thought";
      thinkText.textContent = `${label} for ${thinkSecs.toFixed(2)} seconds`;
    }
  }
  if (turnTokens > 0) {
    // Decode phase: same format as the onToken handler so the
    // transition from prefill -> decode is visually seamless.
    const decodeEl = Math.max(0.001, (performance.now() - firstTokenAt) / 1000);
    const tps = turnTokens / decodeEl;
    meta.textContent = `${turnTokens} tokens · ${el.toFixed(1)}s · ${tps.toFixed(2)} tok/s`;
  } else {
    // Prefill phase (no tokens yet): just show elapsed time so the
    // clock visibly starts from the click. The phase indicator
    // (above this line in the chat) shows prefill progress.
    meta.textContent = `prefill · ${el.toFixed(1)}s`;
  }
  if (busy) _elapsedRaf = requestAnimationFrame(tickElapsed);
}

window.atf.onThink((open) => {
  if (!turn) return;
  turn.thinkOpen = open;
  if (open) {
    if (!thinkStartedAt) thinkStartedAt = performance.now();
    thinkEndedAt = 0;
    const thinkText = turn.thinkSummary.querySelector(".think-text");
    if (thinkText) thinkText.textContent = "Thinking for 0.00 seconds";
    turn.wrap.style.display = ""; turn.wrap.open = true;
    if (turn.phase) {
      turn.phase.classList.remove("prefill");
      turn.phase.classList.add("thinking");
      turn.phaseLabel.textContent = "Thinking…";
    }
  } else {
    if (thinkStartedAt && !thinkEndedAt) thinkEndedAt = performance.now();
    turn.answerStarted = true;
    if (turn.wrap) turn.wrap.open = false;       // collapse when answering starts
    if (turn.phase) {
      turn.phase.classList.remove("prefill", "thinking");
      turn.phaseLabel.textContent = "Answering…";
    }
  }
  scheduleRender();
});

window.atf.onTier((t) => {
  if (turn) turn.tier = t;
  setPhase(`${t.badge} · difficulty ${t.score}`);
  sbTier.textContent = `${t.badge} ${t.name || ""}`.trim();
  tierInfoEl.innerHTML = "";
  tierInfoEl.append(
    h("span", "tier-badge", `${t.badge} ${t.name || ""}`.trim()),
    h("div", "tier-detail", `difficulty ${typeof t.score === "number" ? t.score.toFixed(2) : t.score}`),
  );
});

// throttle tracker for prefill progress events (so we don't redraw more
// than ~30fps even if the bridge fires faster)
let _prefillLastRender = 0;
let _prefillRenderQueued = false;
let _prefillLastDone = 0;        // last `done` we rendered, for tok/s rate
let _prefillLastDoneAt = 0;      // perf.now() at the time of `_prefillLastDone`
let _prefillSmoothingFrom = 0;   // previous width we're animating from
let _prefillSmoothingTo = 0;     // target width
let _prefillCompleteFlashed = false;

function _renderPrefillBar(p, now) {
  if (!turn || !turn.phase) return;
  const done = Math.max(0, p.done || 0);
  const total = Math.max(0, p.total || 0);
  const hasTotal = total > 0;
  const pct = hasTotal ? Math.min(100, Math.max(0, (100 * done) / total)) : 0;
  const bar = turn.phaseBar;
  const fill = turn.phaseBarFill;

  // indeterminate state: engine emitted `done` but no `total` yet (very
  // first event on big prompts can arrive that way). Show a sweeping
  // gradient instead of a static 0% so the user sees motion immediately.
  if (!hasTotal) {
    if (!bar.classList.contains("indeterminate")) {
      bar.classList.add("indeterminate");
    }
  } else if (bar.classList.contains("indeterminate")) {
    bar.classList.remove("indeterminate");
  }

  // width: clamp to forward-only motion (progress events are monotonic,
  // but throttling can cause us to render the same `done` twice and a
  // rounding jump backwards reads as flicker). Use rAF-smoothed target.
  if (hasTotal) {
    _prefillSmoothingTo = pct;
    if (_prefillSmoothingFrom === 0) _prefillSmoothingFrom = pct;
    fill.style.width = `${pct}%`;
  }

  // text label
  const labelTxt = hasTotal
    ? `Thinking ${done.toLocaleString()}/${total.toLocaleString()}`
    : `Thinking ${done.toLocaleString()}…`;
  turn.phaseLabel.textContent = labelTxt;

  // tok/s + elapsed meta line. Rate is smoothed across the last two events
  // to avoid a single noisy chunk blowing up the number.
  const elapsedS = Math.max(0.001, (now - genStart) / 1000);
  let tokS = 0;
  if (done > _prefillLastDone && now > _prefillLastDoneAt) {
    const dTok = done - _prefillLastDone;
    const dT = (now - _prefillLastDoneAt) / 1000;
    if (dT > 0) tokS = dTok / dT;
  }
  _prefillLastDone = done;
  _prefillLastDoneAt = now;
  if (turn.phaseMeta) {
    const tokTxt = tokS > 0 ? `${tokS.toFixed(0)} tok/s` : "— tok/s";
    const etaTxt = hasTotal && tokS > 0
      ? ` · ETA ${((total - done) / tokS).toFixed(1)}s`
      : "";
    turn.phaseMeta.innerHTML =
      ` · <strong>${tokTxt}</strong> · ${elapsedS.toFixed(1)}s${etaTxt}`;
  }

  // brief "complete" pulse when we hit 100%
  if (hasTotal && done >= total && !_prefillCompleteFlashed) {
    _prefillCompleteFlashed = true;
    bar.classList.add("complete");
    setTimeout(() => bar.classList.remove("complete"), 700);
  }
}

window.atf.onProgress((p) => {
  if (!p || p.stage !== "prefill") return;
  const now = performance.now();
  // rAF-throttle: if a render is already queued this frame, just stash the
  // latest `p` so we render with the freshest data next frame.
  if (_prefillRenderQueued) return;
  if (now - _prefillLastRender < 33) {
    _prefillRenderQueued = true;
    requestAnimationFrame(() => {
      _prefillRenderQueued = false;
      _prefillLastRender = performance.now();
      _renderPrefillBar(p, _prefillLastRender);
    });
    return;
  }
  _prefillLastRender = now;
  _renderPrefillBar(p, now);
  // status-bar / overlay mirrors (kept exactly as before so we don't
  // regress anything outside the chat bubble)
  const txt = `Thinking ${p.done}/${p.total}`;
  if (busy) sbPhase.textContent = txt;
  const bar = $("bar-fill");
  if (bar) {
    const pct = Math.round((100 * (p.done || 0)) / Math.max(1, p.total || 1));
    bar.style.width = `${pct}%`;
    $("load-label").textContent = txt;
  }
});

window.atf.onUsage((u) => {
  // REAL prompt-token count from the tokenizer (not an estimate)
  sessionTokens = u.prompt_tokens || sessionTokens;
  sbPrefill.textContent = `${u.prompt_tokens} tok`;
  updateCtxUsage();
});

window.atf.onDone(async (s) => {
  if (!turn || !matchesCurrent(s)) return;
  const totalEl = (performance.now() - genStart) / 1000;
  const prefillEl = turnTokens > 0 ? Math.max(0, (firstTokenAt - genStart) / 1000) : totalEl;
  const decodeEl = Math.max(0.001, totalEl - prefillEl);
  const tps = turnTokens > 0 ? (turnTokens / decodeEl).toFixed(2) : "?";
  if (s.stopped) meta.textContent = "■ stopped";
  else meta.textContent =
    `✓ ${turnTokens} tokens · decode ${decodeEl.toFixed(2)}s (${tps} tok/s)` +
    (s.truncated ? " · TRUNCATED at budget" : "");
  sbPrefill.textContent = `${s.prompt_tokens ?? "?"} tok`;
  sbDecode.textContent = `${decodeEl.toFixed(2)}s`;
  sbTps.textContent = `${tps} tok/s`;
  sbTotal.textContent = `${totalEl.toFixed(1)}s`;

  // Update the "Thought for X.XX seconds" disclosure with the real elapsed time.
  // Primary signal: firstTokenAt - genStart (covers prefill + first thought
  // token, which is what "time spent thinking" means to the user). Fallback
  // to totalEl when no token ever arrived (e.g. stop was hit mid-think) so
  // the disclosure doesn't get stuck showing "Thought for 0.00 seconds".
  if (turn.thinkSummary) {
    const thinkSecs = thinkStartedAt > 0
      ? ((thinkEndedAt || performance.now()) - thinkStartedAt) / 1000
      : (firstTokenAt > 0 ? (firstTokenAt - genStart) / 1000 : totalEl);
    const thinkText = turn.thinkSummary.querySelector(".think-text");
    if (thinkText) thinkText.textContent = `Thought for ${thinkSecs.toFixed(2)} seconds`;
  }

  // Build a row of pill chips: tokens · tok/s · seconds · stop reason. Matches
  // the LM Studio layout and persists each metric with the saved message.
  if (turn.turnStats) {
    turn.turnStats.innerHTML = "";
    const makePill = (icon, text, cls) => {
      const p = h("span", `turn-pill${cls ? " " + cls : ""}`);
      if (icon) p.appendChild(h("span", "pill-icon", icon));
      p.appendChild(h("span", "pill-text", text));
      return p;
    };
    const stopReason = s.stopped ? "stopped" : (s.truncated ? "truncated" : "eos");
    // v17+: show the reasoning level the user asked for. Falls back to
    // settings.thinking (the last persisted default) when the live
    // turn.thinking wasn't captured — e.g. for messages generated by an
    // older build. Hidden when the level is "off" because then there
    // was no thinking to reason about.
    const thinkingLevel = (turn && turn.thinking) || (settings && settings.thinking) || "auto";
    const pills = [
      makePill("⚡", `${tps} tok/s`),
      makePill("▦", `${turnTokens} tokens`),
      makePill("⏱", `${totalEl.toFixed(2)}s`),
    ];
    if (thinkingLevel && thinkingLevel !== "off") {
      const label = `thinking: ${thinkingLevel}`;
      pills.push(makePill("🧠", label, "thinking-pill"));
    }
    pills.push(makePill(null, `Stop reason: ${stopReason}`, "stop-pill"));
    turn.turnStats.append(...pills);
    turn.turnStats.classList.remove("hidden");
  }

  // commit assistant message into the session
  const sess = activeSession();
  if (sess) {
    const content = turn.ansBuf.trim();
    if (content || turn.thinkBuf.trim()) {
      const stopReason = s.stopped ? "stopped" : (s.truncated ? "truncated" : "eos");
      const thinkSecs = thinkStartedAt > 0
        ? Number((((thinkEndedAt || performance.now()) - thinkStartedAt) / 1000).toFixed(2))
        : (firstTokenAt > 0 ? Number(((firstTokenAt - genStart) / 1000).toFixed(2)) : 0);
      sess.messages.push({
        role: "assistant",
        model: turn.modelId || currentModelId || undefined,
        content: content || "(no answer)",
        thinking: turn.thinkBuf.trim() || undefined,
        tier: turn.tier ? { badge: turn.tier.badge, name: turn.tier.name, score: turn.tier.score } : undefined,
        truncated: !!s.truncated,
        stats: {
          tokens: turnTokens,
          seconds: Number(totalEl.toFixed(2)),
          tps: Number(tps),
          thinkSeconds: thinkSecs,
          stopReason,
          // v17+: which reasoning level produced this turn. Null when
          // the user had thinking off. The renderer falls back to
          // settings.thinking for older messages that don't have it.
          thinking: (turn && turn.thinking) || null,
        },
      });
      sess.updated = Date.now();
      persistSessions();
    }
  }

  const truncatedNow = !!s.truncated && !s.stopped;
  finishMessage();

  if (truncatedNow) addContinueButton(sess);
});

function addContinueButton(sess) {
  const btn = h("button", "continue-btn", "⚠ Response hit the token budget — Continue ▸");
  btn.addEventListener("click", () => {
    btn.remove();
    submit("Continue exactly where you stopped.");
  });
  chat.appendChild(btn);
  scrollChat(true);
}

function finishMessage() {
  busy = false;
  if (_elapsedRaf) { cancelAnimationFrame(_elapsedRaf); _elapsedRaf = 0; }
  stopBtn.style.display = "none";
  renderTurn();                       // final full-markdown render
  if (turn) {
    if (turn.phase) turn.phase.classList.add("hidden");
    turn.actions.style.display = "flex";
    buildMessageActions(turn.actions);
    turn = null;
    curReqId = null;
  }
  if (currentStatus !== STATUS.ERROR) setStatus(STATUS.READY);
  // v17+: drop the "active" highlight from the right-pane thinking
  // level pill now that the generation has ended.
  if (paneThinkLevel) paneThinkLevel.classList.remove("active");
  send.disabled = !ready;
  sbConn.className = "sb-item on";
  input.focus();
}

window.atf.onMem((t) => {
  sbGpu.textContent = `gpu: ${t}`;
  logStatus(`gpu mem: ${t}`);
});

/* static (restored) message rendering) */
function rebuildChatFromSession() {
  chat.innerHTML = "";
  if (paneThink) paneThink.textContent = "(no active thinking)";
  tierInfoEl.textContent = "—";
  // The model-not-loaded welcome is appended first, then removed if the
  // active session already has saved messages. v19+ (SQLite) sessions
  // whose message history is still being lazy-loaded will briefly show
  // this welcome; openSession's .then() re-runs rebuildChatFromSession
  // once the messages arrive and removes it.
  const w = welcomeEl.cloneNode(true);
  w.style.display = "flex";
  chat.appendChild(w);
  const s = activeSession();
  if (s && s.messages.length) w.remove();
  if (s) {
    // Backfill: any assistant message saved before we started recording the
    // model id (or any future migration) gets the most-recent known model
    // from earlier in the same session. If none, fall back to the currently
    // loaded model, then to the last-used model from settings. This way
    // every assistant bubble shows the real model that produced it instead
    // of the static "ATF" placeholder.
    let backfilled = false;
    let lastKnownModel = null;
    for (const m of s.messages) {
      if (m.role === "assistant") {
        if (m.model) {
          lastKnownModel = m.model;
        } else {
          m.model = lastKnownModel
            || currentModelId
            || (settings && settings.lastModelId)
            || "ATF";
          backfilled = true;
        }
      }
    }
    for (const m of s.messages) {
      if (m.role === "user") appendUserBubble(m.content);
      else chat.appendChild(newTurnViewStatic(m));
    }
    const last = s.messages[s.messages.length - 1];
    if (last && last.role === "assistant" && last.truncated) addContinueButton(s);
    if (backfilled) persistSessions();
  }
  scrollChat(true);
}

function appendUserBubble(text) {
  const existingWelcome = chat.querySelector("#welcome");
  if (existingWelcome) existingWelcome.style.display = "none";
  const msgEl = h("div", "msg user");
  msgEl.appendChild(h("div", "who", "You"));
  const bubble = h("div", "bubble", text);
  const actions = h("div", "msg-actions user-actions");
  const copyBtn = h("button", "action-btn", "Copy ⧉");
  copyBtn.title = "Copy your message";
  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(text);
    copyBtn.textContent = "Copied ✓";
    setTimeout(() => (copyBtn.textContent = "Copy ⧉"), 1200);
  });
  const editBtn = h("button", "action-btn", "Edit ✎");
  editBtn.title = "Edit and resend";
  editBtn.addEventListener("click", () => editUserMessage(text));
  actions.append(copyBtn, editBtn);
  msgEl.append(bubble, actions);
  chat.appendChild(msgEl);
  scrollChat(true);
}

function newTurnViewStatic(m) {
  const msgEl = h("div", "msg assistant");
  // Restore the model that produced this reply; fall back to "ATF" for
  // legacy sessions saved before the model name was captured.
  msgEl.appendChild(h("div", "who", m.model || currentModelId || (settings && settings.lastModelId) || "ATF"));

  // "Thought for X.XX seconds" disclosure, only if there's thinking content.
  let wrap = null;
  if (m.thinking) {
    wrap = h("details", "think-wrap");
    const thinkSecs = m.stats?.thinkSeconds ?? 0;
    const summary = h("summary", "thinking-summary");
    summary.append(
      h("span", "think-chev", "›"),
      h("span", "think-text", `Thought for ${thinkSecs.toFixed(2)} seconds`)
    );
    wrap.appendChild(summary);
    wrap.appendChild(h("pre", "think-pre", m.thinking));
    msgEl.appendChild(wrap);
  }

  const answer = h("div", "bubble answer");
  answer.innerHTML = renderMarkdown(m.content);
  if (m.truncated) answer.appendChild(h("span", "trunc-note", " (truncated at token budget)"));

  // Restore the metric pill row if stats were saved with the message.
  const turnStats = h("div", "turn-stats");
  if (m.stats) {
    const makePill = (icon, text, cls) => {
      const p = h("span", `turn-pill${cls ? " " + cls : ""}`);
      if (icon) p.appendChild(h("span", "pill-icon", icon));
      p.appendChild(h("span", "pill-text", text));
      return p;
    };
    const stopReason = m.stats.stopReason || (m.truncated ? "truncated" : "eos");
    // v17+: reasoning level pill. Persisted per-message in v17+, fall
    // back to settings.thinking for older messages that don't have it.
    // Hidden when the level is "off" or unknown (no settings either).
    const thinkingLevel = m.stats.thinking
      || (settings && settings.thinking)
      || null;
    const pills = [
      makePill("⚡", `${m.stats.tps} tok/s`),
      makePill("▦", `${m.stats.tokens} tokens`),
      makePill("⏱", `${Number(m.stats.seconds).toFixed(2)}s`),
    ];
    if (thinkingLevel && thinkingLevel !== "off") {
      pills.push(makePill("🧠", `thinking: ${thinkingLevel}`, "thinking-pill"));
    }
    pills.push(makePill(null, `Stop reason: ${stopReason}`, "stop-pill"));
    turnStats.append(...pills);
  } else {
    turnStats.classList.add("hidden");
  }

  const actions = h("div", "msg-actions");
  buildMessageActions(actions);
  msgEl.append(answer, turnStats, actions);
  return msgEl;
}

function buildMessageActions(container) {
  container.innerHTML = "";
  const copyBtn = h("button", "action-btn", "Copy ⧉");
  copyBtn.title = "Copy answer";
  copyBtn.addEventListener("click", () => {
    const bubble = container.closest(".msg")?.querySelector(".bubble.answer");
    navigator.clipboard.writeText(bubble ? bubble.textContent : "");
    copyBtn.textContent = "Copied ✓";
    setTimeout(() => (copyBtn.textContent = "Copy ⧉"), 1200);
  });
  const regenBtn = h("button", "action-btn", "Regenerate ↻");
  regenBtn.title = "Regenerate this reply";
  regenBtn.addEventListener("click", regenerateLast);
  container.append(copyBtn, regenBtn);
}

function regenerateLast() {
  const s = activeSession();
  if (!s || busy) return;
  // drop trailing assistant message(s); resend from the last user turn
  while (s.messages.length && s.messages[s.messages.length - 1].role === "assistant") s.messages.pop();
  if (!s.messages.length || s.messages[s.messages.length - 1].role !== "user") {
    toast("Nothing to regenerate.", "warn"); return;
  }
  rebuildChatFromSession();
  persistSessions();
  startGeneration();
}

function editUserMessage(originalText) {
  const s = activeSession();
  if (busy) return;
  const idx = s.messages.findIndex((m) => m.role === "user" && m.content === originalText);
  if (idx >= 0) s.messages.splice(idx);      // drop it and everything after
  persistSessions();
  rebuildChatFromSession();
  input.value = originalText;
  input.focus();
  input.dispatchEvent(new Event("input"));
}

/* composer */
send.addEventListener("click", () => submit());
stopBtn.addEventListener("click", () => window.atf.stop());
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  else if (e.key === "Escape") window.atf.stop();
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

/* clear / new chat */
function clearConversation() {
  if (busy) { toast("Stop the current generation first.", "warn"); return; }
  const s = activeSession();
  if (!s) return;
  s.messages = [];
  s.title = "New chat";
  persistSessions();
  rebuildChatFromSession();
  renderSessionList();
  meta.textContent = "";
  sessionTokens = 0;
  updateCtxUsage();
}
$("new-chat").addEventListener("click", () => createSession());

/* tabs */
/* ── v17: Dashboard — mirrors live values from the status pane/API tab ── */
let _dashTick = 0;  // v19: used by dashSync to throttle SSD stats fetches

function dashSync() {
  const $id = (x) => document.getElementById(x);
  const txt = (x, fb) => ($id(x)?.textContent || fb || "").replace(/^(model|gpu):\s*/, "");
  const set = (x, v) => { const el = $id(x); if (el && el.textContent !== v) el.textContent = v; };

  const modelTxt = $id("sb-model")?.textContent || "";
  const loaded = /\((.+)\)\s*$/.exec(modelTxt);
  set("dash-model", loaded ? modelTxt.replace(/^model:\s*/, "") : "—");
  set("dash-model-sub", loaded ? "resident on GPU · unload via ⏏ in the title bar"
    : "not loaded — pick one above");
  set("dash-phase", $id("sb-phase")?.textContent || "idle");
  set("dash-conn", $id("sb-conn")?.textContent || "—");
  set("dash-gpu", txt("sb-gpu", "gpu: —"));
  set("dash-tps", $id("sb-tps")?.textContent?.trim() || "—");
  set("dash-prefill", $id("sb-prefill")?.textContent?.trim() || "no prompt yet");

  // context meter mirrors the session usage computed for sb-ctx
  const ctxEl = $id("ctx-used"), fill = $id("dash-ctx-fill");
  if (ctxEl) set("dash-ctx", ctxEl.textContent);
  if (fill) {
    const w = parseFloat($id("ctx-meter-fill")?.style.width) || 0;
    fill.style.width = `${w}%`;
  }

  // v19: Paged SSD KV cache footprint in the right pane. Throttled to
  // every second (dashSync runs at 500ms) so the main-process dir walk
  // does not become the bottleneck on a hot kvpages dir. Errors are
  // logged so a stuck "scanning…" never goes unnoticed.
  if (window.atf?.getKvSsdStats && ((_dashTick = (_dashTick + 1) % 2) === 0)) {
    window.atf.getKvSsdStats().then((s) => updateKvSsdPanel(s))
      .catch((e) => console.warn("[atf] getKvSsdStats failed:", e));
  }

  // API tile
  const apiState = $id("api-state")?.textContent || "stopped";
  set("dash-api", apiState.toLowerCase().includes("running") ? "running" : apiState);
  set("dash-api-url", $id("api-url")?.textContent || "http://localhost:8000/v1");

  // welcome subtitle follows app state
  const sub = $id("dash-sub");
  if (sub) {
    let s;
    if (typeof currentStatus !== "undefined" && currentStatus === STATUS.ERROR) s = "Engine error — see the log below.";
    else if (ready) s = `Ready. Chat in Inference, or serve it on ${$id("api-url")?.textContent || "the API page"}.`;
    else if (currentStatus === STATUS.LOADING) s = "Loading model…";
    else if (modelsLoaded) s = `Local models, private by design. Pick a model${settings.lastModelId ? ` (last used: ${settings.lastModelId})` : ""} to begin.`;
    else s = "Local models, private by design.";
    if (sub.textContent !== s) sub.textContent = s;
  }
}
setInterval(dashSync, 500);
// v19: kick the runtime-cache panel once at startup so the user never sees
// the "scanning…" placeholder. dashSync is throttled to 1s, so without
// this the first call may not run for up to a second after launch.
if (window.atf?.getKvSsdStats) {
  window.atf.getKvSsdStats().then((s) => updateKvSsdPanel(s))
    .catch((e) => console.warn("[atf] getKvSsdStats (boot) failed:", e));
}

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-page").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${name}`));
  // v18: the right pane belongs to the chat; CSS shows it only on inference
  document.body.classList.toggle("tab-inference", name === "inference");
}
// v19: refuse to navigate away from the Chat tab while a generation is
// running. Switching tabs hides the live streaming bubble (it lives
// inside #tab-inference), so the model would appear to "go silent" from
// the user's POV even though the engine is still running. They can stop
// the generation (Esc) or wait for it to finish and then navigate.
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    if (busy && target !== "inference") {
      toast("Generation in progress — stop it (Esc) or wait, then switch tabs.", "warn");
      return;
    }
    activateTab(target);
  });
});

/* context slider */
const CTX_MEM_GUARD = 24576;
function updateCtxLabel() {
  const v = parseInt($("ctx").value, 10);
  $("ctx-val").textContent = v + (v > CTX_MEM_GUARD ? " ⚠" : "");
  $("ctx").title = v > CTX_MEM_GUARD
    ? "Contexts above ~24k use several extra GB of KV cache — may pressure 16 GB RAM"
    : "";
}
$("ctx").addEventListener("input", updateCtxLabel);

function updateCtxUsage() {
  const cap = parseInt($("ctx").value, 10) || 65536;
  const used = Math.min(sessionTokens, cap);
  ctxUsedEl.textContent = `${used} / ${cap} tok`;
  ctxMeter.style.width = `${Math.min(100, (used / cap) * 100)}%`;
  sbCtx.textContent = `${used} / ${cap} tok`;
}

// v19: Render the paged SSD KV stats into the right pane. Hides the
// whole row when there is no on-disk footprint, so a user who has not
// run a long context sees a clean panel. Pulses the dot when the
// footprint changed in the last tick (i.e. an active generation is
// spilling pages right now).
let _lastSsdBytes = -1;
// v19: Runtime-cache panel — "Cache files / Total size / Location",
// modeled on the oMLX Status page layout. The whole block gets a
// pulsing accent dot while the on-disk footprint is changing (i.e. an
// active generation is spilling fresh pages) and dims to text-dim when
// the footprint is stable. Zero-byte stats render the same rows with
// em-dashes so the user always sees the path even before the first
// spill — fixing the previous "scanning…" indefinite placeholder.
function updateKvSsdPanel(stats) {
  // v19: use the module-scope $ helper (document.getElementById). The
  // earlier $id is only a local inside dashSync; this function lives
  // at module scope and must use the global helper.
  const block = $("ctx-ssd-block");
  const filesEl = $("ctx-ssd-files");
  const usedEl = $("ctx-ssd-used");
  // The Location row was removed in v19; pathEl is optional and only
  // written to when present. clearBtn may also be absent if the panel
  // was rendered without the action button.
  const pathEl = $("ctx-ssd-path");
  const clearBtn = $("ctx-ssd-clear");
  if (!block || !filesEl || !usedEl) return;

  const bytes = (stats && Number.isFinite(stats.totalBytes)) ? stats.totalBytes : 0;
  const files = (stats && Number.isFinite(stats.fileCount)) ? stats.fileCount : 0;
  if (pathEl) {
    const path = (stats && typeof stats.path === "string" && stats.path.length)
      ? stats.path : "~/.cache/atf/kvpages";
    pathEl.textContent = path;
    pathEl.title = path;
  }

  filesEl.textContent = files > 0 ? String(files) : "—";
  usedEl.textContent = bytes > 0 ? formatGiB(bytes) : "—";

  if (clearBtn) clearBtn.disabled = files === 0;

  if (bytes !== _lastSsdBytes && bytes > 0) {
    block.classList.add("active");
    block.classList.remove("idle");
  } else {
    block.classList.add("idle");
    block.classList.remove("active");
  }
  _lastSsdBytes = bytes;
}

// v19: wire the Clear-cache button. We only fire on user click, never
// in a tick — the IPC is cheap (one readdir + a few unlinks) but it
// should still feel deliberate. After clearing we re-paint the panel
// from the snapshot returned by main.
$("ctx-ssd-clear")?.addEventListener("click", async () => {
  if (!window.atf?.clearKvSsdCache) return;
  const btn = $("ctx-ssd-clear");
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "Clearing…";
  try {
    const fresh = await window.atf.clearKvSsdCache();
    updateKvSsdPanel(fresh);
    const n = fresh?.removed ?? 0;
    toast(n > 0
      ? `Cleared ${n} cached page${n === 1 ? "" : "s"} (${formatGiB(fresh.bytes || 0)}).`
      : "Cache was already empty.", "ok", 2400);
  } catch (e) {
    console.warn("[atf] clearKvSsdCache failed:", e);
    toast("Could not clear cache — see dev console.", "warn");
  } finally {
    btn.textContent = original;
    // re-enable based on the now-zero footprint
    const files = $("ctx-ssd-files")?.textContent;
    btn.disabled = files === "—" || files === "0";
  }
});

function formatGiB(n) {
  // Always show in GiB per the chat-tab right-pane spec. Two-decimal
  // precision keeps a 640 MiB kvmm file reading as "0.62 GiB" rather
  // than rounding away the data. The sub-line carries the byte-exact
  // file count for users who want to see what's there.
  if (!Number.isFinite(n) || n <= 0) return "0.00 GiB";
  return `${(n / (1 << 30)).toFixed(2)} GiB`;
}

/* ── §8 settings modal ─────────────────────────────────────────── */
function activateSettingsSection(name) {
  document.querySelectorAll(".settings-nav-item").forEach((n) =>
    n.classList.toggle("active", n.dataset.section === name));
  document.querySelectorAll(".settings-section").forEach((s) =>
    s.classList.toggle("active", s.dataset.section === name));
}
document.querySelectorAll(".settings-nav-item").forEach((btn) => {
  btn.addEventListener("click", () => activateSettingsSection(btn.dataset.section));
});

function setRadio(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((el) => {
    el.checked = el.value === value;
  });
}
function getRadio(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}
function bindRadio(name, onChange) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((el) => {
    el.addEventListener("change", () => { if (el.checked) onChange(el.value); });
  });
}

function openSettings() {
  // ── Chat & generation
  $("set-system").value = settings.system || "";
  $("set-max-tokens").value = settings.maxTokens;
  $("set-top-p").value = settings.topP;
  $("set-repeat").value = settings.repeatPenalty;
  $("set-temp").value = settings.temperature;
  $("set-temp-val").textContent = Number(settings.temperature).toFixed(2);
  $("inference-temp").value = settings.temperature;
  $("inference-temp-val").textContent = Number(settings.temperature).toFixed(2);
  $("set-mirror-think").checked = !!settings.mirrorThink;
  $("set-show-tokens").checked = !!settings.showTokens;

  // ── Model & context
  $("set-confirm-unload").checked = !!settings.confirmUnload;
  $("set-auto-resume").checked = !!settings.autoResume;
  // v19: Paged SSD KV cache
  $("set-kvssd-enabled").checked = settings.kvSsdEnabled !== false;
  $("set-kvssd-hot-pages").value = settings.kvSsdHotPages || 256;
  $("set-kvssd-hot-pages-val").textContent = String(settings.kvSsdHotPages || 256);
  // The path field is async-resolved from main (it may differ from the
  // raw stored value due to "~/" expansion). Show the resolved value.
  if (window.atf?.getKvSsdPath) {
    window.atf.getKvSsdPath().then((p) => {
      $("set-kvssd-path").value = p || "";
      $("set-kvssd-path").placeholder = "~/.cache/atf/kvpages";
    });
  }

  // ── Appearance (live update; no Save click needed)
  setRadio("theme-mode", settings.theme);
  setRadio("accent", settings.accent);
  setRadio("fontSize", settings.fontSize);
  setRadio("density", settings.density);
  $("set-animations").checked = settings.animations !== false;

  // ── Advanced
  $("set-telemetry").checked = !!settings.telemetry;
  $("set-clear-on-exit").checked = !!settings.clearOnExit;

  // Populate the models-dir field with the live value from main.
  if (window.atf?.getModelsDir) {
    window.atf.getModelsDir().then((d) => {
      $("set-models-dir").value = d || "";
      $("set-models-dir").placeholder = d ? "" : "(not yet resolved)";
    });
  }
  activateSettingsSection("appearance");
  $("settings-modal").classList.remove("hidden");
}

function closeSettings(saveIt) {
  if (saveIt) {
    settings.system = $("set-system").value.trim();
    settings.maxTokens = clampInt($("set-max-tokens").value, 64, 65536, 65536);
    settings.topP = clampFloat($("set-top-p").value, 0.05, 1, 0.9);
    settings.repeatPenalty = clampFloat($("set-repeat").value, 1, 1.5, 1.15);
    settings.temperature = parseFloat($("set-temp").value);
    settings.mirrorThink = $("set-mirror-think").checked;
    settings.showTokens = $("set-show-tokens").checked;
    settings.confirmUnload = $("set-confirm-unload").checked;
    settings.autoResume = $("set-auto-resume").checked;
    settings.telemetry = $("set-telemetry").checked;
    settings.clearOnExit = $("set-clear-on-exit").checked;
    settings.animations = $("set-animations").checked;
    // v19: Paged SSD KV cache settings. Persist the path through main
    // so the "~" expansion happens in one place; the renderer just
    // stashes the user's raw text and trusts main to canonicalize.
    settings.kvSsdEnabled = $("set-kvssd-enabled").checked;
    settings.kvSsdHotPages = clampInt($("set-kvssd-hot-pages").value, 32, 2048, 256);
    const _kvssdPathText = $("set-kvssd-path").value.trim();
    if (window.atf?.setKvSsdPath) {
      window.atf.setKvSsdPath(_kvssdPathText || null).then((resolved) => {
        if (resolved) $("set-kvssd-path").value = resolved;
      });
    }
    // The raw user-typed path lands in settings so the next openSettings()
    // call has something to show; main also persists the canonical one.
    settings.kvSsdPath = _kvssdPathText || null;
    // radio groups already saved on change, but read final value for safety
    const themeMode = getRadio("theme-mode");
    if (themeMode) settings.theme = themeMode;
    const accent = getRadio("accent");
    if (accent) settings.accent = accent;
    const fs = getRadio("fontSize");
    if (fs) settings.fontSize = fs;
    const dn = getRadio("density");
    if (dn) settings.density = dn;
    applyAppearance();
    saveSettingsSoon();
    toast("Settings saved.", "ok", 1800);
  }
  $("settings-modal").classList.add("hidden");
  input.focus();
}
function clampInt(v, lo, hi, dflt) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
}
function clampFloat(v, lo, hi, dflt) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
}
$("settings-btn").addEventListener("click", openSettings);

// Theme toggle button in the titlebar: cycle light → dark → auto → light
$("theme-btn").addEventListener("click", () => {
  const order = ["light", "dark", "auto"];
  const cur = order.indexOf(settings.theme);
  settings.theme = order[(cur + 1) % order.length];
  applyAppearance();
  saveSettingsSoon();
  toast(
    "Theme: " + (settings.theme === "auto"
      ? "follows system (" + resolveThemeMode("auto") + ")"
      : settings.theme),
    "ok", 1400
  );
});

// Live-update radios so the UI reflects changes before the user clicks Done.
bindRadio("theme-mode", (v) => { settings.theme = v; applyAppearance(); saveSettingsSoon(); });
bindRadio("accent", (v) => { settings.accent = v; applyAppearance(); saveSettingsSoon(); });
bindRadio("fontSize", (v) => { settings.fontSize = v; applyAppearance(); saveSettingsSoon(); });
bindRadio("density", (v) => { settings.density = v; applyAppearance(); saveSettingsSoon(); });

// Reset to defaults button
$("set-reset")?.addEventListener("click", async () => {
  if (!confirm("Reset all settings to defaults? Saved sessions and models are not affected.")) return;
  settings = { ...DEFAULTS };
  applyAppearance();
  saveSettingsSoon();
  openSettings();   // re-populate the form
  toast("Settings reset to defaults.", "ok", 1800);
});
$("settings-close").addEventListener("click", () => closeSettings(false));
$("settings-cancel").addEventListener("click", () => closeSettings(false));
$("settings-save").addEventListener("click", () => closeSettings(true));

// Models-dir controls. Save fires immediately (doesn't wait for the Save
// button) because the change is unrelated to sampling/system-prompt and
// should take effect at once — including re-scanning the registry.
async function changeModelsDir(dir) {
  if (!dir || !dir.trim()) {
    toast("Enter a folder path first.", "warn", 2500);
    return;
  }
  try {
    const abs = await window.atf.setModelsDir(dir.trim());
    $("set-models-dir").value = abs;
    toast(`Models folder set to ${abs}`, "ok", 2500);
  } catch (e) {
    toast(`Could not set models folder: ${e.message || e}`, "err", 4000);
  }
}
$("set-models-dir-pick").addEventListener("click", async () => {
  const picked = await window.atf.pickModelsDir();
  if (picked) await changeModelsDir(picked);
});
$("set-models-dir").addEventListener("keydown", async (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    await changeModelsDir($("set-models-dir").value);
  }
});
$("set-models-dir-reveal").addEventListener("click", () => {
  const v = $("set-models-dir").value;
  if (v) toast(`Path: ${v}`, "ok", 4000);
});

// v19: Paged SSD KV cache live bindings. The slider only updates the
// label in real time; the actual value is read on Save (closeSettings).
$("set-kvssd-hot-pages")?.addEventListener("input", () => {
  $("set-kvssd-hot-pages-val").textContent = String(parseInt($("set-kvssd-hot-pages").value, 10) || 256);
});
$("set-kvssd-path-pick")?.addEventListener("click", async () => {
  if (!window.atf?.pickKvSsdPath) return;
  const picked = await window.atf.pickKvSsdPath();
  if (picked) {
    $("set-kvssd-path").value = picked;
    toast("Cache folder picked — click Save to apply.", "ok", 2200);
  }
});
$("set-kvssd-path-reveal")?.addEventListener("click", () => {
  const v = $("set-kvssd-path").value;
  if (v) toast(`Path: ${v}`, "ok", 4000);
});
if (window.atf?.onModelsDirChanged) {
  window.atf.onModelsDirChanged((p) => {
    $("set-models-dir").value = p?.dir || "";
    if (p?.dir) toast(`Models folder updated: ${p.dir}`, "ok", 2500);
  });
}
$("set-temp").addEventListener("input", () => {
  const value = parseFloat($("set-temp").value);
  $("set-temp-val").textContent = value.toFixed(2);
  $("inference-temp").value = value;
  $("inference-temp-val").textContent = value.toFixed(2);
});
$("inference-temp").addEventListener("input", () => {
  const value = parseFloat($("inference-temp").value);
  settings.temperature = value;
  $("inference-temp-val").textContent = value.toFixed(2);
  $("set-temp").value = value;
  $("set-temp-val").textContent = value.toFixed(2);
  saveSettingsSoon();
});
$("settings-modal").addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeSettings(false);
});
$("settings-modal").addEventListener("mousedown", (e) => {
  if (e.target === $("settings-modal")) closeSettings(false);
});

/* ── §9 API server tab ─────────────────────────────────────────── */
let apiRunning = false;
let apiAdoptedExternal = false;

function setApiUI(running, state) {
  apiRunning = running;
  apiToggle.textContent = running ? "Stop server" : "Start server";
  apiToggle.classList.toggle("danger", running);
  apiDot.className = `api-dot ${running ? "on" : ""}`;
  if (!running && (state === "starting…" || state === "stopping…")) {
    apiDot.classList.add("starting");
  }
  apiState.textContent = state || (running ? "running" : "stopped");
  apiPortInput.disabled = running && !apiAdoptedExternal;
  // v10: Start button only enabled when a model is loaded (or already
  // running so the user can stop it). Without a model the server has
  // nothing to serve.
  apiToggle.disabled = !running && !ready;
  const port = parseInt(apiPortInput.value, 10) || 8000;
  apiUrl.textContent = `http://localhost:${port}/v1`;
}

apiToggle.addEventListener("click", () => {
  apiAdoptedExternal = false;
  if (apiRunning) {
    window.atf.apiServerStop();
    apiState.textContent = "stopping…";
  } else {
    // v10: guard against race where model was ejected between clicks
    if (!ready || !currentModelId) {
      apiState.textContent = "load a model first";
      return;
    }
    window.atf.apiServerStart({ port: parseInt(apiPortInput.value, 10) || 8000 });
    apiState.textContent = "starting…";
  }
});

apiPortInput.addEventListener("change", () => {
  const port = parseInt(apiPortInput.value, 10) || 8000;
  apiUrl.textContent = `http://localhost:${port}/v1`;
  if (apiRunning) apiState.textContent = "restart required after port change";
});

window.atf.onApiServerStatus((s) => {
  apiAdoptedExternal = !!(s.running && !s.owned);
  setApiUI(s.running, s.state);
});

window.atf.onApiLogAppend((chunk) => {
  if (apiLogEl.textContent.startsWith("(")) apiLogEl.textContent = "";
  const stick = apiLogEl.scrollTop + apiLogEl.clientHeight >= apiLogEl.scrollHeight - 32;
  apiLogEl.textContent += chunk;
  if (stick) apiLogEl.scrollTop = apiLogEl.scrollHeight;
});

/* ── §9b Convert tab (GGUF → ATF) ─────────────────────────────── */
const cvGgufBtn = $("cv-gguf-btn"), cvGgufPath = $("cv-gguf-path"),
  cvOutBtn = $("cv-out-btn"), cvOutPath = $("cv-out-path"),
  cvStart = $("cv-start"), cvCancel = $("cv-cancel"),
  cvState = $("cv-state"), cvDot = $("cv-dot"), cvLogEl = $("cv-log");
let cvGguf = null, cvOutput = null, cvRunning = false, cvSource = "gguf";
const cvSourceSel = $("cv-source"), cvMlxOpts = $("cv-mlx-opts"),
  cvGgufOpts = $("cv-gguf-opts");

cvSourceSel.addEventListener("change", () => {
  cvSource = cvSourceSel.value;
  const mlx = cvSource === "mlx";
  cvGgufBtn.textContent = mlx ? "Choose MLX folder…" : "Choose Model…";
  cvGgufPath.textContent = "nothing selected";
  cvGguf = null;
  cvOutput = null;
  cvOutPath.textContent = "—";
  cvStart.disabled = true;
  cvMlxOpts.classList.toggle("hidden", !mlx);
  cvGgufOpts.classList.toggle("hidden", mlx);
});

function cvSetUI(running, stateText) {
  cvRunning = running;
  cvStart.disabled = running || !cvGguf;
  cvCancel.classList.toggle("hidden", !running);
  cvState.textContent = stateText;
  cvDot.classList.toggle("on", running);
}

function cvSuggestOutput(ggufPath) {
  // default: <models dir>/v3/<name>.atf next to the repo's other models,
  // falling back to the source folder if we cannot know the models dir.
  const base = ggufPath.split("/").filter(Boolean).pop() || "model";
  const name = base.replace(/\.gguf$/i, "").replace(/-\d+bit$/i, "") + ".atf";
  const root = ggufPath.includes("/atf/") ? ggufPath.split("/atf/")[0] : null;
  return root ? `${root}/models/v3/${name}` : name;
}

cvGgufBtn.addEventListener("click", async () => {
  const picked = cvSource === "mlx"
    ? await window.atf.convertPickMlx()
    : await window.atf.convertPickGguf();
  if (!picked) return;
  cvGguf = picked;
  cvGgufPath.textContent = picked;
  cvGgufPath.title = picked;
  if (!cvOutput) {
    cvOutput = cvSuggestOutput(picked);
    cvOutPath.textContent = cvOutput;
    cvOutPath.title = cvOutput;
  }
  if (!cvRunning) cvSetUI(false, "idle");
});

cvOutBtn.addEventListener("click", async () => {
  const picked = await window.atf.convertPickOutput(cvOutput || cvSuggestOutput(cvGguf || ""));
  if (!picked) return;
  cvOutput = picked.endsWith(".atf") ? picked : picked + ".atf";
  cvOutPath.textContent = cvOutput;
  cvOutPath.title = cvOutput;
});

cvStart.addEventListener("click", async () => {
  if (cvRunning || !cvGguf) return;
  cvLogEl.textContent = "";
  cvSetUI(true, "starting…");
  try {
    const res = await window.atf.convertStart({
      source: cvSource,
      targetBits: $("cv-bits") ? parseInt($("cv-bits").value, 10) : 4,
      gguf: cvGguf,
      output: cvOutput,
      experts: clampInt($("cv-experts").value, 1, 512, 8),
      topK: clampInt($("cv-topk").value, 1, 512, 4),
      expertStorage: $("cv-storage").value,
      denseFp4: $("cv-densefp4").checked,
      raw: $("cv-raw").checked,
    });
    if (!res?.ok) {
      toast(res?.error || "Could not start the conversion.", "error", 4000);
      cvSetUI(false, "error");
    }
  } catch (err) {
    toast(`Convert failed to start: ${err}`, "error", 4000);
    cvSetUI(false, "error");
  }
});

window.atf.onConvertStarted(() => cvSetUI(true, "converting…"));

window.atf.onConvertLog((line) => {
  if (!cvLogEl) return;
  cvLogEl.textContent += line + "\n";
  cvLogEl.scrollTop = cvLogEl.scrollHeight;
});

window.atf.onConvertDone(async ({ code, cancelled } = {}) => {
  if (cancelled) {
    cvSetUI(false, "cancelled");
    cvLogEl.textContent += "\n[cancelled]\n";
    return;
  }
  if (code === 0) {
    cvSetUI(false, "done ✓");
    cvDot.classList.add("on");
    toast("Conversion finished — refreshing model list.", "ok", 3000);
    try { await window.atf.listModels(); } catch { /* registry refresh best-effort */ }
  } else {
    cvSetUI(false, `failed (exit ${code})`);
  }
});

cvCancel.addEventListener("click", () => {
  if (!cvRunning) return;
  cvSetUI(true, "cancelling…");
  window.atf.convertCancel();
});

/* ── §10 keyboard, menu, boot ──────────────────────────────────── */
document.addEventListener("keydown", (e) => {
  // Cmd+1..5 → switch tabs
  if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey &&
    ["1", "2", "3", "4", "5"].includes(e.key)) {
    const map = { "1": "dashboard", "2": "inference", "3": "models", "4": "api", "5": "convert" };
    const target = map[e.key];
    if (target) {
      e.preventDefault();
      // v19: same busy-guard as the click handler — the chat tab is the
      // only safe destination while a generation is streaming.
      if (busy && target !== "inference") {
        toast("Generation in progress — stop it (Esc) or wait, then switch tabs.", "warn");
        return;
      }
      activateTab(target);
      return;
    }
  }
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "l") {
    e.preventDefault();
    const order = ["light", "dark", "auto"];
    const cur = order.indexOf(settings.theme);
    settings.theme = order[(cur + 1) % order.length];
    applyAppearance();
    saveSettingsSoon();
    toast("Theme: " + (settings.theme === "auto"
      ? "follows system (" + resolveThemeMode("auto") + ")"
      : settings.theme), "ok", 1200);
    return;
  }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault(); clearConversation();
  }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
    e.preventDefault(); createSession();
  }
});

window.atf.onMenuAction((action) => {
  switch (action) {
    case "new-chat": createSession(); break;
    case "clear-chat": clearConversation(); break;
    case "settings": openSettings(); break;
    case "toggle-theme": $("theme-btn")?.click(); break;
  }
});

/* restore UI state after reload without faking a loading screen */
(async function boot() {
  await loadSettings();
  applyAppearance();
  thinkingSel.value = settings.thinking || "auto";
  thinkingSel.addEventListener("change", () => {
    settings.thinking = thinkingSel.value;
    saveSettingsSoon();
    updateThinkingLevelPill();
  });
  // v17+: keep the right-pane "Model thinking" level pill in sync with
  // the dropdown. Called on boot, on every change, and whenever a
  // generation starts or ends so the pill reflects the level that was
  // actually used for the in-flight (or most-recently-completed) turn.
  function updateThinkingLevelPill(level) {
    if (!paneThinkLevel) return;
    const v = level || (thinkingSel && thinkingSel.value) || (settings && settings.thinking) || "auto";
    paneThinkLevel.textContent = v === "auto" ? "Adaptive" : v.charAt(0).toUpperCase() + v.slice(1);
    paneThinkLevel.classList.remove("active", "off");
    if (v === "off") paneThinkLevel.classList.add("off");
    paneThinkLevel.title = v === "auto"
      ? "Adaptive — reasoning effort chosen by the model"
      : `Reasoning effort: ${v}`;
  }
  updateThinkingLevelPill();
  $("ctx").value = settings.contextTokens || 65536;
  updateCtxLabel();
  updateCtxUsage();
  $("ctx").addEventListener("change", () => {
    settings.contextTokens = parseInt($("ctx").value, 10) || 65536;
    saveSettingsSoon();
    updateCtxUsage();
  });

  try {
    const st = await window.atf.getState();
    if (st.apiServer?.running) {
      if (st.apiServer.port) apiPortInput.value = st.apiServer.port;
      apiAdoptedExternal = st.apiServer.running && !st.apiServer.owned;
      setApiUI(true, "running");
      if (st.apiLog) apiLogEl.textContent = st.apiLog;
    } else if (st.apiLog) {
      setApiUI(false, "stopped");
      apiLogEl.textContent = st.apiLog;
    }
    if (st.engineReady && st.currentModelId) {
      ready = true;
      currentModelId = st.currentModelId;
      selectedModelId = currentModelId;
      setStatus(STATUS.READY);
      send.disabled = false;
      sbModel.textContent = `model: ${st.currentModelId}`;
    } else {
      // engine connected but no model loaded -> idle, not loading
      setStatus(st.engineReady ? STATUS.IDLE : STATUS.LOADING);
    }
  } catch {
    setStatus(STATUS.LOADING);
  }

  await loadSessions();
})();
