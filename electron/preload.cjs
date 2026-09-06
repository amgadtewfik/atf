const { contextBridge, ipcRenderer } = require("electron");

// Single, narrow surface between the sandboxed renderer and the main
// process. Every engine event carries the originating request id so the
// UI can correlate streams with the message that produced them.
contextBridge.exposeInMainWorld("atf", {
  // requests (return the assigned request id where applicable)
  generate: (req) => ipcRenderer.invoke("generate", req),
  stop: () => ipcRenderer.invoke("stop"),
  reloadEngine: () => ipcRenderer.invoke("reload-engine"),
  unloadModel: () => ipcRenderer.invoke("unload-model"),
  listModels: () => ipcRenderer.invoke("list-models"),
  loadModel: (id) => ipcRenderer.invoke("load-model", id),
  deleteModel: (id) => ipcRenderer.invoke("delete-model", id),
  getModels: () => ipcRenderer.invoke("list-models"),

  // persistent store (sessions / settings / window state)
  storeGet: (key) => ipcRenderer.invoke("store-get", key),
  storeSet: (key, value) => ipcRenderer.invoke("store-set", key, value),
  storeDelete: (key) => ipcRenderer.invoke("store-delete", key),

  // v19: chat sessions live in a SQLite DB on the main process. The
  // renderer never touches the file system directly. list() returns
  // the metadata for the sidebar; get(id) returns the full message
  // history; save(sess) upserts a session and its messages; delete
  // and rename are obvious. Persisting settings and the kvssd cache
  // still goes through storeGet / storeSet (JSON under the same dir).
  sessions: {
    list: () => ipcRenderer.invoke("sessions-list"),
    get: (id) => ipcRenderer.invoke("sessions-active-get", id),
    save: (sess) => ipcRenderer.invoke("sessions-save", sess),
    delete: (id) => ipcRenderer.invoke("sessions-delete", id),
    rename: (id, title) => ipcRenderer.invoke("sessions-rename", id, title),
  },

  // models directory (user-configurable from the Settings panel)
  getModelsDir: () => ipcRenderer.invoke("get-models-dir"),
  setModelsDir: (dir) => ipcRenderer.invoke("set-models-dir", dir),
  pickModelsDir: () => ipcRenderer.invoke("pick-models-dir"),
  onModelsDirChanged: (cb) => ipcRenderer.on("models-dir-changed", (_e, p) => cb(p)),

  // v19: paged SSD KV cache — settings + live stats for the right pane.
  // The renderer asks main for the current on-disk footprint of the
  // kvpages directory every dashSync tick; main walks the dir, sums
  // file sizes, and reports the active path. No Python involvement.
  getKvSsdPath: () => ipcRenderer.invoke("get-kvssd-path"),
  setKvSsdPath: (dir) => ipcRenderer.invoke("set-kvssd-path", dir),
  pickKvSsdPath: () => ipcRenderer.invoke("pick-kvssd-path"),
  getKvSsdStats: () => ipcRenderer.invoke("get-kvssd-stats"),
  clearKvSsdCache: () => ipcRenderer.invoke("clear-kvssd-cache"),

  // OpenAI-compatible server control
  getState: () => ipcRenderer.invoke("get-state"),
  apiServerStart: (opts) => ipcRenderer.invoke("api-server-start", opts),
  apiServerStop: () => ipcRenderer.invoke("api-server-stop"),

  // events
  onModels: (cb) => ipcRenderer.on("bridge-models", (_e, items) => cb(items)),
  onModelLoaded: (cb) => ipcRenderer.on("model-loaded", (_e, m) => cb(m)),
  onModelUnloaded: (cb) => ipcRenderer.on("model-unloaded", cb),
  onReloading: (cb) => ipcRenderer.on("reloading", cb),
  onReady: (cb) => ipcRenderer.on("bridge-ready", cb),
  onCrashed: (cb) => ipcRenderer.on("bridge-crashed", (_e, info) => cb(info)),
  onStatus: (cb) => ipcRenderer.on("status", (_e, t) => cb(t)),
  onProgress: (cb) => ipcRenderer.on("progress", (_e, p) => cb(p)),
  onToken: (cb) => ipcRenderer.on("token", (_e, t) => cb(t)),
  onTier: (cb) => ipcRenderer.on("tier", (_e, t) => cb(t)),
  onThink: (cb) => ipcRenderer.on("think", (_e, o) => cb(o)),
  onUsage: (cb) => ipcRenderer.on("usage", (_e, u) => cb(u)),
  onDone: (cb) => ipcRenderer.on("done", (_e, s) => cb(s)),
  onError: (cb) => ipcRenderer.on("error", (_e, m) => cb(m)),
  onMem: (cb) => ipcRenderer.on("mem", (_e, t) => cb(t)),
  onMenuAction: (cb) => ipcRenderer.on("menu-action", (_e, a) => cb(a)),

  // API server log is delivered incrementally
  onApiServerStatus: (cb) => ipcRenderer.on("api-server-status", (_e, s) => cb(s)),
  onApiLogAppend: (cb) => ipcRenderer.on("api-log-append", (_e, chunk) => cb(chunk)),

  // GGUF -> ATF conversion (Convert tab)
  convertPickGguf: () => ipcRenderer.invoke("convert-pick-gguf"),
  convertPickMlx: () => ipcRenderer.invoke("convert-pick-mlx"),
  convertPickOutput: (suggested) => ipcRenderer.invoke("convert-pick-output", suggested),
  convertStart: (req) => ipcRenderer.invoke("convert-start", req),
  convertCancel: () => ipcRenderer.invoke("convert-cancel"),
  onConvertStarted: (cb) => ipcRenderer.on("convert-started", (_e, r) => cb(r)),
  onConvertLog: (cb) => ipcRenderer.on("convert-log", (_e, line) => cb(line)),
  onConvertDone: (cb) => ipcRenderer.on("convert-done", (_e, r) => cb(r)),

  // Hugging Face model search/download (Models tab)
  hfSearchModels: (opts) => ipcRenderer.invoke("hf-search-models", opts),
  hfListFiles: (repoId) => ipcRenderer.invoke("hf-list-files", repoId),
  hfDownloadFiles: (req) => ipcRenderer.invoke("hf-download-files", req),
  hfCancelDownload: (downloadId) => ipcRenderer.invoke("hf-cancel-download", downloadId),
  onHfDownloadLog: (cb) => ipcRenderer.on("hf-download-log", (_e, line) => cb(line)),
  onHfDownloadProgress: (cb) => ipcRenderer.on("hf-download-progress", (_e, p) => cb(p)),
  onHfDownloadDone: (cb) => ipcRenderer.on("hf-download-done", (_e, r) => cb(r)),
});
