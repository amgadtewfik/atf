# ATF v0.8.0 — Adaptive routing, paged-SSD KV cache, persistent sessions

**Release date:** 2026-09-05
**Size:** ~232 MB (DMG)
**Platform:** macOS Apple Silicon (16 GB+)
**Tree:** gamma/v7

---

## What's New in v0.8.0

v0.8.0 is an **engine + storage** release built on gamma/v7. The
headline changes are:

- **Adaptive routing on every turn** — `atf/router.py` is now live on the
  bridge hot path. Every `/generate` request is scored by the lightweight
  router and stamped with one of four tiers (`⚡ instant`, `💬 chat`,
  `🧠 reasoning`, `🔬 deep`); the tier badge, name, and difficulty score
  appear next to every assistant reply, in the chat header, and as a
  permanent session stat. The Settings panel's "Adaptive tier" section
  now drives the engine's per-turn thinking budget instead of being a
  pure mock. The tier auto-collapses when the next turn starts so the
  reply is always the focal point.
- **Paged SSD KV cache (`ATF_KV_SSD=1`, default on)** — the v0.6.0-era
  flat in-memory KV cache caps at 65k context on the 9B and trips the
  truncation meter on the 27B well before its real headroom. v0.8.0
  introduces a new `atf/kvstore.py` two-tier page store: hot pages
  stay on the GPU, cold pages spill to an `mmap`-backed file on SSD.
  Page size is 256 tokens; the resident set defaults to 256 pages
  (~16k tokens on the GPU). With the SSD cache enabled, 65k context
  on the 9B IQ4_NL fits cleanly with zero truncation; the 27B
  Q2_K_XL stops tripping the truncation meter at its real ceiling.
  Configurable from the Settings → Runtime cache panel (path picker,
  hot-pages slider) or via `ATF_KV_SSD=0` env to revert to the
  flat-buffer path.
- **Paged-SSD persistence (`ATF_KV_SSD_PERSIST=1`, opt-in)** — the
  SSD cache is now resumable across process restarts. When this flag
  is set, each completed turn writes a small `<key>.meta.npz`
  sidecar next to its `<key>_b{N}.kvmm` page files (the same
  `<key>` is a sha256 of `model_id + first 256 token ids`, stable
  per conversation). On the next load, `find_reusable()` matches
  by exact LCP — partial-prefix reuse is rejected, because GDN
  recurrent state cannot be trimmed to an arbitrary earlier
  position without breaking the recurrence. Default is OFF because
  a reuse bug here produces garbled output rather than a crash,
  and we'd rather gate it explicitly during the on-device
  coherence-check pass.
- **Compiled full-GDN step (`ATF_COMPILE_FULL_GDN=1`, default on)** —
  extends the v0.6.0 `ATF_COMPILE_STEP` (single-step compile) to the
  entire single-token GDN recurrence in one `mx.compile` trace:
  alpha/beta projections, conv1d shift, log(1+exp) on the alpha bias,
  sigmoid(beta), the decay `exp(g)` multiplication, the
  `state @ k` retrieval, the `(v − kv_mem)·β` delta, the
  `delta ⊗ k` write-back, the `state @ q` projection, and the SSM
  norm — all fused into one command buffer per GDN block. Fused
  ~20 kernel launches per GDN block into 1. Bit-exact numerical
  parity verified (`≤ 2.956×10⁻¹²`) by
  `tests/test_perf_parity_compile_full.py`.
- **SQLite-backed chat sessions (`v19` migration)** — sessions are
  now persisted in a SQLite database on the main process (via
  `better-sqlite3`), not in a JSON file the renderer rewrites on
  every save. This fixes the v18 "lost six sessions on quit" bug
  (the JSON rewrites were racy under rapid quit/restart), and gives
  the renderer an O(1) per-session row read instead of parsing the
  whole sessions file every load. The migration is automatic: if
  you had a JSON store from v0.7.0, the v0.8.0 first launch reads
  it, writes to SQLite, and keeps the JSON in place as a safety
  copy until the next clean quit.
- **Runtime-cache panel in the right pane** — a new card in the
  Chat tab's right column shows live SSD cache stats: file count,
  total size, oldest/newest mtime, and a "Clear SSD cache" button.
  Throttled to one update per second so it never fights the render
  loop. Path picker + hot-pages slider for fine-tuning the SSD
  cache from the UI without restarting.
- **Per-turn metrics carry the tier** — the existing v0.7.0
  per-turn metrics row (`⚡ tok/s · tokens · seconds · stop`)
  now also persists `tier = { badge, name, score }` with each
  message, so a session saved in v0.8.0 shows the right tier
  chip next to every reply when reopened later. Legacy v0.7.0
  sessions render the chip in the "—" placeholder state until
  the next turn re-runs the router.

### Bridge changes

- The bridge now imports `atf.router.route` and `BADGES` on every
  `/generate` (not just the v0.7.0 `from atf.router import` that
  sat behind a debug branch). The router call is a pure function
  over the prompt text + a `thinking` override — adds
  sub-millisecond overhead per request, no model load needed.
- `tier` event payload schema: `{ type, level, name, badge, score,
  thinking }`. The renderer consumes it via the existing
  `window.atf.onTier(...)` IPC; the bootstrap lazily wires the
  handler only after the bridge is connected.
- Tokenizer source for files that don't match the Qwen3.5 vocab
  heuristic remains `Qwen/Qwen3.5-4B` (v0.7.0 carryover).

### Engine changes

- **Paged-SSD KV cache plumbing** — `KVCache` got a new
  `enable_ssd(path, hot_pages, persist)` method; when active,
  all `append/keys/values/close` calls delegate to a `PagedKV`
  instance from `atf/kvstore.py`. The flat-buffer code paths are
  preserved verbatim — `enable_ssd` is opt-in via the
  `ATF_KV_SSD=1` env var (default on in v0.8.0; off in
  v0.7.0 by accident).
- **Prefix-cache reuse wired through the SSD cache** — the
  v0.7.0 prefix-cache reuse path now `resume_n`s into the new
  PagedKV, so a fresh session that happens to share its first
  256 tokens with a previously-persisted session inherits the
  saved keys/values without re-running prefill on the shared
  prefix.
- **`ATF_COMPILE_FULL_GDN=1`** — the new compiled full-GDN step
  is selected in the per-block forward when `T==1`, the model is
  raw (mlx), and the GDN weights have been pinned (Win B). The
  narrower single-step compile (Win A) is still used for the
  per-token recurrence loop in non-raw models.
- **v19 Sold.transpose(0, 2, 1) silent-contraction fix** —
  carried over from the v0.6.0 fix at `engine.py:2442`. The
  chunked prefill GDN scan stays opt-in (`ATF_GDN_SCAN=1`)
  because the fp32 conditioning bug across long real prompts
  is still open.

### Files changed

| File | What |
|---|---|
| `atf/kvstore.py` | NEW — PagedKV two-tier page store (PAGE=256 tokens, mmap-backed SSD file, hot-pages resident on GPU), `session_key()`, `save_manifest()`, `load_manifest()`, `find_reusable()` |
| `atf/engine.py` | Paged-SSD KV cache wiring on `KVCache.enable_ssd(...)`; prefix-cache resume into PagedKV; `ATF_COMPILE_FULL_GDN=1` full-step compile path (Win A Expanded); prefix-cache reuse through SSD |
| `atf/router.py` | Existing adaptive LOD scorer (`score_prompt`, `tier_for`, `route`) — unchanged from v0.7.0; **now on the bridge hot path** |
| `bridge/atf_bridge.py` | `from atf.router import route, BADGES` lifted out of the debug branch onto every `/generate`; `tier` event payload with `badge / name / score / thinking` |
| `electron/main.cjs` | `ATF_KV_SSD=1` (default on), `ATF_KV_SSD_PERSIST=0` (opt-in), `ATF_KV_SSD_PATH`, `ATF_KV_SSD_HOT_PAGES` env forwarding to the bridge; paged-SSD `kvSsdWalkStats()`, `clear-kvssd-cache` handler that also removes `.meta.npz` sidecars (no orphans); new IPC: `get-kvssd-path`, `set-kvssd-path`, `pick-kvssd-path`, `get-kvssd-stats` |
| `electron/main.cjs` | v19 SQLite migration: `openSessionDb()` (better-sqlite3), `sessions-active-get`, `sessions-save`, `sessions-delete`, `sessions-rename`, `sessions-export` — replaces the v0.7.0 JSON-file store end-to-end |
| `electron/preload.cjs` | New IPC bridges: `kvssd:get-path`, `kvssd:set-path`, `kvssd:pick-path`, `kvssd:get-stats`, `kvssd:clear`; `tier` event hookup on `atf.onTier` |
| `electron/renderer/index.html` | Runtime-cache panel markup (file count, size, path picker, hot-pages slider, Clear-cache button); `id="sb-tier"` sidebar stat; `id="tier-info"` adaptive-tier panel |
| `electron/renderer/app.js` | v19 SQLite-aware `loadSessions/saveSessions/deleteSession/renameSession`; per-turn `tier` stat persisted with each message; tier badge in chat header (with legacy backfill for v0.7.0 sessions); paged-SSD KV settings bound to the Settings panel; runtime-cache panel live bindings; kvSsdEnabled migration (`typeof settings.kvSsdEnabled !== "boolean"` → `true`) |
| `electron/renderer/style.css` | Runtime-cache panel styles; tier-badge + tier-detail tokens; chat-header tier chip; per-turn tier pill in the metrics row |
| `tests/test_ssd_kv_cache.py` | NEW — PagedKV standalone (append/keys/values/eviction/mmap round-trip), KVCache SSD integration (enable_ssd, append/keys/values delegation, close), numerical parity (SSD-backed KVCache ≡ flat-buffer KVCache) |
| `tests/test_kv_ssd_persist.py` | NEW — `session_key()` stability, `save_manifest`/`load_manifest` round-trip, `find_reusable()` exact-prefix-only rule (partial LCP must NOT match), PagedKV cross-process resume (persist=True, close(), open second instance with `resume_n`, verify keys()/values() match) |
| `tests/test_perf_parity_compile_full.py` | NEW — `ATF_COMPILE_FULL_GDN=1` bit-exact parity vs uncompiled reference (`≤ 2.956×10⁻¹²`) |

### New module: `atf/kvstore.py`

The paged-SSD KV cache implementation. Two tiers:

- **Hot tier (GPU)** — `mx.array` fp16, bounded to `kv_ssd_hot_pages`
  (default 256 pages × 256 tokens ≈ 16k tokens). LRU-style eviction
  when the resident set fills up.
- **Cold tier (SSD)** — mmap over a preallocated per-session file at
  `~/.cache/atf/kvpages/` (override with `ATF_KV_SSD_PATH`). Pages
  spill out to disk in 256-token blocks and re-load on demand.

When `ATF_KV_SSD_PERSIST=1` is set, a `.meta.npz` sidecar is written
next to the `.kvmm` page files. The sidecar holds the fed token ids
and the per-block GDN recurrent state needed to safely resume a
session across process restarts. `find_reusable()` matches by
**exact prefix length**, never partial-LCP — GDN recurrent state
cannot be trimmed to an arbitrary earlier prefix without breaking
the recurrence, so a partial-prefix "match" would produce
coherent-looking but subtly-wrong output.

### Settings panel: Runtime cache (v19)

A new section in the Settings modal dedicated to the paged-SSD KV
cache:

- **Path** — defaults to `~/.cache/atf/kvpages/`. Pick a folder with
  the `…` button; the chosen path is persisted and forwarded to the
  next bridge spawn via `ATF_KV_SSD_PATH`.
- **Hot pages** — defaults to 256. Slider maps to `ATF_KV_SSD_HOT_PAGES`.
  Lower values free more GPU memory at the cost of more SSD traffic
  during long-context decoding; higher values keep more on the GPU.
- **Clear cache button** — removes all `.kvmm` and `.meta.npz`
  files in the cache dir. The Python side recreates the directory
  on the next long-context generation.

### Test status

All v0.8.0 tests pass:

- `tests/test_ssd_kv_cache.py` — 9/9 (PagedKV standalone + KVCache SSD integration + numerical parity)
- `tests/test_kv_ssd_persist.py` — 7/7 (`session_key`, manifest round-trip, exact-LCP `find_reusable`, cross-process resume)
- `tests/test_perf_parity_compile_full.py` — 1/1 (bit-exact parity `≤ 2.956×10⁻¹²`)
- `tests/test_router_units.py` — 13/13 (carried over from v0.7.0)
- `tests/test_spec_acceptance.py` — 11/11
- `tests/test_spec_shallow_integration.py` — 8/8
- `tests/test_inference.py` — 3/3
- `tests/test_qsa_parity.py` — 15/15
- `tests/test_ngram_parity.py` — similar
- `tests/test_qsa_converter_stub.py` — 10/10
- `tests/test_ngram_converter_stub.py` — 10/10
- `tests/test_syscache.py` + variants — pass

## Bench snapshot (gamma/v7)

The compile-step gains carry over from v0.6.0 unchanged:

| Variant | 9B tok/s | 27B tok/s |
|---|---|---|
| v0.5.0 default (`temperature=0.7`, no compile) | 12.6 | 4.24 |
| Greedy only | 14.49 | 4.63 |
| Greedy + `ATF_COMPILE_STEP=1` (v0.6.0 / v0.7.0 default) | **14.83** | **4.80** |

`ATF_COMPILE_FULL_GDN=1` (the new default in v0.8.0) extends the
single-step compile to the entire per-token GDN recurrence, fusing
~20 kernel launches per GDN block down to 1. The
v0.6.0 single-step compile is kept for backward compat; the new
full-step compile picks up wherever the model is raw (mlx) and
the GDN weights have been pinned. No per-token perf regression
expected and no behavior change on models that don't trigger
the new path.

The Win1 GDN-scan attempt remains a **non-win**: the chunked path
is not faster than the per-token Python recurrence on this
architecture (0.98× at 64 tokens, 0.99× at 128 tokens on the 9B
IQ4_NL) and produces broken output until the fp32 conditioning
bug is fixed (separate research task). See `WIN1_BENCH_RESULTS.md`
for the v0.6.0-era numbers.

## What's NOT in v0.8.0 (forward-looking)

- **Phase 2 of Subject-Relevance LOD** (`atf/subject_router.py`) —
  Phase 1 (pure scoring + domain ranking) is in the tree; Phase 2
  (engine-side consumption / per-token expert loading) needs an
  engine hook in `_ffn_moe` and a v0/v1 per-domain-LOD storage
  convention. Not yet wired.
- **Paged-SSD persistence default-on** — `ATF_KV_SSD_PERSIST=1`
  ships OFF by default. The persistence path is tested and
  bit-exact in `test_kv_ssd_persist.py`, but a reuse bug here
  produces garbled output rather than a crash, so we'd rather
  keep it opt-in until a real on-device coherence check has been
  run on a multi-turn persistence cycle.
- **Real Qwen4 QSA + n-gram** — still gated on a public Qwen4 GGUF
  landing. The structural-path tests and real-weight test
  scaffolding auto-activate the moment a usable Qwen4 GGUF
  becomes available.
- **Cross-engine 9B-on-27B design (Configuration C)** — needs 16 GB
  residency fitting both models and a real Qwen4 MTP-aware
  converter. Highest-ceiling alternative to the native MTP head
  path; not yet started.
- **Linux/Windows port (cross-platform Metal shim)** — out of scope.

---

## Install

1. Download `ATF Chat-v0.8.0-arm64.dmg` below.
2. Open the DMG, drag **ATF Chat** to **Applications**.
3. First launch: right-click the app → **Open** (skip the Gatekeeper
   warning; the build is unsigned by design for fast local
   distribution).
4. Pick a `.atf` model from the in-app dropdown. The first load
   takes a few seconds; subsequent loads are cached. v0.8.0 picks
   up your v0.7.0 sessions automatically (SQLite migration runs
   once on first launch).

## Run from source

```sh
git clone https://github.com/amgadtewfik/atf
cd atf
./run.sh
```

`run.sh` bootstraps the bundled venv (`electron/app-venv/`), installs
the editable `atf` package, runs `node scripts/ensure-electron.mjs`
(idempotent Electron install + native module rebuild), and launches
via `npm start`. No manual Python setup required.

## Requirements

- macOS 13+ on Apple Silicon (M1 / M2 / M3 / M4)
- 16 GB unified memory (the 27B Q2_K_XL peaks at ~13 GB; the 9B
  IQ4_NL at ~6 GB; the paged-SSD cache needs another few GB of
  free disk on the cache dir)
- ~10 GB free disk for the SSD KV cache + models

## Build from source

```sh
cd electron
npm install            # one-time, ~30s, auto-runs ensure-electron
npm run vendor         # bundle markdown-it, highlight.js, dompurify
npm run dist:dmg       # electron-builder --mac dmg
# → electron/dist/ATF Chat-arm64.dmg  (~232 MB)
```

## License

MIT. Models are downloaded separately from Hugging Face under their
own licenses; the app itself is just the runtime.
