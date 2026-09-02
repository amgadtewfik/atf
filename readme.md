# ATF v0.6.0 — Greedy default, faster compile-step, full-context responses

**Release date:** 2026-09-02
**Size:** ~236 MB (DMG)
**Platform:** macOS Apple Silicon (16 GB+)

---

## What's New in v0.6.0

v0.6.0 is a **usability and performance release** built on gamma/v4.
The headline changes are:

- **Greedy decoding by default** — every generation now runs at
  `temperature=0.0` (deterministic, fastest, most reproducible). The
  previous default of 0.7 is still available per-request.
- **`ATF_COMPILE_STEP=1` shipped as the default** — the per-token GDN
  recurrence is now `mx.compile`-traced into a single Metal command
  buffer, measured at **+4% on the 27B Q2_K_XL** (4.63 → 4.80 tok/s
  on the "hi" prompt) at byte-identical output.
- **No artificial token cap** — `max_tokens` defaults to the full
  65 536-token context budget, with no separate cap. The bridge
  automatically clamps to `max_context − actual_prompt_length − 16`
  so the engine never sees a sum that overflows the context window.
- **Elapsed-time clock in the chat** — the per-turn timer now starts
  the instant the user clicks Submit / Enter / Regenerate, ticking
  every animation frame. Previously it only updated on the first
  progress event (during prefill) and on the first token (during
  decode), so the user saw a frozen "0.0s" until something arrived.
- **Chat session switching restored** — the previous build's
  generation-aware session guard was reverted after it incorrectly
  blocked switches between non-generating sessions. The generation
  itself is never interrupted by a session click; only the
  *display* of the in-flight turn is rebuilt (the stream continues in
  the background and the live tokens are dropped until the user
  switches back).

### Bug fixes

- **250-id LM-head padding guard** — the `.atf` files report
  `vocab_size=248320` but the Qwen tokenizer only knows 24 8070 entries.
  The 250-id gap is standard Qwen3 LM-head padding (zero-initialized
  for hardware alignment) and `tok.decode([248319])` panics in
  gigatoken's Rust backend. A new `clamp_padding` helper in
  `atf/spec.py` masks those rows to `-inf` before sampling so the
  drafter and verifier can never pick a panic-causing id.
- **Spec-decoding correctness fix** — the closed-fence SVG preview
  path was broken because `DOMPurify.sanitize` was stripping the
  `data-svg-preview` attribute (only the streaming path was
  allowlisting it). Both paths now permit the attribute.
- **Electron `setWindowOpenHandler` regression** — the default
  handler was denying `blob:` URLs with `action: "deny"`, silently
  breaking the SVG preview. The handler now opens a new
  `BrowserWindow` for `blob:` URLs and falls back to
  `shell.openExternal` for `http(s):`.

### Files changed

| File | What |
|---|---|
| `atf/engine.py` | GenConfig defaults flipped (temperature 0.7→0.0);<br>`ATF_COMPILE_STEP` default opt-out instead of opt-in;<br>spec decode + EMA-tracked `spec_min_acceptance` auto-disable |
| `atf/spec.py` | `clamp_padding` helper + 4 unit tests (padding guard) |
| `bridge/atf_bridge.py` | `max_tokens` default raised 512→65536;<br>clamped to `ctx − prompt_tokens − 16` before GenConfig;<br>request-error reporting (busy flag, no silent drops) |
| `atf/server_openai.py` | `max_tokens` cap 8192→65536 to match bridge |
| `electron/renderer/app.js` | SVG preview button (▶) next to copy (⎘) in code-pane header;<br>elapsed-time ticker via `requestAnimationFrame`;<br>session switching back to unconditional |
| `electron/renderer/style.css` | Bigger copy icon (18px→22px);<br>blue play icon (`#3b82f6`);<br>`.code-actions` group so copy + play cluster together |
| `electron/renderer/index.html` | Temperature slider default 0.7→0;<br>max-tokens input default 2048→65536 |
| `electron/main.cjs` | `setWindowOpenHandler` now opens a new `BrowserWindow`<br>for `blob:` URLs (SVG preview) |
| `docs/spec_vocab_check.md`,<br>`docs/spec_vocab_check_findings.md`,<br>`docs/phase0_summary.md`,<br>`docs/gamma_v3_progress.md` | Phase 0 docs for spec decoding, plus the<br>2026-09-02 Phase 3 bench result update |
| `docs/mtp_prior_art/` | Cloned `quivent/qwen-mtp-research` for<br>acceptance-rate / draft-cost ratio reference |

### Bench results (gamma/v4, 2026-09-02)

| Variant | 9B tok/s | 9B speedup | 27B tok/s | 27B speedup |
|---|---|---|---|---|
| Baseline (default, v0.5.0 behavior) | 12.6 | 1.00× | 4.24 | 1.00× |
| Greedy default only | 14.49 | 1.15× | 4.63 | 1.09× |
| Greedy + `ATF_COMPILE_STEP=1` | 14.83 | 1.18× | 4.80 | **1.13×** |
| Shallow self-spec K=2 (opt-in) | 9.04 | 0.72× | 2.46 | 0.58× |
| Shallow self-spec K=4 (opt-in) | n/m | n/m | 2.11 | 0.50× |

Shallow self-spec is **off by default** (`spec_draft=None`); the
v0.5.0-recommended headline of 12-15 tok/s on the 27B required a
different drafter design (cross-engine 9B-on-27B or native MTP head)
that's blocked on the 16 GB residency and a real Qwen4 GGUF
respectively. The measured 1.13× at the default settings is the
honest shipped win.

---

## Install

1. Download `ATF Chat-v0.6.0-arm64.dmg` below.
2. Open the DMG, drag **ATF Chat** to **Applications**.
3. First launch: right-click the app → **Open** (skip the Gatekeeper
   warning; the build is unsigned by design for fast local
   distribution).
4. Pick a `.atf` model from the in-app dropdown. The first load
   takes a few seconds; subsequent loads are cached.

## Run from source

```sh
git clone https://github.com/amgadtewfik/atf
cd atf
./run.sh
```

`run.sh` bootstraps the bundled venv (`electron/app-venv/`), installs
the editable `atf` package, and launches via `npm start`. No
network, no manual Python setup.

## Requirements

- macOS 13+ on Apple Silicon (M1 / M2 / M3 / M4)
- 16 GB unified memory (the 27B Q2_K_XL peaks at ~13 GB; the 9B
  IQ4_NL at ~6 GB)
- ~2 GB free disk per model on first load

## Build from source

```sh
cd electron
npm install            # one-time, ~30s
npm run vendor         # bundle markdown-it, highlight.js, dompurify
npm run dist:dmg       # electron-builder --mac dmg
# → electron/dist/ATF Chat-arm64.dmg  (~236 MB)
```

## License

MIT. Models are downloaded separately from Hugging Face under their
own licenses; the app itself is just the runtime.
