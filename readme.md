# ATF v0.7.0 — Per-turn metrics, multi-file HF repo, smoother prefill progress

**Release date:** 2026-09-04
**Size:** ~226 MB (DMG)
**Platform:** macOS Apple Silicon (16 GB+)

---

## What's New in v0.7.0

v0.7.0 is a **chat-observability + HF-repo polish release** built on gamma/v6.
The headline changes are:

- **Per-turn metrics row under every assistant reply** — small pill chips
  show `tok/s`, total tokens, wall-clock seconds, and the stop reason
  (`stop`, `length`, `cancel`). Hidden while streaming, populated when
  the turn finishes, and persisted with the message so reopening a
  session shows the same numbers next to every answer.
- **Active model name in the chat header** — instead of the static `ATF`
  label, each turn shows the model that produced it
  (`Qwen3.5-9B-IQ4_NL.atf` etc.). Backfills the most recent known model
  on legacy sessions saved before v0.7.0.
- **"Thought for X.XX seconds" disclosure with real elapsed time** —
  the thinking-summary line now shows prefill-plus-first-thought-token
  wall time, not the placeholder `0.00`. Thinking auto-collapses when
  answering starts so the reply is immediately readable.
- **Copy button on user turns** — every user message has a `Copy ⧉`
  action in its row; the assistant copy button has been there since v17.
- **HF Models tab is now a single multi-file repo** — the Models tab
  now reads from `huggingface.co/amgadtewfik/atf/tree/main/models`
  and shows **one card per .atf file** (instead of one card per repo).
  Per-file cards carry the file's `path`, the repo's shared
  downloads/likes/lastModified, and a download button that grabs
  exactly the named file. The old per-repo cards (still supported
  for legacy items) are keyed by `repoId::path` to avoid collisions.
- **Smoother prefill progress in the status bar** — the engine now
  emits per-block progress on every change with a 60 Hz soft cap
  (was a 0.25 s throttle that sometimes collapsed a 60 ms multi-block
  prefill into a single 0 % → 100 % jump). Configurable via
  `ATF_PREFILL_HEARTBEAT=0` to disable, or
  `ATF_PREFILL_HEARTBEAT_BLOCKS=N` to force every-Nth-block cadence.
- **Qwen3.5-4B tokenizer as the unified fallback** — the bridge,
  CLI, and `bench_win1.py` now default to `Qwen/Qwen3.5-4B`
  (unified ~250k BBPE tokenizer that works for Qwen3.5/3.6/3.8) when
  the model path doesn't say `3.5` explicitly. The previous default
  `Qwen/Qwen3-8B` only fits the older Qwen3.0 generation.

### Engine changes

- **Prefill heartbeat** — block-cadence (default ≈ 8 evenly-spaced
  updates, regardless of `num_blocks`) with a 10 s wall-clock backstop
  and a final-block guarantee. Previously the wall-clock-only cadence
  meant a 64-block model where the entire prefill finished in
  well under 10 s only emitted the last block.
- **Per-block progress forwarded to the bridge** — the engine's
  per-block heartbeat now also calls `progress_cb(...)` with
  `(s + (b+1)/num_blocks * len(piece), total)` so the renderer sees
  a smooth 0 → 100 % bar during single-chunk prefills (the chunk-level
  callback at the bottom of the function only fires once for prompts
  under 1024 tokens, which is most of them).
- **MoE shared-expert reuse-mode fallback** —
  `atf/convert_moe.py` writes a "reuse" scaffold where the routed
  experts are untrained slices of the original dense FFN. For files
  without real `ffn_gate_shexp.weight` records, the engine now
  computes the shared-expert term directly from the block's own
  dense FFN tensors (the same `qmm()` / `merge_ffn()` path the plain
  dense forward pass already uses) instead of returning an
  untrained, narrow, CPU-dequantized synthetic slice. Real trained
  MoE files (`m.has(...) True`) are unaffected — they fall through
  to the router/routed-expert path exactly as before.
- **`convert-moe` CLI** — new `atf convert-moe` command exposing the
  v2 recipes (`--shexp-alias` / `--shexp-slice` — default is sliced,
  bf16, faster at inference), the active-expert count, the top-k,
  the router dtype (`f32` for A3B compat, `f16` to save bytes), and
  the shexp gate bias (constant 10.0 → sigmoid ≈ 1.0).

### Bridge changes

- **Tokenizer source** — `Qwen3-8B` → `Qwen3.5-4B` for files that
  don't match the Qwen3.5 vocab/path heuristic.
- **Prefill progress** — emit on every meaningful change (was a
  0.25 s throttle that sometimes coalesced an entire prefill into
  one event). 60 Hz soft cap protects the socket from pathological
  back-to-back events; in-flight prefill only.

### Models-tab changes (HF multi-file repo)

- The HF repo moved from `amgadtewfik` (one repo per model) to
  `amgadtewfik/atf` (one repo, many .atf files under `models/`).
- The Models tab fetches the `tree/main/models` listing once,
  groups by file, and shows one card per file. The repo metadata
  (likes, downloads, lastModified, tags) is shared across all cards.
- Downloads use the file path directly against
  `huggingface.co/amgadtewfik/atf/resolve/main/<path>`.
- Legacy items (one repo per file) still work: the renderer's HF card
  key is `<repoId>::<path>` and the IPC handlers fall back to the
  old `/api/models?author=...` URL when given an author that isn't
  a `repo/subpath`.

### Files changed

| File | What |
|---|---|
| `atf/engine.py` | Prefill heartbeat rewritten (block-cadence + per-block progress_cb call);<br>MoE shared-expert reuse-mode fallback for files without real `ffn_gate_shexp` records |
| `atf/cli.py` | New `convert-moe` command exposing v2 recipes;<br>`Qwen3-8B` → `Qwen3.5-4B` as default tokenizer |
| `atf/convert_moe.py` | V2 rewrite: aliasing-shared (`--shexp-alias`) and sliced-shared (`--shexp-slice`, default) recipes;<br>`MoeConfig` dataclass, BF16 shexp packing, A2B naming |
| `atf/convert_ngram.py` | NEW — Qwen3.8-Flash-Next ngram-lookup-table converter (fast MoE-via-ngram alternative) |
| `atf/router.py` | Existing adaptive LOD scorer (`score_prompt`, `tier_for`, `route`) — ready to wire, not yet on the bridge hot path |
| `atf/subject_router.py` | NEW — Subject-Relevance LOD (SR-LOD) Phase 1: pure scoring + domain ranking, not wired to the engine |
| `bridge/atf_bridge.py` | Tokenizer src `Qwen3-8B` → `Qwen3.5-4B`;<br>prefill progress: 60 Hz soft cap, no event coalescing |
| `electron/main.cjs` | HF IPC: single repo `amgadtewfik/atf` with `tree/main/models`;<br>per-file cards (`id = repoId::path`) |
| `electron/renderer/app.js` | HF per-file card rendering (repoId + path, `cardKey` registry);<br>active model name in chat header (with legacy backfill);<br>per-turn metrics row (`tok/s · tokens · seconds · stop`);<br>"Thought for X.XX seconds" disclosure with real elapsed time;<br>thinking auto-collapses when answering starts;<br>Copy button on user turns |
| `electron/renderer/index.html` | Composer placeholder `Message ATF…` → `Send a message to the model…`;<br>HF author label updated to `huggingface.co/amgadtewfik/atf` |
| `electron/renderer/style.css` | `.who` chip redesign (no icon, model-name colored by sender);<br>per-turn pill chips (`.turn-pill`, `.stop-pill`);<br>think-chev rotating disclosure;<br>input padding + line-height tweak for the multi-line composer |
| `tests/bench_win1.py` | Tokenizer fallback `Qwen3-8B` → `Qwen3.5-4B` (and primary in `DEFAULT_TOKENIZERS`) |
| `tests/test_router_units.py` | NEW — unit tests for `atf/router.py` (pure scoring, tier mapping, overrides, degenerate-output detection) |

### New tool: `convert-moe` (v2 recipes)

```sh
# Default: sliced-shared (BF16), 64 experts, top-5 — A2B recipe
atf convert-moe models/Qwen3.5-9B-IQ4_NL.atf

# Legacy alias-shared recipe (slower at inference but full FFN as shexp)
atf convert-moe models/Qwen3.5-9B-IQ4_NL.atf --shexp-alias

# Tune the active-expert count and top-k
atf convert-moe models/Qwen3.5-9B-IQ4_NL.atf --num-experts 32 --top-k 4
```

Output file is named `<stem>_A2B.atf` next to the source and adds
~6–300 MB of scaffold (BF16 sliced shexp + router) on top of the
already-quantized dense base. The dense FFN is reused as-is for the
routed experts via the existing column-wise slicing path, so no
per-expert records are stored.

### New infrastructure: `convert_ngram.py`

The ngram converter (`atf convert-ngram`, sibling to `convert-moe`)
implements the **Qwen3.8-Flash-Next lookup-table** approach: an
ngram table indexed by recent token-pairs / token-triples is a
strictly faster form of "knowledge scaling" than MoE — one
`mx.take` + one tiny `[ngram_dim, hidden]` matmul per token, no
per-expert dispatch, LRU-paged hot rows. Recipe N is fast but
untrained (the table is small Gaussian init); continued
pre-training is a separate work item.

## Bench results

No engine perf regression in v0.7.0. The v0.6.0 defaults remain
the fastest path: greedy + `ATF_COMPILE_STEP=1` (now on by default)
yields **14.83 tok/s on the 9B IQ4_NL** and **4.80 tok/s on the 27B
Q2_K_XL** on the gamma/v4/5/6 dev line (Apple M4, 16 GB unified,
greedy, temperature 0, seed 42). The new prefill-heartbeat
cadence adds zero per-token cost; it only fires the engine's existing
heartbeat on more boundaries.

The Win1 GDN-scan attempt remains a **non-win**: the chunked path
is not faster than the per-token Python recurrence on this
architecture (0.98× at 64 tokens, 0.99× at 128 tokens on the 9B
IQ4_NL) and produces broken output until the fp32 conditioning
bug called out at `engine.py:711` is fixed (separate research
task). See `WIN1_BENCH_RESULTS.md`.

---

## Install

1. Download `ATF Chat-v0.7.0-arm64.dmg` below.
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
# → electron/dist/ATF Chat-arm64.dmg  (~226 MB)
```

## License

MIT. Models are downloaded separately from Hugging Face under their
own licenses; the app itself is just the runtime.
