# ATF Chat v0.17.0 — Release Notes

**Release:** [v0.17.0 on GitHub](https://github.com/amgadtewfik/atf/releases/tag/v0.17.0)

This release change the name of the package to ATF.Chat-arm64.dmg to make the update works and fixes the silent-mode 47-second GDN decode stall and introduces the exception-handling coding practice.

## 🚀 Key Highlights

### 1. Fix Silent-Mode GDN Per-Timestep Stall (~43s)
- **Root cause**: `_gdn_step_compiled()` (the per-token GDN recurrence compiled via `@mx.compile`) had a warm-up that could fail silently when `ATF_VERBOSE == 0`. The broad `except` swallowed errors, `warmed` stayed 0, and the full Metal JIT-trace cost (~43s) was deferred to the first real decode token after the 3.73s TTFT.
- **Fix** (`atf/engine.py`): warm-up exceptions now always surface via `console.print` (`[red]`), `_vlog(0, ...)`, and `results/stall_debug_20260918.log` (full traceback) regardless of verbosity. Silent mode now completes warm-up correctly.

### 2. Coding Practice — Never Swallow Exceptions
- Added to `AGENTS.md`: **Exception Handling** — never swallow exceptions silently; always raise and print with full traceback so stalls and failures are visible regardless of `ATF_VERBOSE` mode.

### 3. Infrastructure & Versioning
- `pyproject.toml`: `0.21.1`, `electron/package.json`: `0.16.0`.
- `docs/STATUS.md`: changelog updated; `results/fix_20260918_verbosestall.txt` added.

## 🛠 Technical Changes
- **Version**: `v0.16.0` (Electron) / `0.21.1` (Python package).
- **Fix**: `atf/engine.py` warm-up block (line ~1377) — exception no longer hidden when `_VERBOSE == 0`.
- **Practice**: `AGENTS.md` — Exception Handling guideline added.

---
*For detailed status and historical changes, please refer to `docs/STATUS.md`.*
