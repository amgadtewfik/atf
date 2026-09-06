"""Model registry: discover .atf models without loading weights.

Scans configured directories (recursively) and reads only each file's
128-byte header to report name, format version, and size. This is what
lets the app start with ZERO models resident and load on demand.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"ATF1"
_HEADER_SIZE = 128


@dataclass
class ModelInfo:
    name: str          # display name, e.g. "v2/9b" or "Qwen3.5-9B-BF16"
    path: Path
    format_major: int  # 1 = legacy (v1-v7 layout), 2 = LOD-aware layout
    format_minor: int
    size_bytes: int

    @property
    def id(self) -> str:
        return self.name


def peek_header(path: Path) -> tuple[int, int] | None:
    """Read just the header's magic + version. Returns (major, minor) or None."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(_HEADER_SIZE)
        if len(hdr) < _HEADER_SIZE or hdr[:4] != MAGIC:
            return None
        # Header.pack: <4sHHhhhHhiiBBII -> magic, vmajor(H), vminor(H), ...
        major, minor = struct.unpack_from("<HH", hdr, 4)
        return major, minor
    except OSError:
        return None


# Canonical location: <project root>/models. The registry scans this
# directory non-recursively for *.atf files, so the supported layouts are:
#   models/MyModel.atf              (flat — preferred for new installs)
#   models/v1/<name>.atf            (legacy format)
#   models/v2/<name>.atf            (LOD-aware)
#   models/v4/<name>.atf            (MLX-native + GGUF-quantization)
# i.e. an ATF model can live directly under models/ OR under a versioned
# subdirectory; both are picked up.
MODELS_ROOT = Path(__file__).resolve().parents[3] / "models"
# Alias kept for backwards compatibility with anything that still imports it.
MODELS_DIR = MODELS_ROOT / "v4"
# The default scan target is the *root* (so flat files are picked up too);
# `default_models_dirs()` callers see every *.atf directly under models/.
_DEFAULT_MODEL_DIR = MODELS_ROOT


def set_models_root(path) -> Path:
    """Runtime override of the models root (rebinds at call time).

    Electron passes the user's configured models directory to the bridge
    ($ATF_MODELS_DIR at spawn, {"type":"set_models_dir"} at runtime) and the
    bridge calls this, so every scan resolves against the real location
    instead of the compile-time parents[3]/models guess.
    """
    global _DEFAULT_MODEL_DIR
    p = Path(path).expanduser()
    _DEFAULT_MODEL_DIR = p
    return p


def get_models_root() -> Path:
    return _DEFAULT_MODEL_DIR


def default_models_dirs() -> list[Path]:
    # Scan the root (catches flat *.atf files) plus every direct subdirectory
    # that contains at least one *.atf. That way users can drop models in
    # either the canonical location (`models/MyModel.atf`) or alongside their
    # ancestors (`models/v4/MyModel.atf`, `models/v1/MyModel.atf`, ...).
    if not _DEFAULT_MODEL_DIR.is_dir():
        return [_DEFAULT_MODEL_DIR]
    dirs: list[Path] = [_DEFAULT_MODEL_DIR]
    for child in sorted(_DEFAULT_MODEL_DIR.iterdir()):
        if child.is_dir() and any(child.glob("*.atf")):
            dirs.append(child)
    return dirs


def scan_models(dirs: list[Path] | None = None) -> list[ModelInfo]:
    """Find every readable .atf file under `dirs` (recursive).

    Naming convention (supported layout):
        models/<File>.atf          -> id "<File>"          (top-level)
        models/v1/<name>.atf       -> id "v1/<name>"       (legacy format)
        models/v2/<name>.atf       -> id "v2/<name>"       (LOD-aware)
    i.e. the id is the path relative to the models root, suffix stripped,
    so the same model can exist as both v1/9b and v2/9b.
    """
    if dirs is None:
        dirs = default_models_dirs()
    out: dict[str, ModelInfo] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        # rglob picks up files at any depth (v4/9b/Qwen-...atf also works).
        # The id is computed relative to the models root so the same file
        # gets the same name regardless of which directory yielded it.
        for p in sorted(d.rglob("*.atf")):
            ver = peek_header(p)
            if ver is None:
                continue
            try:
                size = p.stat().st_size
                try:
                    rel = p.relative_to(_DEFAULT_MODEL_DIR)
                except ValueError:
                    rel = p.relative_to(d)
                name = "/".join(rel.with_suffix("").parts)
                info = ModelInfo(name=name, path=p, format_major=ver[0],
                                 format_minor=ver[1], size_bytes=size)
            except (OSError, ValueError):
                continue
            out.setdefault(str(p), info)
    return sorted(out.values(), key=lambda i: (i.name.lower(), i.path))


def resolve_model(ref: str, dirs: list[Path] | None = None) -> Path | None:
    """Resolve a model id like 'v2/9b', a bare stem, or an exact path."""
    ref = ref.strip()
    p = Path(ref).expanduser()
    if p.suffix == ".atf" and p.exists():
        return p
    if dirs is None:
        dirs = default_models_dirs()
    matches = [i for i in scan_models(dirs) if i.id == ref or i.path.stem == p.stem]
    if len(matches) == 1:
        return matches[0].path
    if len(matches) > 1:  # ambiguous: prefer highest format version
        matches.sort(key=lambda i: (-i.format_major, str(i.path)))
        return matches[0].path
    return None
