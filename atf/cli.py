"""ATF command-line interface: convert, inspect, chat."""
from __future__ import annotations

import hashlib
import os
import signal
import struct
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__

console = Console()

MANIFEST_ENTRY_FMT = "<QBBBBBQQQI"
MANIFEST_ENTRY_SIZE = struct.calcsize(MANIFEST_ENTRY_FMT)


def _fail(msg: str) -> None:
    console.print(f"[red]error:[/red] {msg}")
    raise SystemExit(1)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _sha256_chunked(path: Path, nbytes: int | None = None) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = nbytes
        while True:
            to_read = (1 << 24) if remaining is None else min(1 << 24, remaining)
            if to_read <= 0:
                break
            chunk = f.read(to_read)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.digest()


@click.group()
@click.version_option(version=__version__, prog_name="atf")
def cli() -> None:
    """ATF — Adaptive Tensor Format (dynamic MoE + Graphic LOD)."""


# ─── convert ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("gguf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Output .atf file path")
@click.option("--experts", default=8, show_default=True, type=int,
              help="Experts per block (must divide the FFN intermediate dim)")
@click.option("--top-k", default=4, show_default=True, type=int,
              help="Default expert activation count")
@click.option("--ffn-pattern", default=r"blk\.(\d+)\.ffn_(gate|up|down)\.weight",
              show_default=False, help="Regex identifying FFN tensors to split")
@click.option("--expert-storage", default="auto",
              type=click.Choice(["auto", "lod0", "dual", "lod1", "lod1-all"]),
              show_default=True,
              help="v2 expert storage: lod1/lod1-all = smallest files (4-bit), "
                   "dual = adaptive INT8+4-bit, lod0 = legacy v1 layout")
@click.option("--dense-fp4/--dense-int8", "dense_fp4", default=False, show_default=True,
              help="Store remaining dense 2-D tensors as FP4 (E2M1, g64) instead of INT8")
@click.option("--raw", is_flag=True, default=False,
              help="v3 raw-GGUF mode: keep original quantization byte-for-byte "
                   "(loaded by the packed Metal-kernel path)")
def convert(gguf: Path, output: Path, experts: int, top_k: int, ffn_pattern: str,
            expert_storage: str, dense_fp4: bool, raw: bool) -> None:
    """Convert a GGUF model to ATF format."""
    if experts < 1:
        _fail("--experts must be >= 1")
    if top_k < 1 or top_k > experts:
        _fail("--top-k must be between 1 and --experts")
    if output.exists():
        _fail(f"output already exists: {output} (delete it first)")

    from .convert import ConvertConfig
    from .convert import convert as run_convert
    


    # ── FIX: auto-route pre-quantized GGUF sources to raw mode ─────────────
    # Pre-quantized GGUFs (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ*, UD-*) already
    # use 4-bit or less per weight. Re-quantizing to INT8 (dense) + adding a
    # 4-bit merged FFN (LOD1) produces an output LARGER than the source
    # (~2× for Dirk-Qwen3.8-27B-UD-Q2_K_XL: 9.8 GB → 19 GB). The correct
    # path for pre-quantized sources is `convert_raw`, which preserves the
    # original GGUF bytes byte-for-byte and produces a file smaller than
    # or equal to the source.
    if not raw:
        src_name = gguf.name.upper()
        is_quantized = any(t in src_name for t in
                           ("Q2_", "Q3_", "Q4_", "Q5_", "Q6_", "IQ", "UD-"))
        if is_quantized:
            console.print(f"  [cyan]Pre-quantized source detected → using raw mode "
                          f"(preserves original GGUF bytes, no re-quantization).[/cyan]")
            raw = True

    result = run_convert(ConvertConfig(expert_storage=expert_storage,
                    dense_fp4=dense_fp4,
                    mode="raw" if raw else "legacy",
                    gguf_path=gguf,
        output_path=output,
        num_experts=experts,
        top_k=top_k,
        ffn_pattern=ffn_pattern,
    ))

    console.print(f"  Experts (LOD 0): {result.num_experts}")
    console.print(f"  Compression: {result.total_size / max(result.original_size, 1):.2f}x "
                  f"({_fmt_bytes(result.total_size)} from {_fmt_bytes(result.original_size)})")


# ─── convert-mlx ──────────────────────────────────────────────────────────────

@cli.command(name="convert-mlx")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Output .atf file path")
@click.option("--bits", default=4, show_default=True, type=click.IntRange(2, 8),
              help="Target quantization bits for matmul weights (source is dequantized once)")
def convert_mlx_cmd(source_dir: Path, output: Path, bits: int) -> None:
    """Convert an MLX safetensors checkpoint directory to ATF format."""
    if output.exists():
        _fail(f"output already exists: {output} (delete it first)")

    from .convert_mlx import convert_mlx as run_convert

    result = run_convert(source_dir, output, target_bits=bits)
    console.print(f"  Blocks: {result['blocks']}  "
                  f"Size: {result['size'] / 1e9:.3f} GB  "
                  f"Duration: {result['duration'] / 60:.1f} min")




# ─── convert-moe ──────────────────────────────────────────────────────────────

@cli.command(name="convert-moe")
@click.argument("src", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Output .atf path (default: <src>_A2B.atf)")
@click.option("--num-experts", default=64, show_default=True, type=int,
              help="Number of experts per block (must divide ffe). Default 64 (A2B recipe).")
@click.option("--top-k", default=5, show_default=True, type=int,
              help="Active experts per token. Default 5 (A2B recipe).")
@click.option("--ffe-shex", default=None, type=int,
              help="Shared-expert intermediate dim (default: ffe // num_experts)")
@click.option("--router-dtype", default="f16", show_default=True,
              type=click.Choice(["f32", "f16"]),
              help="Router dtype (f32=A3B compat, f16=saves bytes)")
@click.option("--shexp-gate-bias", default=10.0, show_default=True, type=float,
              help="Constant for the shared-expert per-channel gate (sigmoid(bias) ~ 1.0)")
@click.option("--shexp-alias", "shexp_storage_flag", flag_value="alias", default=None,
              help="Alias the FULL dense FFN as shexp (legacy recipe, slow at inference: ~20s/token on the 27B).")
@click.option("--shexp-slice", "shexp_storage_flag", flag_value="slice", default=None,
              help="Dequant + slice + re-store the first ffe_shex columns of the dense FFN as shexp (default). Fast at inference.")
@click.option("--shexp-dtype", default="bf16", show_default=True,
              type=click.Choice(["f32", "bf16"]),
              help="Dtype for sliced shexp records. bf16 = half the size with ~3%% mantissa loss.")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--dry-run", is_flag=True, help="Plan only, do not write")
def convert_moe_cmd(src: Path, output: Path | None, num_experts: int,
                    top_k: int, ffe_shex: int | None,
                    router_dtype: str, shexp_gate_bias: float,
                    shexp_storage_flag: str | None, shexp_dtype: str,
                    seed: int, dry_run: bool) -> None:
    """Add MoE scaffold (router + shared expert) to a raw-GGUF .atf file.

    The output adds a ~250 MB scaffold to the source. The engine routes
    the FFN through the existing fast column-wise slicing path
    (`m.qmm("...e{e}", Xe)`), so no per-expert records are stored.
    v2 adds the sliced-shared recipe (default) for fast inference on
    narrow shexp, plus the legacy alias-shared recipe for full FFN aliasing.
    """
    from .convert_moe import (MoeConfig, convert_moe as run_convert,
                              GGUF_F32, GGUF_F16, GGUF_BF16, _default_output_path)

    dst = output if output else _default_output_path(src)
    if dst.exists():
        _fail(f"output already exists: {dst} (delete it first)")
    shexp_storage = shexp_storage_flag or "slice"
    shexp_dt_code = GGUF_F32 if shexp_dtype == "f32" else GGUF_BF16
    cfg = MoeConfig(
        src_path=src, dst_path=dst,
        num_experts=num_experts, top_k=top_k, ffe_shex=ffe_shex,
        router_dtype=GGUF_F32 if router_dtype == "f32" else GGUF_F16,
        shexp_gate_bias=shexp_gate_bias,
        shexp_storage=shexp_storage, shexp_dtype=shexp_dt_code,
        seed=seed, dry_run=dry_run,
    )
    try:
        result = run_convert(cfg)
    except Exception as exc:
        _fail(f"convert_moe failed: {exc}")
    if not dry_run:
        console.print(f"  Output: {_fmt_bytes(result['dst_bytes'])} "
                      f"({result['dst_bytes']/1e9:.3f} GB)")
        console.print(f"  Experts/block: {result['num_experts']}, "
                      f"top_k: {result['top_k']}, ffe_shex: {result['ffe_shex']}, "
                      f"shexp_storage: {result.get('shexp_storage', '?')}")


# ─── inspect ──────────────────────────────────────────────────────────────────

@cli.command(name="inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                metavar="FILE.atf")
@click.option("--verbose", is_flag=True, help="Per-LOD and per-domain breakdown")
def inspect_cmd(file: Path, verbose: bool) -> None:
    """Inspect an ATF file (header, manifest, checksum)."""
    from .format import (
        HEADER_SIZE, FOOTER_SIZE, Header, LodLevel,
    )

    size = file.stat().st_size
    if size < HEADER_SIZE + FOOTER_SIZE:
        _fail(f"not an ATF file (too small): {file}")

    with open(file, "rb") as f:
        header = Header.unpack(f.read(HEADER_SIZE))

    if header.offset_manifest == 0 or header.offset_manifest >= size:
        _fail("manifest offset missing — file may be incomplete")

    # Parse manifest without loading the whole file
    with open(file, "rb") as f:
        f.seek(header.offset_manifest)
        nblocks, nexp_per_blk = struct.unpack("<HH", f.read(4))
        (ndomains,) = struct.unpack("<H", f.read(2))
        domain_names = []
        for _ in range(ndomains):
            (slen,) = struct.unpack("<H", f.read(2))
            domain_names.append(f.read(slen).decode("utf-8"))
        (nsubs,) = struct.unpack("<H", f.read(2))
        sub_domain_names = []
        for _ in range(nsubs):
            (slen,) = struct.unpack("<H", f.read(2))
            sub_domain_names.append(f.read(slen).decode("utf-8"))
        (nexperts,) = struct.unpack("<I", f.read(4))
        raw = f.read(nexperts * MANIFEST_ENTRY_SIZE)
    entries = [struct.unpack_from(MANIFEST_ENTRY_FMT, raw, i * MANIFEST_ENTRY_SIZE)
               for i in range(nexperts)]
    # (eid, bidx, did, sid, spec, lod, woff, wsize, coff, cdim)

    t = Table(title=f"ATF: {file.name}", show_lines=False)
    t.add_column("Field", style="cyan")
    t.add_column("Value")
    t.add_row("Size", _fmt_bytes(size))
    t.add_row("Version", f"{header.version_major}.{header.version_minor}")
    t.add_row("Arch", str(header.arch))
    t.add_row("Blocks", str(header.num_blocks))
    t.add_row("Hidden", str(header.hidden_dim))
    t.add_row("Heads (KV)", f"{header.num_heads} ({header.num_kv_heads})")
    t.add_row("Vocab", str(header.vocab_size))
    t.add_row("Experts/block", str(header.num_experts_per_block))
    t.add_row("top_k", str(header.top_k))
    t.add_row("LOD levels", str(header.num_lod_levels))
    t.add_row("Experts in manifest", str(nexperts))
    console.print(t)

    by_lod: dict[int, list] = {}
    for e in entries:
        by_lod.setdefault(e[5], []).append(e)
    if by_lod:
        lt = Table(title="Experts by LOD" if verbose else "LOD summary")
        lt.add_column("LOD", style="cyan")
        lt.add_column("Count")
        lt.add_column("Bytes")
        for lod in sorted(by_lod):
            es = by_lod[lod]
            lt.add_row(f"LOD {lod}", str(len(es)), _fmt_bytes(sum(x[7] for x in es)))
        console.print(lt)

    if verbose and domain_names:
        dt = Table(title="Experts by domain")
        dt.add_column("Domain", style="cyan")
        dt.add_column("Experts")
        for d, name in enumerate(domain_names):
            n = sum(1 for e in entries if e[2] == d)
            dt.add_row(name, str(n))
        console.print(dt)

    console.print("  Verifying checksum (full file scan)...")
    computed = _sha256_chunked(file, size - FOOTER_SIZE)
    with open(file, "rb") as f:
        f.seek(size - FOOTER_SIZE)
        stored = f.read(FOOTER_SIZE)
    ok = computed == stored
    console.print(f"  Checksum: [green]OK[/green]" if ok else f"  Checksum: [red]MISMATCH[/red]")
    if not ok:
        raise SystemExit(1)


# ─── chat ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                metavar="FILE.atf")
@click.option("--prompt", default=None, help="Prompt text (omit for interactive REPL)")
@click.option("--max-tokens", default=128, show_default=True, type=int)
@click.option("--temperature", default=0.0, show_default=True, type=float,
              help="0 = greedy")
@click.option("--top-p", default=0.9, show_default=True, type=float)
@click.option("--repeat-penalty", default=1.1, show_default=True, type=float)
@click.option("--seed", default=None, type=int, help="RNG seed for reproducibility")
@click.option("--mem-log-interval", default=20, show_default=True, type=int,
              help="Print host/GPU memory every N generated tokens (0 to disable)")
@click.option("--cache-gb", default=None, type=float,
              help="F32 weight-dequant LRU cache size in GB (default: 3 GB, or $ATF_CACHE_GB). "
                   "With --exact-ffn, every forward pass touches every expert in every block; "
                   "a cache smaller than the full dequantized model causes it to thrash and "
                   "re-dequantize from int8 on almost every call. Set this near your available "
                   "free RAM (in GB) to let the model stay resident in f32 after first use.")
@click.option("--tokenizer", "tokenizer_src", default=None,
              help="HF repo id for gigatoken (default: $ATF_TOKENIZER or Qwen/Qwen3.5-4B)")
@click.option("--exact-ffn/--sparse-moe", default=True, show_default=True,
              help="Exact unweighted FFN evaluation vs sparse MoE routing")
@click.option("--diag/--no-diag", default=False, show_default=True,
              help="Print full per-block + logits diagnostics to the console")
@click.option("--chat-template/--raw", "use_chat_template", default=True, show_default=True,
              help="Format prompts with ChatML template (<|im_start|>...)")
@click.option("--max-context", default=None, type=int,
              help="KV capacity in tokens (default 65536). Lower this for "
                   "large models on 16 GB machines -- KV grows linearly.")
@click.option("--thinking", default="medium", show_default=True,
              type=click.Choice(["off", "low", "medium", "high", "xhigh"]),
              help="Reasoning effort: off answers directly; low/medium/high cap the "
                   "thinking budget (256/1024/4096 tokens); xhigh is unlimited")
def chat(file: Path, prompt: str | None, max_tokens: int, temperature: float,
         top_p: float, repeat_penalty: float, seed: int | None,
         mem_log_interval: int, cache_gb: float | None, tokenizer_src: str | None,
         exact_ffn: bool, use_chat_template: bool, diag: bool, thinking: str, max_context) -> None:
    from .engine import Engine, GenConfig
    from .model import load_atf
    from .tokenizer import Tokenizer

    if cache_gb is not None:
        cache_bytes = int(cache_gb * (1 << 30))
    elif os.environ.get("ATF_CACHE_GB"):
        cache_bytes = int(float(os.environ["ATF_CACHE_GB"]) * (1 << 30))
    else:
        cache_bytes = None  # model.py falls back to its 3 GB default

    model = load_atf(file, lru_cap_bytes=cache_bytes)
    if tokenizer_src is not None:
        src = tokenizer_src
    elif os.environ.get("ATF_TOKENIZER"):
        src = os.environ["ATF_TOKENIZER"]
    elif model.vocab_size > 200000 or "3.5" in file.name or "3_5" in file.name:
        src = "Qwen/Qwen3.5-9B"
    else:
        src = "Qwen/Qwen3.5-4B"

    console.print(f"  Tokenizer: {src}")
    try:
        tok = Tokenizer.from_hf(src)
    except Exception as exc:
        _fail(f"failed to load tokenizer {src!r}: {exc}")

    engine = Engine(model, tok)
    cfg = GenConfig(max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, repeat_penalty=repeat_penalty, seed=seed,
                    mem_log_interval=mem_log_interval, exact_ffn=exact_ffn, diag=diag,
                    thinking=thinking, max_context=max_context or 65536)

    def _format(p: str) -> str:
        if use_chat_template and "<|im_start|>" not in p:
            return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        return p

    # ── generation wall-time guard ────────────────────────────────────
    # A wedged Metal kernel or a runaway decode must never leave an `atf
    # chat` process alive for tens of minutes holding GB of wired memory
    # (observed: orphaned repro processes stuck 45+ min in uninterruptible
    # wait). Default 15 min, override with ATF_GEN_TIMEOUT_SEC (0=off).
    try:
        gen_timeout = float(os.environ.get("ATF_GEN_TIMEOUT_SEC", "900"))
    except ValueError:
        gen_timeout = 900.0
    if gen_timeout > 0 and prompt is not None and hasattr(signal, "SIGALRM"):
        def _on_timeout(signum, frame):
            sys.stderr.write(
                f"\n[atf] generation exceeded {gen_timeout:.0f}s wall time "
                "-- aborting to release memory/GPU (ATF_GEN_TIMEOUT_SEC).\n")
            raise SystemExit(70)
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(int(gen_timeout))

    if prompt is not None:
        try:
            engine.generate(_format(prompt), cfg, stream=True)
        finally:
            if gen_timeout > 0 and hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        return

    console.print("[dim]Interactive mode — type a prompt, 'exit' to quit.[/dim]")
    while True:
        try:
            line = console.input("[bold green]>>>[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            break
        text = engine.generate(_format(line), cfg, stream=True)
        console.print()


main = cli

if __name__ == "__main__":
    cli()
