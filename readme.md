# ATF — Adaptive Tensor Format for Apple Silicon

**Introducing ATF (Adaptive Tensor Format)** — a custom model format and high-throughput inference runtime that converts quantized **GGUF and MLX** checkpoints into a single-file `.atf` container. With hand-written Metal dequantization compute kernels, zero-copy memory-mapped loading, GPU-resident weights, and an adaptive reasoning router, ATF pushes inference directly to Apple Silicon's unified memory bandwidth ceiling with instant load times.

**Latest Release:** [v0.17.0 on GitHub](https://github.com/amgadtewfik/atf/releases/tag/v0.17.0)

## The Problem ATF Solves

ATF ships hand-optimized Metal compute shaders for less-common or ultra-dense GGUF quantization formats (`IQ1_M`, `IQ4_NL`, `Q6_K`, `UD-Q2_K_XL`), so decode runs at **DRAM-bandwidth-bound speed** on Apple Silicon instead of stalling on dequantization:

| Kernel | Metal Kernel | Memory Throughput | Max Rel. Error |
|:---|:---|:---:|:---:|
| **IQ4_NL** | Native Metal compute shader | **74 GB/s** | $\le 2.1 \times 10^{-6}$ |
| **Q6_K** | Native Metal compute shader | **62 GB/s** | $\le 2.3 \times 10^{-6}$ |
| **IQ1_M** | Vectorized fast-path Metal shader | Bandwidth-bound | Verified loss-free |

On Apple Silicon with unified memory, these formats run memory-bandwidth-bound rather than compute- or kernel-bound. (Numbers above are ATF's own measured kernel throughput/error, not a comparison against other runtimes.)

## Models & Benchmarks

Prompt: fixed filler + question (217 words), `max_tokens=128`, `temperature=0` (greedy), `thinking=off`. Evaluated on Apple Silicon with 16 GB unified memory:

| Model | Checkpoint Size | Prefill Speed | Decode Speed |
|:---|:---:|:---:|:---:|
| **Qwen3.5-9B-IQ4_NL.atf** | 5.52 GB | **65.00 tok/s** *(274 tok, 4.20s)* | **14.04 tok/s** *(104 tok, 7.40s)* |
| **Qwen3.8-27B-UD-Q2_K_XL.atf** | 9.52 GB | **18.90 tok/s** *(274 tok, 14.50s)* | **4.97 tok/s** *(96 tok, 19.30s)* |
| **Qwen-AgentWorld-35B-A3B-UD-IQ2_M.atf** | 11.64 GB | **24.30 tok/s** *(274 tok, 11.30s)* | **6.93 tok/s** *(105 tok, 15.20s)* |

*All three models run comfortably within 16 GB unified memory without memory pressure or paging.*

## Getting Started

1. **[Download ATF Chat (DMG)](https://github.com/amgadtewfik/atf)** — drag to `/Applications`. Python, MLX, and Metal runtimes are fully bundled with zero configuration required.
2. **Download a Model**: Grab any `.atf` model from the repository into `~/Library/Application Support/atf-chat/models` — or let the app download it directly from the in-app **Models** tab.
3. **Start Chatting**: Select the model from the dropdown to start chatting.

### OpenAI-Compatible API

ATF Chat ships with a built-in server compatible with standard OpenAI client tooling (OpenWebUI, LibreChat, ChatBox, Cursor):

```sh
# Point any OpenAI client at:
http://localhost:8000/v1/chat/completions
```

Supports Server-Sent Events (SSE) streaming and drop-in integration with any OpenAI chat completions client.

## Key Features

- **Single mmap-able File**: Zero sidecar configurations or split shards. A unified container houses metadata, dimensions, dtypes, tokenizer tables, and 16-byte-aligned tensor weights.
- **Hand-Optimized Metal Compute Kernels**: Dedicated vectorized shaders for `IQ1_M`, `IQ4_NL`, and `Q6_K` keep decode throughput pinned to hardware bandwidth limits.
- **Zero-Copy Instant Cold Starts**: Models memory-map directly into unified GPU memory with instant startup and complete memory reclamation when switching models.
- **Live Performance HUD & TTFT**: Real-time monitoring tracks Time for First Token (TTFT), prefill throughput, decode speed, active memory footprint, and token counts.
- **Adaptive Reasoning Router**: Automatically scores prompt complexity and dynamically adjusts thinking budget, sampling temperature, and token allocation across *instant*, *chat*, *reasoning*, and *deep* tiers.
- **64k Context Window**: High-capacity chunked prefill engine backed by an automatically growing KV cache.
- **Persistent Conversation History**: Multi-conversation organization with SQLite persistence, rich Markdown formatting, syntax highlighting, and JSON export.
- **Fast Rust-Backed Tokenizer**: Utilizes `gigatoken` for high-throughput encoding and decoding.

## Storage-Tier Adaptivity (Experimental)

ATF's runtime supports a dual-tier execution model, built to let a model dynamically trade precision for speed per turn:

- **Base Tier**: a lightweight quantized tier intended for fast, simple turns.
- **Residual Delta Tier**: a higher-precision correction, added on top of the Base tier via a fused dual-buffer Metal kernel, intended for harder turns (code, reasoning, math) without keeping a second full copy of the weights on disk.
- **Memory-Aware Tier Selection**: a headroom check (Darwin `host_statistics64`) is designed to clamp execution to the Base tier automatically if free unified memory drops too low, to avoid OOM/jetsam termination.

This is implemented in the runtime (`atf convert --tiers residual`, the fused kernel, and the tier-selection logic), but **has not yet been validated end-to-end on real hardware or benchmarked for quality/perf impact** — no `--tiers residual` model is published yet. Treat this section as a description of the design, not a performance claim.

## Roadmap

The tiered-precision loading the format's name points at — a Base tier plus a residual Delta tier that's loaded on demand for harder prompts — is implemented in the runtime (`atf convert --tiers residual`, a fused Metal kernel that adds the delta without materializing a dequantized copy, memory-aware tier selection, and page-aligned on-demand loading/eviction) but not yet validated end-to-end on real hardware or benchmarked for quality impact — no `--tiers residual` model is published yet. That validation is the near-term priority; after it, graph-level fusion (`mx.compile`) for a further decode speedup is next.

## Converting Your Own Models

Convert any GGUF checkpoint or MLX / Hugging Face safetensors into a single `.atf` file:

```sh
# From GGUF (preserves quant schemes, links fast Metal kernels)
python -m atf.cli convert model.gguf --raw -o MyModel.atf

# From MLX / safetensors checkpoints
python -m atf.convert_mlx <mlx_dir> -o MyModel.atf
python -m atf.convert <model_dir> -o MyModel.atf
```

## File Format

```
offset 0   magic "ATF1"          4 bytes
offset 4   version major/minor   2 × uint16
...        header (128 bytes)    dims, dtypes, tokenizer, LOD table
...        weights               mmap-able, 16-byte aligned
```

The container format maps MLX-native tensors carrying GGUF quantization schemes on a 16-byte-aligned layout. Model load time is an instantaneous GPU mapping rather than an expensive deserialization pass.

## Requirements

- Apple Silicon Mac, 16 GB unified memory minimum
- macOS 13+

## License

- Model weights adhere to the [Qwen License](https://huggingface.co/Qwen).
- The ATF format and runtime are licensed under Apache-2.0.
