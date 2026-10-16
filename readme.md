# ATF — Adaptive Tensor Format for Apple Silicon

**Introducing ATF (Adaptive Tensor Format)** — a custom model format and high-throughput inference runtime that converts quantized **GGUF and MLX** checkpoints into a single-file `.atf` container. It has hand-written Metal dequantization kernels, GPU-resident weights, on-demand loading of mixture-of-experts (MoE) experts, and an adaptive reasoning router. Decode runs close to Apple Silicon's memory-bandwidth limit, and a 35B MoE model runs in 6.4 GiB of GPU memory.

**Latest release: [ATF Chat v0.19.1](https://github.com/amgadtewfik/atf/releases/tag/v0.19.1).** v0.19.1 fixes in-app updates, which downloaded but did not install. v0.19.0 added on-demand expert loading (35B-A3B: 10.9 → 6.4 GiB of GPU memory) and the Spark-X2.5 and Qwen3.5-4B models.

## The Problem ATF Solves

ATF ships hand-optimized Metal compute shaders for 19 GGUF quantization formats, including the less-common and ultra-dense ones (`Q2_K`–`Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`/`XS`/`S`, `IQ3_XXS`/`S`, `IQ4_NL`/`XS`, and the `UD` mixes built from them). Decode therefore runs at **DRAM-bandwidth-bound speed** on Apple Silicon instead of stalling on dequantization. Every kernel is tested against a CPU reference on real model weights:

| Kernel | Metal Kernel | Memory Throughput | Max Rel. Error |
|:---|:---|:---:|:---:|
| **IQ4_NL** | Native Metal compute shader | **74 GB/s** | $\le 2.1 \times 10^{-6}$ |
| **Q6_K** | Native Metal compute shader | **62 GB/s** | $\le 2.3 \times 10^{-6}$ |
| **IQ1_M** | Vectorized fast-path Metal shader | Bandwidth-bound | Verified loss-free |

On Apple Silicon with unified memory, these formats run memory-bandwidth-bound rather than compute- or kernel-bound. (Numbers above are ATF's own measured kernel throughput/error, not a comparison against other runtimes.)

## Models & Benchmarks

### Mixture-of-experts: the 35B-A3B in about 6 GiB (v0.19)
A MoE model uses only 8 of its 256 experts per layer for each token. ATF keeps 128 experts per layer in GPU memory and reads the others from the `.atf` file when the router picks them. Each expert's weights are contiguous in the file, so loading one is three small reads (gate, up, down) straight into a GPU slot. Greedy replies are token-for-token identical to keeping every expert loaded (tested on 5 prompts at 32, 64 and 128 slots). It turns on automatically for a MoE model that would leave less than 2 GiB of GPU memory free. The cost is decode speed: about 75% of the all-loaded speed.

| Qwen-AgentWorld-35B-A3B-UD-IQ2_M (16 GB Mac) | All experts loaded | v0.19 (experts on demand) |
|:---|:---:|:---:|
| GPU memory after load | 10.9 GiB | **6.4 GiB** |
| Peak memory | 11.9 GiB (above the 11.84 GiB GPU working set) | **7.4–10 GiB** |
| Load time | 5 s (23 s when memory is busy) | **under 2 s** |
| Decode speed | 17–20 tok/s | 12.5–15 tok/s |
| 5,400-token agent prompt through the app | ran out of GPU memory (2026-09-29) | **80 s; the follow-up turn 1.6 s** |

### Earlier benchmarks (v0.17, before on-demand experts and the MoE dispatch work)
Prompt: fixed filler + question (217 words), `max_tokens=128`, `temperature=0` (greedy), `thinking=off`. Evaluated on Apple Silicon with 16 GB unified memory:

| Model | Checkpoint Size | Prefill Speed | Decode Speed |
|:---|:---:|:---:|:---:|
| **Qwen3.5-9B-IQ4_NL.atf** | 5.52 GB | **65.00 tok/s** *(274 tok, 4.20s)* | **14.04 tok/s** *(104 tok, 7.40s)* |
| **Qwen3.8-27B-UD-Q2_K_XL.atf** | 9.52 GB | **18.90 tok/s** *(274 tok, 14.50s)* | **4.97 tok/s** *(96 tok, 19.30s)* |
| **Qwen-AgentWorld-35B-A3B-UD-IQ2_M.atf** | 11.64 GB | **24.30 tok/s** *(274 tok, 11.30s)* | **6.93 tok/s** *(105 tok, 15.20s)* |

*In v0.19, every model the app offers passes the release gate on a 16 GB Mac: real replies through the app's own engine, and peak GPU memory within the working set (35B-A3B 9.3 GB, 27B 11.7–11.9 GB, 9B 9.6 GB, 4B models 2.7–6.9 GB).*

## Getting Started

1. **[Download ATF Chat v0.19.1 (DMG)](https://github.com/amgadtewfik/atf/releases/download/v0.19.1/ATF.Chat-arm64.dmg)** ([release notes](https://github.com/amgadtewfik/atf/releases/tag/v0.19.1)) — drag to `/Applications`. Python, MLX, and Metal runtimes are fully bundled with zero configuration required.
   *Coming from v0.18.0 or v0.19.0?* Install v0.19.1 from the DMG once. Those versions can download updates but not install them; from v0.19.1 on, **File → Check for Updates…** installs and relaunches by itself.
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

- **Single Self-Contained File**: Zero sidecar configurations or split shards. A unified container houses metadata, dimensions, dtypes, the model's own tokenizer and chat template (copied from the source GGUF), and 16-byte-aligned tensor weights. Nothing is downloaded from Hugging Face at runtime.
- **Hand-Optimized Metal Compute Kernels**: Dedicated vectorized shaders for 19 GGUF formats keep decode throughput pinned to hardware bandwidth limits. A MoE layer's routed experts run in 3 GPU dispatches.
- **On-Demand MoE Experts**: A MoE model keeps a cache of experts in GPU memory and loads the others from the file as the router picks them (35B-A3B: 6.4 instead of 10.9 GiB, decode about 75% of the all-loaded speed).
- **Fast Cold Starts**: Weights are read with large parallel reads straight into wired GPU memory, which macOS can't page out. The 35B-A3B loads in under 2 s, and memory is fully reclaimed when switching models.
- **Live Performance HUD & TTFT**: Real-time monitoring tracks Time for First Token (TTFT), prefill throughput, decode speed, active memory footprint, and token counts.
- **Adaptive Reasoning Router**: Automatically scores prompt complexity and dynamically adjusts thinking budget, sampling temperature, and token allocation across *instant*, *chat*, *reasoning*, and *deep* tiers.
- **64k Context Window**: High-capacity chunked prefill engine backed by an automatically growing KV cache. A prefill step that runs out of GPU memory is retried in smaller pieces instead of failing.
- **Prompt Reuse Across Turns**: Shared prefixes (the system prompt, earlier turns) are reused exactly, recurrent state included, so a follow-up on a 5,400-token agent prompt answers in under 2 s.
- **Persistent Conversation History**: Multi-conversation organization with SQLite persistence, rich Markdown formatting, syntax highlighting, and JSON export.
- **Fast Rust-Backed Tokenizer**: Hugging Face `tokenizers`, built at load time from the tokenizer embedded in the `.atf`.
- **Self-Installing Updates**: The app checks GitHub releases, downloads the update, replaces itself and relaunches (from v0.19.1).

## Storage-Tier Adaptivity (Experimental)

ATF's runtime supports a dual-tier execution model, built to let a model dynamically trade precision for speed per turn:

- **Base Tier**: a lightweight quantized tier intended for fast, simple turns.
- **Residual Delta Tier**: a higher-precision correction, added on top of the Base tier via a fused dual-buffer Metal kernel, intended for harder turns (code, reasoning, math) without keeping a second full copy of the weights on disk.
- **Memory-Aware Tier Selection**: a headroom check (Darwin `host_statistics64`) is designed to clamp execution to the Base tier automatically if free unified memory drops too low, to avoid OOM/jetsam termination.

This is implemented in the runtime (`atf convert --tiers residual`, the fused kernel, and the tier-selection logic), but **has not yet been validated end-to-end on real hardware or benchmarked for quality/perf impact** — no `--tiers residual` model is published yet. Treat this section as a description of the design, not a performance claim.

## Roadmap

- **Done (v0.19):** on-demand loading of MoE experts.
- **Next:** raise the expert cache's hit rate and hide the remaining reads, so on-demand experts cost less decode speed. The plan: evict by frequency instead of recency, prefetch the next layer's likely experts, and put frequently used experts first in the file.
- **Then:** validate the storage tiers above on real hardware and publish a `--tiers residual` model.

## Known Limitations

- With on-demand experts, the 35B-A3B decodes at 12.5–15 tok/s instead of 17–20. Set `ATF_MOE_EXPERT_SLOTS=0` to keep every expert loaded (more memory, full speed).
- The 35B-A3B (an agent fine-tune) writes its reasoning as plain text ("Thinking Process: …") even with thinking off, before answering.
- On the 27B IQ-quant models, a short prompt (13 tokens) takes about 2 s to prefill.
- The app is not notarized: macOS warns the first time a downloaded copy is opened.

## Converting Your Own Models

Convert a GGUF checkpoint into a single `.atf` file:

```sh
# Keeps the GGUF's quantized bytes unchanged and uses the fast Metal kernels
python -m atf.cli convert model.gguf --raw -o MyModel.atf
```

The converter copies the GGUF's own tokenizer and chat template into the `.atf`. The runtime builds the tokenizer only from that section and refuses a file without one. Re-convert older `.atf` files from their GGUF.

## File Format

```
offset 0   magic "ATF1"          4 bytes
offset 4   version major/minor   2 × uint16
...        header (192 bytes, v5) dims, dtypes, LOD table, vocab-section offset
...        vocab section         the model's tokenizer + chat template
...        weights               16-byte aligned, original GGUF bytes
```

The container keeps the source GGUF's quantized bytes unchanged on a 16-byte-aligned layout, so loading is a straight copy into GPU memory rather than a re-quantization or deserialization pass, and one MoE expert can be read on its own.

## Requirements

- Apple Silicon Mac, 16 GB unified memory minimum
- macOS 13+

## License

- Model weights adhere to the [Qwen License](https://huggingface.co/Qwen).
- The ATF format and runtime are licensed under Apache-2.0.
