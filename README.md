# ARK — LLM Training Engine from Scratch

### EKO · NOUS Project · IAsesoria Informática · Villarrica, Chile · 2026


> 🇪🇸 [Versión en español](README_ES.md)

> No PyTorch. No TensorFlow. No cloud GPU.

---

**Live Training Dashboard:** [https://iasesoria.github.io/ARK/](https://iasesoria.github.io/ARK/)

---

## What it is

ARK is a large language model training engine written entirely from scratch in **Rust, Objective-C, and AArch64 NEON assembly**. It depends on neither PyTorch nor TensorFlow. Every layer of the compute stack — GPU forward pass, GPU backward pass via MPSGraph AutoGrad, assembly optimizer, custom math kernels — is written to squeeze the maximum out of Apple Silicon, specifically M1.

The first model ARK trains is called **EKO**, part of the **NOUS Project**. EKO is a 237-million-parameter transformer trained primarily on Spanish-language corpora — encyclopedic, mathematical, and conversational content — all built and refined specifically for this project.

My goal is to build an AI from scratch, on consumer hardware, in Spanish, from Chile, going against the grain by avoiding standard frameworks — and in the process, to learn and pursue the curiosity that started months ago trying to answer: "Is it possible to create artificial intelligence with more control and autonomy from its foundations?"

> **Status:** Epoch 1 is actively training. Full training history, corpus corrections, and architecture decisions are documented in [CHANGELOG.md](CHANGELOG.md).
---

## Why it matters

Most language models are trained on NVIDIA GPU clusters costing tens or hundreds of thousands of dollars. ARK pursues three things:

**1. Real hardware efficiency.**
A MacBook Air M1 with 8GB is mass-market consumer hardware. ARK makes the most of it: Zero-Copy architecture over unified memory, GPU forward pass via MPSGraph with AOT-compiled graphs, native GPU backward pass via AutoGrad with `gradientForPrimaryTensor:withTensors:`, AdamW optimizer in SIMD assembly. No unnecessary CPU↔GPU copies. No framework overhead.

**2. Technological sovereignty.**
The corpus is in Spanish. The project is Chilean. The model is entirely our own. My aim is to avoid external API dependencies as much as possible — no third-party language models, no third-party datasets, no cloud infrastructure. Training and inference are 100% local on my M1 Mac.

**3. Full reproducibility.**
ARK compiles and runs on any Mac with Apple Silicon. The only requirements are Rust and Xcode Command Line Tools. The code is transparent: what you read is exactly what executes.

---

## System architecture

The pipeline divides work across the three compute units available in the M1 chip:

### GPU — MPSGraph (full forward + native AutoGrad backward)

The complete forward pass runs on the GPU via Metal Performance Shaders Graph (MPSGraph), with AOT-compiled graphs built before the first training step:

- **Inference architecture (EKO):** 30-layer transformer (RMSNorm → Multi-head SDPA Attention → SwiGLU FFN). Implements Rotary Position Embedding (RoPE) with precomputed sin/cos tables and an LM Head with weight tying to the embedding.
- **Accelerated forward pass (GPU):** Executed on Apple Silicon via MPSGraph, with AOT-compiled graphs before the first step and a causal mask fused natively into the SDPA.
- **Backward pass (AutoGrad):** The backward pass for all 30 layers and the LM Head runs on the GPU via MPSGraph's native AutoGrad (`gradientForPrimaryTensor:withTensors:`). Symbolic differentiation automatically generates gradients for all 10 tensors per layer (dx, dwq, dwk, dwv, dwo, dw1, dw2, dw3, dg1, dg2). No manual chain-rule implementation.
- **Memory optimization (UMA):** Zero-Copy AdamW optimizer. Leverages M1's unified memory architecture by writing directly to shared `MTLBuffer` buffers in FP16, eliminating `memcpy` bottlenecks.
- **Training pipeline:** Automatic Mixed Precision (AMP) with dynamic loss scaling, zero-copy corpus streaming without intermediate files, and a fault-tolerant rotating checkpoint system (3 slots on disk).

Cross-entropy (numerically stable log-sum-exp) and embedding scatter-add run on the CPU via Accelerate.

### CPU — Accelerate / AMX

- Numerically stable cross-entropy using `vDSP_maxv` and log-sum-exp
- Gradient scatter-accumulation into embedding rows (sequential, handles repeated tokens)
- Global L2 gradient norm via `vDSP_svesq` before the Adam step

### AArch64 NEON Assembly — optimizer and math kernels

The AdamW optimizer and math kernels are written directly in AArch64 assembly:

- **`asm/opti.s`** — Full AdamW: vectorized Adam formula, 4 floats per SIMD cycle, bias correction, weight decay, global L2 grad clip.
- **`asm/kern.s`** — RMSNorm FP32/FP16 (forward + backward v0.62 with 4 fixed bugs), softmax FP32/FP16 (3-pass, underflow-safe), SwiGLU forward FP16, gather embedding FP16→FP32, dequant/quant FP16↔FP32 (width 8), FP16 dot product with FP32 accumulation.
- **`asm/aten.metal`** — Metal attention kernels: `attention_scores_f16`, `attention_softmax_f32`, `attention_weighted_sum_f16`, fused `attention_full_f16_causal` kernel.

### Zero-Copy over unified memory

All model weights live in `MTLBuffer` with `storageModeShared`. CPU and GPU access the same physical memory block — no intermediate copies. The assembly optimizer receives a direct pointer to VRAM, reads FP16 weights, updates them in-place, and writes them back to the same address the GPU will read on the next forward pass.

---

## EKO — Model specifications

| Parameter               | Value                                          |
|-------------------------|------------------------------------------------|
| Total parameters        | 237M                                           |
| Transformer layers      | 30                                             |
| d_model                 | 768                                            |
| Attention heads         | 12 (head_dim = 64)                             |
| FFN hidden              | 2048 (SwiGLU activation)                       |
| Vocabulary              | BPE 32,063 tokens (SentencePiece, Spanish)     |
| Positional encoding     | RoPE                                           |
| Normalization           | RMSNorm (gamma excluded from weight decay)     |
| Precision               | AMP: FP16 weights / FP32 gradients and moments |

> On M1 8GB, larger seq and batch sizes are constrained by available unified memory. This is why community testing on higher-RAM hardware is especially valuable — see the hardware testing section below.

---

## Automatic Mixed Precision (AMP)

Weights are stored in FP16 in VRAM. Adam gradients and moments (m, v) are kept in FP32. The dynamic scaler starts at 256 and can rise to 8,192 in ×2 steps every 2,000 clean steps. If NaN/Inf appears, the step is discarded, the scale is halved, and training continues from the intact checkpoint without corruption of the optimizer state.

---

## Tokenizer

EKO uses a BPE tokenizer trained with SentencePiece on Spanish-language corpora.

**Active file:** `tokenizador_bpe_32k_v2.model` — 32,063 tokens

The base vocabulary of 32,000 tokens was extended by +63 tokens to cover mathematical, scientific, and ASCII characters that appear at high frequency in the corpus but were not represented:

- Lowercase Greek letters (25): α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω
- Uppercase Greek letters (10): Γ Δ Θ Λ Σ Φ Χ Ψ Ω Π
- Superscripts (5): ² ³ ¹ ⁴ ⁰
- Subscripts (3): ₀ ₁ ₂
- Mathematical operators (8): ° × √ ≈ ∫ → ± ·
- Fraction / currency / ordinal (4): ½ € º ª
- Missing ASCII (8): & # \ ~ ^ @ ` ÷

> **Pending v3:** The Spanish diaeresis `ü`/`Ü` was not included in the expansion and was transliterated to `u` during corpus preprocessing (pingüino → pinguino). The impact on Epoch 1 is marginal given that these are ~250 low-frequency words in encyclopedic text. To be evaluated for correction in Epoch 2: remove the transliteration from the cleaner and add `ü`/`Ü` to the vocabulary (+2 tokens → 32,065).

---

## Corpus and training curriculum

**Epoch 1:** `wiki_esencial19.jsonl` — filtered and cleaned Spanish Wikipedia.

| Stat               | Value                |
|--------------------|----------------------|
| Articles           | 340,275              |
| Estimated tokens   | ~518M                |
| Disk size          | ~2.1 GB              |
| Tokens per doc     | ~1,524 (average)     |

The corpus was processed to remove non-Latin characters (Cyrillic, CJK, Arabic) that had no BPE vocabulary coverage and caused `<unk>` tokens at inference time. Token counts are computed dynamically at startup by sampling 1,000 documents — no hardcoded estimates.

The corpus is not distributed with the repository due to its size. To use your own, any JSONL file with a `"title"` and `"text"` field per line works directly with `--corpus`.

**Epoch 2 (planned):** Spanish reasoning corpus (~176K synthetic examples) aimed at improving logical coherence and response quality.

---

## Building and running

**Requirements:**
- macOS with Apple Silicon (M1 or later)
- Rust toolchain (`rustup`)
- Xcode Command Line Tools

**Build:**
```bash
cd ark01
cargo build --release
```

**Start training from scratch:**

Pass a checkpoint name that does not exist. ARK detects the missing file and initializes weights with Xavier automatically.

```bash
cd ~/Documents/ark/rust/ark01

nohup caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial19.jsonl \
  --vocab=../entren/tokenizador_bpe_32k_v2.model \
  --ckpt=../entren/ckpt_ark_ep1_rot0.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=1024 --batch=1 --lr=5e-5 --clip=0.5 \
  --epochs=1 >> ../entren/ark_ep1.log 2>&1 &
```

**Resume from checkpoint:**

Pass the most recent available checkpoint. ARK restores FP16 weights and FP32 Adam moments and continues from the saved step.

```bash
cd ~/Documents/ark/rust/ark01

nohup caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial19.jsonl \
  --vocab=../entren/tokenizador_bpe_32k_v2.model \
  --ckpt=../entren/ckpt_ark_ep1_rot2.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=1024 --batch=1 --lr=5e-5 --clip=0.5 \
  --epochs=1 >> ../entren/ark_ep1_seq1024.log 2>&1 &
```

**Rotating checkpoint system:**

ARK saves automatically every 500 steps (epoch 1) or 1,000 steps (epoch 2) using 3 rotating slots: `_rot0.bin`, `_rot1.bin`, `_rot2.bin`. The slot is determined by `(step / ckpt_every) % 3`. This guarantees at least two recent checkpoints are always available in case of any failure.

**Monitor training:**
```bash
tail -f ../entren/ark_ep1_seq1024.log
```

**Inference (Ryzen / Windows):**
```powershell
$env:RUSTFLAGS="-C target-cpu=native"
cargo run --release --bin eko_infer -- \
  --ckpt ckpt_ark_ep1_rot2.bin \
  --vocab tokenizador_bpe_32k_v2.model \
  --prompt "La vida es"
```

---

## Tested hardware

| Hardware            | Role      | Status     |
|---------------------|-----------|------------|
| MacBook Air M1 8 GB | Training  | ✅ Active  |
| Ryzen 7 (Windows)   | Inference | ✅ Tested  |

Contributions from testing on other ARM hardware are welcome. If you managed to compile and run ARK on a different machine, open an issue with your results.

---

## Active research

ARK's design is informed by the following works:

- Vaswani et al. (2017) — *Attention Is All You Need*
- Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Touvron et al. (2023) — *LLaMA: Open and Efficient Foundation Language Models*
- Zhang & Sennrich (2019) — *Root Mean Square Layer Normalization*
- Micikevicius et al. (2018) — *Mixed Precision Training*

---

## Project structure

```
proyecto_ark/
├── ark01/                                # Source code
│   ├── asm/
│   │   ├── kern.s                        # RMSNorm, softmax, SwiGLU, embed, dequant
│   │   ├── opti.s                        # AdamW FP16/FP32, L2 grad clip (NEON)
│   │   └── aten.metal                    # Metal attention kernels
│   ├── eko_infer/                        # Inference source code (CPU/Windows)
│   ├── entren/                           # Example directory (corpus and checkpoints not in repo)
│   ├── objc/
│   │   └── bridge.m                      # MPSGraph fwd/bwd AutoGrad, cross-entropy
│   ├── src/
│   │   ├── main.rs                       # Entry point and argument parser
│   │   ├── config.rs                     # Hyperparameters and validation
│   │   ├── training.rs                   # Training loop, AMP, checkpointing
│   │   ├── optimizer.rs                  # Zero-Copy AdamW, global vDSP clip
│   │   ├── memory.rs                     # ModelWeights, AlignedVec, f16↔f32
│   │   ├── io.rs                         # JSONL CorpusStream, CheckpointV4, BPE
│   │   └── ffi.rs                        # Safe Rust↔ObjC↔ASM bindings
│   ├── tokenizador_bpe/                  # BPE tokenizer (included in repo)
│   │   ├── tokenizador_bpe_32k_v2.model  (SentencePiece BPE 32,063 tokens — active)
│   │   ├── tokenizador_bpe_32k.model     (original 32,000 token version — legacy)
│   │   ├── sentencepiece_model.proto     (SentencePiece model definition)
│   │   ├── tokenizador_bpe_32k.vocab     (human-readable vocabulary)
│   │   ├── tokenizer_hf.json             (HuggingFace format)
│   │   └── vocab_bpe_32k.json            (JSON vocab for Rust)
│   ├── build.rs                          # Compiles bridge.m + kern.s + opti.s
│   └── cargo.toml
├── LICENSE
├── README.md                             # English documentation
└── README_ES.md                          # Spanish documentation
```

---

## Contact and collaboration

The code is free (MIT), the time is not. If you need adaptation, integration, technical consulting, or co-development:

[benjaminalonsocarmona@gmail.com](mailto:benjaminalonsocarmona@gmail.com)

For companies billing using this code, I offer formal support with contract and invoice.

---

## About the project

Developed by **Benjamín Alonso Carmona Vega**, founder of [IAsesoria Informática](https://iasesoria.cl).

Villarrica, Chile · 2026

*Developed with assistance from Claude Sonnet (Anthropic) and Gemini Pro (Google) for documentation, debugging, and architectural review.*

---

*Step 1 recorded loss=10.47. The theoretical value for a uniform distribution over 32,063 tokens is log(32,063) ≈ 10.37. The difference is Xavier initialization. That's all — the model knows nothing yet. What comes next is the interesting part.*
