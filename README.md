# ARK — LLM Training Engine from Scratch

### EKO · NOUS Project · IAsesoria Informática · Villarrica, Chile · 2026

## **Live Training Dashboard:** (https://iasesoria.github.io/ARK/)

> 🇨🇱 [Versión en español](README_ES.md)

> No PyTorch. No TensorFlow. No cloud GPU.

---

## What this is

ARK is a large language model training engine written entirely from scratch in **Rust, Objective-C, and AArch64 NEON assembly**. It has no dependency on PyTorch, TensorFlow, or any deep learning framework. Every layer of the compute stack — GPU forward pass, GPU backward pass via MPSGraph AutoGrad, assembly optimizer, custom math kernels — is written to be as optimized as possible for Apple Silicon, specifically M1.

The first model ARK trains is called **EKO**, part of the **NOUS Project**. EKO is a 237-million-parameter transformer trained primarily on Spanish-language corpora — encyclopedic, mathematical, and conversational content — all built and filtered specifically for this project.

The goal is to build an AI from scratch, on consumer hardware, in Spanish, from Chile — going against the grain by avoiding typical frameworks, and in the process learning and satisfying the curiosity that started this months ago.

---

## Why it matters

Most language models are trained on NVIDIA GPU clusters costing tens or hundreds of thousands of dollars. ARK pursues three things:

**1. Real efficiency on limited hardware.**
A MacBook Air M1 with 8GB is mass-market consumer hardware. ARK pushes it to its limits: Zero-Copy architecture over unified memory, AOT-compiled MPSGraph forward pass, native GPU backward via AutoGrad with `gradientForPrimaryTensor:withTensors:`, AdamW optimizer in SIMD assembly. No unnecessary CPU↔GPU copies. No framework overhead.

**2. Technological sovereignty.**
The corpus is in Spanish. The project is Chilean. The model is entirely homegrown. The goal is to avoid external APIs, third-party language models, and cloud infrastructure as much as possible. Training and inference are 100% local on my M1 Mac.

**3. Total reproducibility.**
ARK compiles and runs on any Mac with Apple Silicon. The only requirements are Rust and Xcode Command Line Tools. The code is transparent: what you read is exactly what executes.

---

## System architecture

The pipeline divides work across the three compute units available in the M1 chip:

### GPU — MPSGraph (full forward + native AutoGrad backward)

The complete forward pass runs on GPU via Metal Performance Shaders Graph (MPSGraph), with AOT-compiled graphs before the first training step:

- Embedding lookup (FP16)
- 30 transformer layers (EKO): RMSNorm → multi-head SDPA attention → SwiGLU FFN → RMSNorm
- Rotary Position Embedding (RoPE) with sin/cos tables precomputed on CPU
- Causal mask fused natively in SDPA
- LM head with weight tying to the embedding

The backward pass for all 30 layers and the LM head runs on GPU via MPSGraph native AutoGrad (`gradientForPrimaryTensor:withTensors:`). Each layer graph defines a `loss_proxy` as the reduction sum of `t_out × d_out`, and symbolic differentiation automatically generates gradients for all 10 tensors per layer (dx, dwq, dwk, dwv, dwo, dw1, dw2, dw3, dg1, dg2). No manual chain rule implementation.

Cross-entropy (numerically stable log-sum-exp) and embedding scatter-add run on CPU via Accelerate.

### CPU — Accelerate / AMX

- Numerically stable cross-entropy with `vDSP_maxv` and log-sum-exp
- Gradient scatter-accumulation across embedding rows (sequential, handles repeated tokens)
- Global L2 gradient norm with `vDSP_svesq` before the Adam step

### AArch64 NEON Assembly (optimizer and math kernels)

The AdamW optimizer and math kernels are written directly in AArch64 assembly:

- **`asm/opti.s`** — Full AdamW: vectorized Adam formula, 4 floats per SIMD cycle, bias correction, weight decay, global L2 grad clip.
- **`asm/kern.s`** — RMSNorm FP32/FP16 (forward + backward v0.62 with 4 bugs fixed), softmax FP32/FP16 (3-pass, underflow-safe), SwiGLU forward FP16, gather embedding FP16→FP32, dequant/quant FP16↔FP32 (width 8), FP16 dot product with FP32 accumulation.
- **`asm/aten.metal`** — Metal attention kernels: `attention_scores_f16`, `attention_softmax_f32`, `attention_weighted_sum_f16`, fused causal kernel `attention_full_f16_causal`.

### Zero-Copy over unified memory

All model weights live in `MTLBuffer` with `storageModeShared`. CPU and GPU access the same physical memory block — no intermediate copies. The assembly optimizer receives a direct pointer to VRAM, reads FP16 weights, updates them in-place, and writes them back to the same address the GPU will read on the next forward pass.

---

## EKO — Model specifications

| Parameter | Value |
|---|---|
| Total parameters | 237M |
| Transformer layers | 30 |
| d_model | 768 |
| Attention heads | 12 (head_dim = 64) |
| FFN hidden | 2048 (SwiGLU activation) |
| Context (epoch 1) | 512 tokens |
| Context (epoch 2+) | 1024 tokens |
| Vocabulary | BPE 32,000 tokens (SentencePiece, Spanish) |
| Positional encoding | RoPE |
| Normalization | RMSNorm (gamma excluded from weight decay) |
| Precision | AMP: FP16 weights / FP32 gradients and Adam moments |

### Active training configuration

```
Epoch 1 (linguistic base):
  --corpus=wiki_esencial14.jsonl,wiki_disambig.jsonl
  --layers=30 --heads=12 --d-model=768 --hidden=2048
  --seq=512 --batch=1 --lr=1e-4 --warmup=2000 --clip=0.5
  --epochs=1

AdamW: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
AMP:   loss_scale_init=256, max=8192, step_up_every=2000
Checkpoint: rotating 3-slot, every 500 steps
```

---

## Automatic Mixed Precision (AMP)

Weights are stored in FP16 in VRAM. Gradients and Adam moments (m, v) are kept in FP32. The dynamic scaler starts at 256 and can climb to 8192 in ×2 steps every 2,000 clean steps. If NaN/Inf appears, the step is discarded, the scale is halved, and training continues from the intact checkpoint without corrupting the optimizer state.

---

## Corpus and training curriculum

Epoch 1 corpus: **~617M real tokens** (calculated by sampling 1,000 documents at startup — no hardcoded estimates):

| Corpus | Content |
|---|---|
| `wiki_esencial14.jsonl` | Spanish Wikipedia — 341,147 filtered documents (2.1 GB) |
| `wiki_disambig.jsonl` | Wikipedia disambiguation pages — 63,113 documents |

**Planned epochs:**

| Epoch | Focus | Key corpora |
|---|---|---|
| **Epoch 1** | Linguistic base | Wikipedia + disambiguation. ~1,206,463 steps at seq=512 |
| **Epoch 2** | Reasoning and logic | GSM8K-ES, GSM-Hard, MCOT-Math, Aya-Reasoning, abduction corpus. seq=1024, lr=5e-5 |
| **Epoch 3+** | Dialogue and instruction | Alpaca-ES, Orca-ES, OpenSubtitles, Tatoeba, StackOverflow, NOUS identity corpus |

---

## Checkpoint format

Format v4 (magic bytes `ARK4`). Stores weights in native FP16 + Adam moments m and v in FP32. For 237M parameters: **2,369.6 MB per slot**.

Rotating 3-slot system. At least two valid copies are always simultaneously available. On resume, restores weights and complete optimizer state (271 tensors) to continue with intact accumulated Adam momentum. Compatible with legacy formats v2 (FP32 weights) and v3 (FP32 weights + moments).

---

## Project structure

```
proyecto_ark/
├── entren/                          # Corpus and artifacts
│   ├── wiki_esencial14.jsonl        (2.1 GB — 341,147 docs)
│   ├── wiki_disambig.jsonl          (37 MB — 63,113 docs)
│   ├── tokenizador_bpe_32k.model    (SentencePiece BPE 32k)
│   ├── ckpt_ark_ep1_rot*.bin        (rotating checkpoints, ~2.2 GB each)
│   └── [epoch 2-3+ corpus]
└── ark01/                           # Source code
    ├── src/
    │   ├── main.rs                  # Entry point and argument parser
    │   ├── config.rs                # Hyperparameters and validation
    │   ├── training.rs              # Training loop, AMP, checkpointing
    │   ├── optimizer.rs             # Zero-Copy AdamW, global vDSP clip
    │   ├── memory.rs                # ModelWeights, AlignedVec, f16↔f32
    │   ├── io.rs                    # JSONL CorpusStream, CheckpointV4, BPE
    │   └── ffi.rs                   # Safe Rust↔ObjC↔ASM bindings
    ├── objc/
    │   └── bridge.m                 # MPSGraph fwd/bwd AutoGrad, cross-entropy
    ├── asm/
    │   ├── kern.s                   # RMSNorm, softmax, SwiGLU, embed, dequant
    │   ├── opti.s                   # AdamW FP16/FP32, L2 grad clip (NEON)
    │   └── aten.metal               # Metal attention kernels
    ├── build.rs                     # Compiles bridge.m + kern.s + opti.s
    └── Cargo.toml
```

---

## Build and run

**Requirements:** macOS with Apple Silicon (M1/M2/M3/M4/M5), Rust toolchain, Xcode Command Line Tools.

```bash
git clone https://github.com/IAsesoria/ARK.git
cd ARK/rust/ark050
cargo build --release

caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial14.jsonl,../entren/wiki_disambig.jsonl \
  --vocab=../entren/tokenizador_bpe_32k.model \
  --ckpt=../entren/ckpt_ark_ep1_rot0.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=512 --batch=1 --lr=1e-4 --warmup=2000 --clip=0.5 \
  --epochs=1

tail -f ../entren/ark_ep1.log
```

> `caffeinate -i` prevents macOS from suspending the CPU or GPU during long training sessions.

> **Note on ANE warnings:** On startup, MPSGraph attempts to dispatch some operations 
> to the Apple Neural Engine and prints `Incompatible element type for ANE` warnings. 
> This is normal — ARK does not use ANE, the warnings are Apple's framework being 
> verbose about its own fallback to GPU. Training runs correctly on GPU throughout.

---

## Hardware testing — open call

ARK is developed and tested on M1 8GB. **We need reports from:**

- M1 Pro / Max / Ultra
- M2 / M2 Pro / Max / Ultra
- M3 / M3 Pro / Max
- M4 / M4 Pro / Max / Ultra
- M5 / M5 Pro / Max

If you have a Mac with Apple Silicon and want to contribute: compile ARK, run one training step, and open an Issue with the label `hardware-report` including:

- Chip and RAM
- Compile time
- Time per step (s/step)
- Memory usage

This directly helps optimize for hardware we don't have access to.

---

## Active research

### Think-Anywhere — on-demand reasoning during generation
*(Jiang et al., Peking University / Alibaba, March 2026 — arXiv:2603.29957)*

Demonstrates that LLMs concentrate reasoning in the thinking phase prior to the response, which is inefficient when problem complexity only reveals itself during generation. Think-Anywhere proposes inserting reasoning blocks at high-entropy positions during generation itself. Result: better performance with fewer total reasoning tokens.

**Application to EKO epoch 2:** build training examples where reasoning emerges at high-uncertainty positions within the response, not only in the initial thinking. *(Experimental, subject to kernel architecture variations.)*

### EML — one operator for all elementary functions
*(Odrzywolek, Jagiellonian University, April 2026 — arXiv:2603.21852)*

`eml(x, y) = exp(x) − ln(y)` together with the constant 1 generates the complete standard elementary function basis. Analogous to the NAND gate in digital electronics. Key identities verified in code.

**Application to ARK:** post-training symbolic regression with EML trees to discover which exact elementary function emerged in each layer of EKO. Interpretability analysis aligned with the NOUS Project's undeclared emergence principle.

---

## Relationship to the NOUS Project

ARK is the training engine for EKO, the verbalization and generation layer of the **NOUS Project** — a broad, dynamic emergent graph system where concepts organize themselves in a hypergraph by statistical co-activation, and relationships emerge from the statistical physics of the graph without explicit rules declared by the programmer.

NOUS will have no hardcoded semantic rules — meaning emerges from topological relationships, from the physics of the hypergraph. Structure is the next step to build in the project.

**Long-term goal:** NOUS provides structured knowledge, ARK provides the capacity to express it. Understanding and generation with completely distinct but complementary roots.

---

## Roadmap

- [x] Zero-Copy training engine in Rust/ObjC/ASM
- [x] BPE 32k tokenizer trained on Spanish
- [x] Spanish Wikipedia corpus (~617M real tokens)
- [x] Rotating v4 checkpoints with complete optimizer state
- [x] Stable dynamic AMP (scale 256 → 8192, zero skips)
- [x] AArch64 assembly kernels v0.62 (4 backward bugs fixed)
- [x] Native GPU AutoGrad via MPSGraph (bridge v1.3)
- [ ] Complete epoch 1 (~1,206,463 steps, ~14 days total from start)
- [ ] Inference mode (token-by-token generation from checkpoint)
- [ ] Epoch 2 — mathematical reasoning and abduction
- [ ] Epochs 3+ — instruction and dialogue
- [ ] Evaluation on Spanish benchmarks (HellaSwag-ES, XCOPA-ES, XQuAD)

---

## Contact and Collaboration

The code is free (MIT), the time is not. If you need adaptation, integration, technical consulting, or co-development:

📧 [benjaminalonsocarmona@gmail.com](mailto:benjaminalonsocarmona@gmail.com)

For companies that invoice using this code, formal support with contract and invoice is available.

<!--
## Sponsorship

If this work seems valuable to you, you can support it directly via **GitHub Sponsors**.
No tiers, no commitments — any amount helps keep the hardware running and the research moving forward.

→ [Support on GitHub Sponsors](https://github.com/sponsors/IAsesoriaInformatica)
-->

---

## About

Developed by **Benjamín Alonso Carmona Vega**.

Villarrica, Chile · 2026

*Developed with assistance from Claude Sonnet (Anthropic) and Gemini Pro (Google) 
for documentation, debugging, and architectural review.*

---

*Step 1 recorded loss=10.47. The theoretical value for a uniform distribution over 32,000 tokens is log(32,000) ≈ 10.37. The difference is Xavier initialization. That's all — the model knows nothing yet. What comes next is the interesting part.*
