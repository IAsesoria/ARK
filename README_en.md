# ARK
ARK es un motor para entrenar modelos de lenguaje, escrito en lenguaje; Ensamblador, Rust y Metal GPU, para MacBook con Chip Apple Silicon. Sin PyTorch ni TensorFlow. // ARK is a language model training engine, written in Assembly, Rust, and Metal GPU, for MacBooks with Apple Silicon chips. It does not include PyTorch or TensorFlow.

# ARK — LLM Training Engine from Scratch
### EKO · NOUS Project · IAsesoria Informática · Villarrica, Chile · 2026

> **Training right now.** Epoch 1, step ~12,300 of ~1,206,463. Loss: 10.47 → 3.94. No PyTorch. No TensorFlow. No cloud GPU.

---

## What this is

ARK is a large language model training engine written entirely from scratch in **Rust, Objective-C, and AArch64 NEON assembly**. It does not depend on PyTorch, TensorFlow, or any deep learning framework. Every layer of the compute stack — GPU forward pass, GPU backward via MPSGraph AutoGrad, assembly optimizer, mathematical kernels — is hand-written and tuned specifically for Apple Silicon M1.

The first model ARK trains is called **EKO**, part of the **NOUS Project** by IAsesoria Informática. EKO is a 237-million-parameter transformer trained exclusively in Spanish, on a curated corpus of encyclopedic, mathematical, and conversational text built specifically for this project.

The goal is not to replicate what already exists. It is to prove that serious AI can be built from consumer hardware, in Spanish, from Chile, with no dependency on third-party infrastructure.

---

## Why this matters

Most language models are trained on NVIDIA GPU clusters costing tens or hundreds of thousands of dollars. ARK demonstrates three things that are rarely demonstrated together:

**1. Real efficiency on limited hardware.**
A MacBook Air M1 with 8GB of RAM is consumer hardware. ARK uses it to its fullest: Zero-Copy architecture on unified memory, GPU forward pass via MPSGraph with AOT-compiled graphs, native GPU AutoGrad backward via `gradientForPrimaryTensor:withTensors:`, AdamW optimizer in SIMD assembly. No unnecessary data copies between CPU and GPU. No framework overhead.

**2. Technological sovereignty.**
The corpus is in Spanish. The project is Chilean. The model is fully owned. No external APIs, no third-party base models, no cloud infrastructure. Training and inference are 100% local.

**3. Full reproducibility.**
ARK compiles and runs on any Apple Silicon Mac. The only requirements are Rust and Xcode Command Line Tools. The code is transparent — what you read is exactly what runs.

---

## System architecture

The pipeline divides work across the three compute units of the M1 chip:

### GPU — MPSGraph (full forward + AutoGrad backward)

The complete forward pass runs on GPU via Metal Performance Shaders Graph (MPSGraph), with AOT-compiled graphs before the first training step:

- Embedding lookup (FP16)
- 30 transformer layers: RMSNorm → multi-head SDPA attention → SwiGLU FFN → RMSNorm
- Rotary Position Embedding (RoPE) with CPU-precomputed sin/cos tables
- Native causal masking fused into SDPA
- LM head with weight tying to embedding (Conv1×1 workaround for Metal's W≤16384 limit)

The backward pass for all 30 layers and the LM head runs on GPU via MPSGraph native AutoGrad (`gradientForPrimaryTensor:withTensors:`). Each layer graph defines a `loss_proxy` as the reduction sum of `t_out × d_out`, and symbolic differentiation generates gradients for all 10 tensors per layer (dx, dwq, dwk, dwv, dwo, dw1, dw2, dw3, dg1, dg2) automatically. No manual chain rule implementation.

Cross-entropy (log-sum-exp, numerically stable) and embedding scatter-add run on CPU via Accelerate.

### CPU — Accelerate / AMX

- Numerically stable cross-entropy with `vDSP_maxv` and log-sum-exp
- Gradient scatter-accumulation into embedding rows (sequential, handles repeated tokens)
- Global L2 grad norm with `vDSP_svesq` before the Adam step

### AArch64 NEON assembly (optimizer and math kernels)

The AdamW optimizer and mathematical kernels are written directly in AArch64 assembly:

- **`asm/opti.s`** — Full AdamW: vectorized Adam formula, 4 floats per SIMD cycle, bias correction, weight decay, global L2 grad clip.
- **`asm/kern.s`** — RMSNorm FP32/FP16 (forward + backward v0.62 with 4 corrected bugs), softmax FP32/FP16 (3-pass, underflow-safe), SwiGLU forward FP16, embedding gather FP16→FP32, dequant/quant FP16↔FP32 (8-wide), dot product FP16 with FP32 accumulation.
- **`asm/aten.metal`** — Metal attention kernels: `attention_scores_f16`, `attention_softmax_f32`, `attention_weighted_sum_f16`, fused `attention_full_f16_causal`.

### Zero-Copy unified memory

All model weights live in `MTLBuffer` with `storageModeShared`. CPU and GPU access the same physical memory block — no intermediate copies. The assembly optimizer receives a direct pointer to VRAM, reads FP16 weights, updates them in-place, and writes them back to the same address the GPU will read in the next forward pass.

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

### Current training configuration

```
Epoch 1 (linguistic base):
  --corpus=wiki_esencial14.jsonl,wiki_disambig.jsonl
  --layers=30 --heads=12 --d-model=768 --hidden=2048
  --seq=512 --batch=1 --lr=1e-4 --warmup=2000 --clip=0.5
  --epochs=1

AdamW: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
AMP:   loss_scale_init=256, max=8192, step_up_every=2000
Checkpoint: 3-slot rotating, every 500 steps
```

---

## Automatic mixed precision (AMP)

Weights are stored in FP16 in VRAM. Gradients and Adam moments (m, v) are maintained in FP32. The dynamic loss scaler starts at 256 and can rise to 8192 in ×2 steps every 2000 clean steps. If NaN/Inf appears, the step is discarded, the scale halves, and training continues from the intact checkpoint without optimizer state corruption.

**Current status:** scale reached 8192 (maximum) with zero gradient skips across the entire session. This confirms numerically stable gradients since the v0.62 assembly kernel corrections.

---

## Training status — 2 May 2026

| Metric | Value |
|---|---|
| Current local step | ~12,300 / ~1,206,463 |
| Global step (all sessions) | ~16,800 |
| Initial loss (step 1) | 10.47 — matches log(32,000), confirms correct Xavier init |
| Current loss | ~3.94–5.3 (oscillating normally) |
| Best loss seen | ~3.94 |
| AMP loss scale | 8192 (maximum reached, stable) |
| Gradient skips | 0 |
| Continuous runtime | ~30+ hours |
| Checkpoints saved | Every 500 steps, 3 rotating slots |

Recent log extract:
```
[ep 1  paso  12000/~1206463  g   16500]  loss=3.9473  ppl=51.8   scale=8192  skips=0
[ep 1  paso  12100/~1206463  g   16600]  loss=4.6759  ppl=107.3  scale=8192  skips=0
[ep 1  paso  12200/~1206463  g   16700]  loss=4.2382  ppl=69.3   scale=8192  skips=0
[ep 1  paso  12300/~1206463  g   16800]  loss=4.9375  ppl=139.4  scale=8192  skips=0
```

---

## Corpus and training curriculum

Epoch 1 corpus: **~617M real tokens** (calculated by sampling 1,000 documents at startup — no hardcoded estimates):

| Corpus | Content |
|---|---|
| `wiki_esencial14.jsonl` | Spanish Wikipedia — 341,147 filtered documents (2.1 GB) |
| `wiki_disambig.jsonl` | Wikipedia disambiguation pages — 63,113 documents |

Order matters: encyclopedic structured text first. The model learns the language distribution before seeing reasoning or instruction data.

**Planned epochs:**

- **Epoch 1** — Linguistic base. Wikipedia + disambiguation. ~1,206,463 steps at seq=512.
- **Epoch 2** — Reasoning and logic. GSM8K-ES, GSM-Hard, MCOT-Math, Aya-Reasoning, abduction corpus, multilingual thinking. seq=1024, lr=5e-5.
- **Epoch 3+** — Dialogue and instruction. Alpaca-ES, Somos Alpaca, Orca-ES, natural conversation, OpenSubtitles, Tatoeba, StackOverflow, plain language, NOUS identity corpus.

---

## Checkpoint format

Format v4 (`ARK4` magic bytes). Stores weights in native FP16 + Adam moments m and v in FP32. For 237M parameters: **2,369.6 MB per slot**.

3-slot rotating system. Always at least two valid copies available simultaneously. On resume, restores full weights and optimizer state (271 tensors) so training continues with accumulated Adam momentum intact. Backward compatible with v2 (FP32 weights) and v3 (FP32 weights + moments).

---

## Project structure

```
ark/
└── rust/
    ├── entren/                          # Corpus and artifacts
    │   ├── wiki_esencial14.jsonl        (2.1 GB — 341,147 docs)
    │   ├── wiki_disambig.jsonl          (37 MB — 63,113 docs)
    │   ├── tokenizador_bpe_32k.model    (SentencePiece BPE 32k)
    │   ├── ckpt_ark_ep1_rot*.bin        (rotating checkpoints, ~2.2 GB each)
    │   └── [epoch 2-3+ corpus]          (reasoning, instruction, dialogue)
    └── ark050/                          # Source code
        ├── src/
        │   ├── main.rs                  # Entry point and argument parser
        │   ├── config.rs                # Hyperparameters and validation
        │   ├── training.rs              # Training loop, AMP, checkpointing
        │   ├── optimizer.rs             # AdamW Zero-Copy, global vDSP clip
        │   ├── memory.rs                # ModelWeights, AlignedVec, f16↔f32
        │   ├── io.rs                    # CorpusStream JSONL, CheckpointV4, BPE
        │   └── ffi.rs                   # Safe Rust↔ObjC↔ASM bindings
        ├── objc/
        │   └── bridge.m                 # MPSGraph fwd/bwd AutoGrad, cross-entropy
        ├── asm/
        │   ├── kern.s                   # RMSNorm, softmax, SwiGLU, embed, dequant
        │   ├── opti.s                   # AdamW FP16/FP32, L2 grad clip (NEON)
        │   └── aten.metal               # Metal attention kernels
        ├── build.rs                     # Compiles bridge.m + kern.s + opti.s
        └── cargo.toml
```

---

## Build and run

**Requirements:** macOS with Apple Silicon (M1/M2/M3), Rust toolchain, Xcode Command Line Tools.

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

`caffeinate -i` prevents macOS from suspending CPU or GPU during long training runs.

---

## Hardware testing — open call

ARK is developed and tested on M1 8GB. **We need reports from:**

- M1 Pro / Max / Ultra
- M2 / M2 Pro / Max / Ultra
- M3 / M3 Pro / Max

If you have an Apple Silicon Mac and want to contribute: compile ARK, run one training step, and open an Issue with label `hardware-report` including your chip, RAM, compile time, step time (s/step), and memory usage. This directly helps optimize for hardware we don't have access to.

---

## Active research

### Think-Anywhere — on-demand reasoning during generation
*(Jiang et al., Peking University / Alibaba, March 2026 — arXiv:2603.29957)*

Demonstrates that LLMs concentrate reasoning in upfront thinking, which is inefficient when problem complexity only reveals itself during generation. Think-Anywhere inserts `<thinkanywhere>` reasoning blocks at high-entropy positions mid-generation. Result: better performance with fewer total reasoning tokens.

Application to EKO epoch 2: building training examples where reasoning emerges at uncertainty positions within the response rather than concentrating only in upfront thinking.

### EML — one operator for all elementary functions
*(Odrzywolek, Jagiellonian University, April 2026 — arXiv:2603.21852)*

`eml(x, y) = exp(x) − ln(y)` together with the constant 1 generates the full standard basis of elementary functions. Analogous to the NAND gate in digital electronics. Key identities verified in code.

Application to ARK: post-training symbolic regression using EML trees to discover what exact elementary function emerged in each layer of EKO. Interpretability analysis aligned with NOUS's emergence-without-declaration principle.

---

## Relation to the NOUS Project

ARK is the training engine for EKO, the verbalization and generation layer of the NOUS Project — a broader emergent cognition system where concepts organize in a cognitive hypergraph by co-activation, and relationships emerge from the statistical physics of the graph with no programmer-declared rules.

The philosophical coherence between NOUS and ARK is complete: neither defines linguistic behaviors explicitly. NOUS has no hardcoded semantic rules — relationships emerge from hypergraph physics. ARK has no language heuristics — the model learns token distributions purely from data. Meaning emerges from structure, in both systems.

Long-term goal: NOUS provides structured knowledge, ARK provides the capacity to express it. Comprehension and generation with entirely distinct but complementary roots, both without declarative programmer intervention.

---

## Roadmap

- [x] Zero-Copy training engine in Rust/ObjC/ASM
- [x] BPE 32k tokenizer trained on Spanish
- [x] Spanish Wikipedia corpus (~617M real tokens)
- [x] Rotating v4 checkpoints with full optimizer state
- [x] Stable dynamic AMP (scale 256 → 8192, zero skips)
- [x] v0.62 assembly kernels (4 backward bugs corrected)
- [x] Native GPU AutoGrad via MPSGraph (bridge v1.3)
- [ ] Complete epoch 1 (~1,206,463 steps, est. ~14 days total)
- [ ] Inference mode (token-by-token generation from checkpoint)
- [ ] Epoch 2 — mathematical reasoning and abduction
- [ ] Epochs 3+ — instruction and dialogue
- [ ] Quantitative evaluation on Spanish benchmarks (HellaSwag-ES, XCOPA-ES, XQuAD)
- [ ] Scale to 1B parameters

---

## Sponsorship

ARK and the NOUS Project are independent research privately funded from Villarrica, Chile. Training runs on owned hardware with no cloud cost.

If you are an institution, company, or individual interested in efficient AI on consumer hardware, sovereign Spanish-language models, or emergent cognitive systems without big-provider dependency — support is welcome via **GitHub Sponsors**.

### Sponsor tiers

| Tier | Amount | What you get |
|---|---|---|
| **Follower** | $2/mo | Monthly training progress update — loss curve, steps completed, milestones reached. Name in the README sponsors list. |
| **Supporter** | $8/mo | Everything above + detailed monthly report: loss analysis, gradient stability, corpus coverage, AMP behavior. |
| **Contributor** | $25/mo | Everything above + access to intermediate checkpoints as they become evaluable. Mention in any technical publication from this project. |
| **Collaborator** | $75/mo | Everything above + technical assistance if you're building something on top of ARK or EKO. Priority response to issues. Code adaptation guidance for specific use cases. |
| **Institutional** | $250/mo | Everything above + co-authorship consideration in research outputs. Custom adaptation of ARK or EKO for your organization's use case. Access to full technical documentation and architecture decisions. |

All support goes directly to hardware (storage for corpus and checkpoints, higher-capacity Apple Silicon for faster training) and research time.

---

## About

Developed by **Benjamín Alonso Carmona Vega**, founder of IAsesoria Informática, with collaboration from **Sonia**.

Villarrica, Chile · 2026

---

*Step 1 logged loss=10.47. The theoretical value for uniform distribution over 32,000 tokens is log(32,000)≈10.37. The difference is Xavier initialization. That is all — the model knows nothing yet. What comes next is the interesting part.*
