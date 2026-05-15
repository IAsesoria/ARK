# CHANGELOG — ARK Training Engine

All significant changes, corrections, and decisions made during the project are documented here in chronological order. This is an honest record — including mistakes found and corrected.

---

## [2026-05-13 / 2026-05-15] — Corpus correction, vocabulary expansion, context increase, training resumption

**Global step at start of event:** ~300,000 (epoch 1)
**Global step at close of event:** ~312,300 (epoch 1, ongoing)

### Context

This entry covers three consecutive days of corrective work triggered by the detection of `<unk>` tokens in inference output at approximately step 300,000. The sequence of changes was not planned as a single operation — it unfolded progressively as each fix revealed the next issue. The chronology is documented as accurately as possible.

---

### 2026-05-13 — Root cause identified: corpus and vocabulary

#### What happened

Inference tests on the Ryzen machine using the active checkpoint produced `<unk>` tokens in generated output. Investigation revealed the issue was in the training corpus, not the model itself.

`wiki_esencial14.jsonl` had not been fully cleaned of non-Latin characters (Cyrillic, CJK, Arabic scripts). These characters passed through the filtering pipeline undetected. Since the BPE vocabulary (`tokenizador_bpe_32k.model`) was trained on clean Latin-script Spanish, it had no coverage for these characters — producing `<unk>` at inference time and introducing noise and dispersion into the learned embeddings during training.

This was a corpus preparation error that should have been caught earlier. It was not.

#### Action 1 — Corpus rebuilt: wiki_esencial19.jsonl

The Wikipedia corpus was rebuilt from scratch with strict non-Latin character filtering applied at the character level (`limpiar_wiki_v3.py`). The `title` field serialization order was also corrected (previously `{"text":..., "title":...}`, now consistently `{"title":..., "text":...}`).

| Metric                   | Value                                   |
|--------                  |-------                                  |
| Articles                 | 340,275                                 |
| Size on disk             | 2.1 GB                                  |
| Estimated tokens         | ~518M                                   |
| Avg tokens per doc       | ~1,524 (sampled dynamically at startup) |
| JSON errors              | 0                                       |
| Characters outside vocab | only U+0020 (space — expected)          |

`wiki_esencial14.jsonl` was deleted to recover disk space.

#### Action 2 — Vocabulary expanded: tokenizador_bpe_32k_v2.model

The BPE vocabulary was expanded from 32,000 to 32,063 tokens to cover high-frequency characters that were either causing `<unk>` or absent from the base vocabulary. Expansion was done by appending `USER_DEFINED` tokens to the SentencePiece model — without retraining BPE merges.

|        | Before                      | After                          |
|-       |--------                     |-------                         |
| File   | `tokenizador_bpe_32k.model` | `tokenizador_bpe_32k_v2.model` |
| Tokens | 32,000                      | 32,063                         |
| Added  | —                           | +63 characters (see below)     |

**The 63 characters added (by category):**

| Category                      | Characters                                        | Count  |
|----------                     |-----------                                        |------- |
| Lowercase Greek               | α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω | 25     |
| Uppercase Greek               | Γ Δ Θ Λ Σ Φ Χ Ψ Ω Π                               | 10     |
| Superscripts                  | ² ³ ¹ ⁴ ⁰                                         | 5      |
| Subscripts                    | ₀ ₁ ₂                                             | 3      |
| Math operators                | ° × √ ≈ ∫ → ± ·                                   | 8      |
| Fraction / currency / ordinal | ½ € º ª                                           | 4      |
| Missing ASCII                 | & # \ ~ ^ @ ` ÷                                   | 8      |
| **Total**                     |                                                   | **63** |

**Known omission — `ü` / `Ü`:**

The Spanish diaeresis characters `ü` (lowercase) and `Ü` (uppercase) were not included in this expansion. This was an oversight. The cleaning script `limpiar_wiki_v3.py` contained the transliteration `'ü': 'u', 'Ü': 'u'`, which silently converted all occurrences before the vocabulary audit was performed. As a result, the corpus contains "pinguino" instead of "pingüino", "verguenza" instead of "vergüenza", and approximately 250 affected words of low frequency in encyclopedic text.

The impact on Epoch 1 is considered marginal. Correction is scheduled before Epoch 2, which will contain reasoning corpora with correctly spelled Spanish. The full correction process is documented in:

> `tokenizador_bpe/gestion_vocabulario_bpe.md`

The full technical process for this expansion — including the Python script used,
verification steps, and the general procedure for adding characters to future
versions — is documented in:

> `tokenizador_bpe/vocabulary_bpe_management.md`

#### Action 3 — Checkpoint expanded to match new vocabulary

The embedding tensors were expanded from `[32000, 768]` to `[32063, 768]` to match the new vocabulary size. The 3 affected tensors (`embed_w` FP16, `embed_m` FP32, `embed_v` FP32) were expanded using the same neighbor-averaging initialization strategy documented in `gestion_vocabulario_bpe.md`. All 810 layer tensors were preserved unchanged.

#### Action 4 — Context increased: seq 128/512 → 1024

On the same day, the training context length was increased in two stages:

- First resumption: `seq=512 batch=1` (~4,000 steps, from global step ~300,500)
- Second resumption: `seq=1024 batch=1` (from global step ~304,000 onward)

Motivation: `seq=128` and `seq=512` were speed compromises that had reached their learning plateau. Longer context allows the model to learn longer-range dependencies in Spanish encyclopedic text. `batch=1` is the only configuration that fits `seq=1024` across 30 transformer layers within the 8GB unified memory of the MacBook Air M1 without swap.

The corpus step count at `seq=1024` is ~506,424 steps per epoch (340,275 docs × ~1,524 tokens/doc ÷ 1,024 tokens/step).

#### Training resumption — confirmed clean

```
[checkpoint v4] step=304000 | adam=304000 | layers=30
[ep 1  paso    1  g  304001]  loss=3.7324  ppl=41.8  scale=256  skips=0
```

Loss of 3.73 on the first resumed step at seq=1024 confirms checkpoint integrity and vocabulary compatibility.

**Active command from this point:**

```bash
nohup caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial19.jsonl \
  --vocab=../entren/tokenizador_bpe_32k_v2.model \
  --ckpt=../entren/ckpt_ark_ep1_rot2.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=1024 --batch=1 --lr=5e-5 --clip=0.5 \
  --epochs=1 >> ../entren/ark_ep1_seq1024.log 2>&1 &
```

---

### 2026-05-13 to 2026-05-15 — AMP stabilization at seq=1024

The AMP loss scaler progressed through its phases normally over the first ~8,300 local steps at seq=1024 (global steps ~304,001–312,300):

| Local steps | Global steps    | AMP scale     |
|-------------|-------------    |-----------    |
| 1–1,999     | 304,001–305,999 | 256           |
| 2,000–3,999 | 306,000–307,999 | 512           |
| 4,000–5,999 | 308,000–309,999 | 1,024         |
| 6,000–7,999 | 310,000–311,999 | 2,048         |
| 8,000+      | 312,000+        | 4,096 → 8,192 |

No NaN/Inf events, no skipped steps. AMP is progressing toward the stable `scale=8192` phase.

---

### What this event means for training history

The step counter (`g`) is cumulative and continuous across all changes. The corpus stream restarted from the beginning with `wiki_esencial19.jsonl` at step ~300,500. The ~300,000 steps trained on `wiki_esencial14.jsonl` are not wasted — the model learned real Spanish language structure — but embedding noise from `<unk>` characters will be progressively overwritten by the clean corpus. As a result, completing Epoch 1 requires reaching approximately global step ~806,000 (300,500 restart + 506,424 steps to cover the full corpus once at seq=1024). The previous steps contributed real language learning but do not count toward the current epoch's corpus coverage.

---

### Performance baseline at seq=1024 resumption (global step ~304,000)

| Window         | Avg loss | Avg PPL |
|--------        |-------   |---------|
| Last 10k steps | 4.032    | ~56     |
| Last 50k steps | 3.877    | ~48     |

These serve as the baseline to measure improvement over the next 50k–100k steps at seq=1024.

---

## [2026-05-13] — Corpus correction, vocabulary expansion, context increase

**Step at discovery:** ~300,000 (epoch 1, phase seq=128/batch=2)

### What happened

At approximately step 300,000, inference tests were run on the Ryzen machine using the active checkpoint. The output contained `<unk>` tokens — the model was producing unknown token markers during generation.

### Root cause

Investigation revealed the issue was not in the model itself, but in the training corpus. `wiki_esencial14.jsonl` had not been fully cleaned of non-Latin characters (Cyrillic, CJK, Arabic scripts). These characters passed through the filtering pipeline undetected. Since the BPE vocabulary (`tokenizador_bpe_32k.model`) was trained on clean Latin-script Spanish, it had no coverage for these characters — producing `<unk>` at inference time and introducing noise and dispersion into the learned embeddings during training.

This was a corpus preparation error that should have been caught earlier. It was not.

### Actions taken

**1. Corpus regenerated — wiki_esencial19.jsonl**

The Wikipedia corpus was rebuilt with strict non-Latin character filtering applied at the character level. The `title` field serialization order was also corrected (previously `{"text":..., "title":...}`, now consistently `{"title":..., "text":...}`).

| Metric                    | Value                                      |
|---                        |---                                         |
| Articles                  | 340,275                                    |
| Size                      | 2.1 GB                                     |
| JSON errors               | 0                                          |
| Articles missing `title`  | 0                                          |
| Articles missing `text`   | 0                                          |
| Characters outside vocab  | only U+0020 (space — expected and correct) |

`wiki_esencial14.jsonl` was deleted to recover disk space on the M1 8GB machine.

**2. Vocabulary expanded — tokenizador_bpe_32k_v2.model**

The BPE vocabulary was expanded to cover the characters that were causing `<unk>`. The embedding tensors and checkpoint were expanded accordingly.

|        | Before                      | After                          |
|---     |---                          |---                             |
| Tokens | 32,000                      | 32,063                         |
| File   | `tokenizador_bpe_32k.model` | `tokenizador_bpe_32k_v2.model` |

**3. Context length increased — seq 128 → 512 → 1024**

Training was resumed with `seq=512 batch=1` for approximately 4,000 steps, then increased to `seq=1024 batch=1`. This change was made at the same time as the corpus correction. Motivation: longer context allows the model to learn longer-range dependencies; the previous seq=128 was a speed compromise that had reached its plateau.

**4. Training resumed from step 300,500**

The checkpoint `ckpt_ark_ep1_rot1_expanded.bin` was loaded with full Adam optimizer state (271 tensors). Resumption confirmed clean:

```
[checkpoint v4] step=300500 | adam=300500 | layers=30
[ep 1  paso  1  g 300501]  loss=3.1333  ppl=22.9  scale=256  skips=0
```

Loss of 3.13 on the first resumed step confirms checkpoint and expanded vocabulary are compatible.

**Active command from this point:**

```bash
nohup caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial19.jsonl \
  --vocab=../entren/tokenizador_bpe_32k_v2.model \
  --ckpt=../entren/ckpt_ark_ep1_rot1_expanded.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=512 --batch=1 --lr=5e-5 --clip=0.5 \
  --epochs=1 >> ../entren/ark_ep1_seq512.log 2>&1 &
```

### What this means for the training history

The step counter (`g`) continues from 300,501 and is cumulative. However, the corpus stream restarted from the beginning with the new file. The ~300,000 steps trained on `wiki_esencial14.jsonl` are not wasted — the model learned real Spanish language structure from them — but the embedding noise introduced by `<unk>` characters will need to be overwritten by the clean corpus. Estimated steps needed to fully cover the corpus once at seq=512: ~1,012,849. Context was subsequently increased to seq=1024, reducing that to ~506,424 steps to complete Epoch 1 — see entry [2026-05-13 / 2026-05-15] above. The global step counter will read approximately ~806,000 when Epoch 1 completes (300,500 restart point + 506,424 corpus steps).

### Performance baseline at resumption (loss averages over last N steps)

| Window          | Avg loss | Avg ppl |
|---              |---       |---      |
| Last 10k steps  | 3.871    | 55.00   |
| Last 50k steps  | 3.854    | 54.71   |
| Last 100k steps | 3.853    | 54.60   |

These serve as baseline to measure whether the corpus correction and context increase produce measurable improvement over the next 10k–50k steps.

### Pending validation

Inference on the Ryzen machine with the first stable checkpoint after step ~310,000 will confirm whether `<unk>` tokens have been eliminated from generation output.

---

## Pending — Before Epoch 2

All expansion procedures (tokenizer and checkpoint) are covered step by step in
`tokenizador_bpe/vocabulary_bpe_management.md` and `tokenizador_bpe/expand_checkpoint.js`.

The following actions are required before starting Epoch 2 training:

| Action               | Description                                          | Reference                                      |
|--------              |-------------                                         |-----------                                     |
| Tokenizer v3         | Add `ü` / `Ü` → 32,065 tokens                        | `tokenizador_bpe/vocabulary_bpe_management.md` |
| Checkpoint expansion | Expand embedding `[32063,768]` → `[32065,768]`       | `expand_checkpoint.js`                         |
| config.rs update     | `VOCAB_SIZE: 32063` → `32065`                        | `src/config.rs`                                |
| Corpus audit         | Verify Epoch 2 corpora contain correctly spelled `ü` | `grep -l 'ü' *.jsonl`                          |
| README update        | Update tokenizer section to reflect v3               | `README_ES.md` / `README.md`                   |

---

*ARK is developed by Benjamín Alonso Carmona Vega / IAsesoria Informática, Villarrica, Chile.*
*Development assisted by Claude Sonnet (Anthropic) and Gemini Pro (Google).*
