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

## [2026-05-17] — Ryzen inferencer: vocab v1→v2 migration, first <unk>-free inference

**Reference global step:** ~321,000 (epoch 1, ongoing)

### Context

During inference tests on May 13 (step ~297,000) using the original v1 vocabulary
(`vocab_sp.json`, 32,000 tokens), output contained frequent `<unk>` tokens —
direct evidence of the coverage problem documented in the previous entry.
Those tests are preserved in:

> `inferencias/EKO_paso297k_130526.txt`

On May 17, the Ryzen inferencer was migrated to the expanded v2 vocabulary and
the first clean inference results were recorded:

> `inferencias/EKO_paso321k_170526.txt`

---

### Problem detected

`eko_infer` on Windows loads vocabulary from plain JSON (`vocab_sp.json` +
`vocab_scores.json`), not directly from the SentencePiece `.model` file.
When the tokenizer was expanded from 32,000 to 32,063 tokens, the JSON files
on the Ryzen machine became stale. The active checkpoint (step 321,000) was
trained with v2 but the inferencer was still loading v1 — producing a silent
vocab_size mismatch.

---

### Actions

**1. v2 tokenizer copied to Ryzen from Mac**

```bash
scp benjamin@[local-ip]:/Users/benjamin/Documents/ark/rust/entren/tokenizador_bpe_32k_v2.model D:\proyecto-nwin\llm\eko\
```

**2. v2 JSON files generated on Ryzen with Python**

Without interrupting training on the Mac. sentencepiece was installed on Windows
and the JSON files were exported directly from the `.model`:

```powershell
python -m pip install sentencepiece
python exportar_vocab_v2.py tokenizador_bpe_32k_v2.model
```

Confirmed output:

**3. Active inference command**

```powershell
cargo run --release --bin eko_infer -- `
  --ckpt ..\ckpt_ark_ep1_rot017mayo.bin `
  --vocab ..\vocab_sp_v2.json `
  --scores ..\vocab_scores_v2.json `
  --vocab-size 32063 `
  --prompt "El agua es"
```

**4. Files added to repository**

Folder `tokenizador_bpe/` — commit `ece67bb`:

| File                   | Description                                              |
|------------------------|----------------------------------------------------------|
| `vocab_sp_v2.json`     | v2 vocabulary in JSON format (32,063 tokens)             |
| `vocab_scores_v2.json` | v2 Viterbi scores in JSON format (32,063 tokens)         |
| `exportar_vocab_v2.py` | Script to regenerate JSON files from any `.model` file   |

Folder `inferencias/` — commit `a877ff4`:

| File                      | Description                                        |
|---------------------------|----------------------------------------------------|
| `EKO_paso297k_130526.txt` | Inference log 13-may, step ~297,000, vocab v1      |
| `EKO_paso321k_170526.txt` | Inference log 17-may, step ~321,000, vocab v2      |

---

### Results — direct comparison

| Metric                | 13-may step ~297,000 vocab v1 | 17-may step ~321,000 vocab v2 |
|-----------------------|-------------------------------|-------------------------------|
| `<unk>` tokens        | Frequent                      | None                          |
| Inference speed       | ~4 tok/s                      | ~11 tok/s                     |
| Geographic coherence  | Low                           | Medium                        |
| Active corpus         | wiki_esencial14 (noisy)       | wiki_esencial19 (clean)       |

The ~3× speed improvement is due to weight transposition implemented in
`eko_infer`, not the vocabulary change.

**Observations on output quality at step 321,000:**

- Zero `<unk>` tokens across all output
- Grammatically coherent Spanish at the short-phrase level
- Dominant thematic attractor: etymology / toponymy / languages — consistent
  with Wikipedia letter-A content (~10% of corpus processed at seq=1024)
- Geographic prompts produce the most coherent output at this stage




## [2026-05-22] — Learning rate manual decay, sequence length stabilization, swap diagnostics

**Global step at event:** ~343,500 (epoch 1)

### What happened

A steady upward trend in average loss and perplexity (PPL) was observed over 5 consecutive days:
- Minimum point: Loss `3.491` | PPL `34.25` (approx. step 320,000)
- Peak point: Loss `3.841` | PPL `54.04` (approx. step 343,500)

While the model occasionally encountered local spikes due to dense, non-linguistic data blocks (tables and lists in Wikipedia), the overall daily average failed to return to baseline. This was diagnosed as **no-decay instability (slow divergence)**. A constant learning rate of `5e-5` was too aggressive for the fine-tuning phase of the 1024 context window, preventing the weights from settling into the local minimum.

An experimental test at `seq=2048` was attempted but quickly aborted. Diagnostics via `htop` showed active memory swap climbed to 1.87 GB, dropping CPU utilization to 18.1% (severe disk thrashing). This confirmed `seq=1024` as the absolute physical boundary for training on an 8GB M1 machine.

### Actions taken

**1. Process terminated safely**
The training process was stopped at global step 343,600. The checkpoint `ckpt_ark_ep1_rot0.bin` (saved at 07:48 AM, matching step 343,500) was verified as the most recent clean state.

**2. Manual learning rate decay applied (5e-5 → 2e-5)**
To stabilize the convergence and prevent weight oscillation, the learning rate was decreased by 60% from `5e-5` to `2e-5` (0.00002). The gradient clipping remained at `--clip=0.5` to protect the model from remaining gradient variance.

**3. Training resumed at seq=1024**
The process was restarted using the latest clean checkpoint with the updated parameters.

**Active command from this point:**

```bash
nohup caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial19.jsonl \
  --vocab=../entren/tokenizador_bpe_32k_v2.model \
  --ckpt=../entren/ckpt_ark_ep1_rot0.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=1024 --batch=1 --lr=2e-5 --clip=0.5 \
  --epochs=1 >> ../entren/ark_ep1_seq1024.log 2>&1 &
```
---

### 4. Training Dashboard UI Optimization

Along with the core engine adjustment, two legacy bugs were resolved in the training visualization interface:

- **Canvas Rendering Bug Fixed:** Modified `drawChart()` to read dimension values directly from `canvas.clientWidth/clientHeight` instead of parsing `canvas.style.width`. This resolved a layout bug where chart lines failed to render unless the browser window was zoomed out.
- **KPI Card Fault Tolerant Update:** Adjusted `updateDashboardCards()` to prevent Javascript execution crashes. Previously, removing the unused "Loss Mínimo" KPI card from the HTML caused a fatal null pointer exception when the update script tried to write into its nonexistent element, silently halting subsequent UI updates (such as the dynamic AMP Scale badge). The script and HTML layout have been cleaned and decoupled.

*ARK is developed by Benjamín Alonso Carmona Vega / IAsesoria Informática, Villarrica, Chile.*
*Development assisted by Claude Sonnet (Anthropic) and Gemini Pro (Google).*
