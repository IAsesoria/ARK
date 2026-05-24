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

---

# CHANGELOG — ARK Training Engine (continued)

---

## [2026-05-24] — Tokenizer expansion v2→v7, deep corpus cleanup, bridge.m fix, epoch 2 start

**Global step at event start:** 352,000 (epoch 1, stopped)
**Global step at event end:** 352,001 (epoch 2, started)

### Context

This entry documents a full day of corrective work prior to the start of epoch 2. Epoch 1 training had reached step 352,000 on corpus `wiki_esencial19.jsonl` and loss instability was diagnosed, along with possible corpus noise and incomplete vocabulary. The decision was made to stop training, thoroughly clean all corpora, expand the tokenizer to a definitive version, and fix a critical bug in the Metal engine before resuming.

---

### 1. Loss diagnosis — divergence and corpus change

#### What was observed

With `lr=2e-5` and AMP restarted from scale 256 on resume, loss showed an upward trend during AMP warmup phases. Daily average 48h: 3.949, 72h: 3.780. Real divergence was ruled out (no NaN/Inf, no skips, AMP scaling normally) and AMP scaling noise was identified as the main cause.

#### Corpus change decision

The decision was made to switch the training corpus from `wiki_esencial19.jsonl` (full Wikipedia) to reasoning and knowledge corpora of higher semantic density. This decision also allowed addressing the detected cleanup issues.

---

### 2. Wiki corpus cleanup — wiki_v21 → wiki_v22 → wiki_v23

#### 2a. Additional cleanup of wiki_esencial19 → wiki_v21

Residual patterns were detected in the Wikipedia corpus that had survived the previous cleaner (`limpiar_wiki_v3.py`). A two-step process was applied:

**Shuffle:** `wiki_esencial19_shuffle.jsonl` was generated to break the alphabetical order bias that concentrated difficult articles (tables, lists) in continuous blocks.

**Python cleanup (`limpiar_wiki_v20.py`):** The following patterns were corrected:
- `]]` orphaned MediaWiki link closing brackets
- `[[` residual opening brackets
- `== heading ==` and `===` section headings
- `::` disambiguation lists
- `\frac`, `\text`, `\acute` and other LaTeX math commands

**Rust cleanup with rayon (`limpiar_v21`, Cargo project):** The cleanup was rewritten using `rayon` and `serde_json` to process the 340,275 articles in parallel. The Rust auditor confirmed 0 problematic patterns.

| Metric | Value |
|---|---|
| Articles in | 340,275 |
| Articles out | 340,275 |
| Time | 43.3s |
| Residual patterns | 0 |
| Quality: has period | 100.0% |
| Quality: has verb | 99.7% |

**Generated file:** `wiki_v21.jsonl` (2,168 MB)

#### 2b. Filtering of non-encyclopedic articles → wiki_v22

`filtrar_wiki.py` was created to remove low-value articles using title patterns:

Removed categories:
- Sports events with year (Copa, Liga, Campeonato, Rally, Temporada + year)
- Local sports clubs and stadiums
- Anime/videogame characters (Pokémon, Dragon Ball, Naruto, etc.)
- Specific hardware (Samsung Galaxy, Nokia NNNN, etc.)
- Discographies, albums and specific tours

A protection list was implemented to preserve articles even if they match elimination patterns (international federations, articles with cultural context by country, etc.).

| Metric | Value |
|---|---|
| Articles in | 340,275 |
| Articles removed | 2,228 (0.7%) |
| Articles kept | 338,047 |

**Generated file:** `wiki_v22.jsonl`

#### 2c. Umlaut correction — wiki_v22 → wiki_v23

It was identified that the original cleaner `limpiar_wiki_v3.py` had silently transliterated `ü→u` before the v2 vocabulary audit, leaving ~22,000 articles with incorrect forms: `pinguino`, `verguenza`, `bilingue`, `antiguedad`, `linguista`, etc.

`corregir_dieresis.py` was created with 53 correction pairs, respecting uppercase, lowercase and capitalized forms.

| Metric | Value |
|---|---|
| Articles processed | 338,047 |
| Articles modified | 22,480 (6.7%) |
| Corrected examples | `pinguino→pingüino`, `verguenza→vergüenza`, `bilingue→bilingüe`, `antiguedad→antigüedad`, `linguista→lingüista`, `ciguena→cigüeña` |

Post-correction verification: `grep` of incorrect forms in a 5,000-line sample = 0. Correct forms with umlaut in same sample = 275.

**Definitive file:** `wiki_v23.jsonl` (renamed from `wiki_v22_dieresis.jsonl`)

---

### 3. Training corpus cleanup — entren2 → entren6

5 progressive cleanup passes were performed on the 34 training corpora (excluding wikis). Intermediate folders `entren2` through `entren5` were deleted on completion; the definitive result is in `entrenamiento/`.

#### Initial contamination audit

A Rust auditor with rayon (`otra_carpeta1/`) was built that processed all corpora and automatically generated a `verificar_ids.py` script to contrast each non-ASCII character against the tokenizer.

Relevant results from the initial audit:

| Corpus | Contaminated |
|---|---|
| `ruby_es_limpio_clean.jsonl` | 17.6% — web navigation menus with Japanese/Korean |
| `gsm8k_reasoning_es_final.jsonl` | 1.0% — Cyrillic from Google Translate |
| `stackoverflow_final.jsonl` | 0.1% |
| Rest | 0.0% |

`ruby_es_limpio_clean.jsonl` was **removed** from the training set due to irrecoverable structural contamination.

`stackoverflow_final.jsonl` was also **removed** by design decision: 100% of its documents contain `[CODIGO]...[/CODIGO]` blocks in technical English, introducing noise without linguistic value at this stage.

#### Cleanup passes

**entren2 — surgical cleanup of non-Latin scripts (`limpiar_quirurgico.py`):**
Characters from the following were removed: Cyrillic, Japanese hiragana/katakana, Chinese CJK, Arabic, Korean, Hebrew, Devanagari. Only on contaminated files; the rest were copied directly.

**entren3 — visual noise removal (`limpiar_ruido_final.py`):**
Removed: `█` (U+2588), `♪` (U+266A), `➞` (U+279E), box-drawing characters (╱▼▶△), ideographic punctuation (。、「」《》), invisible control characters (U+200B, U+0008, U+2060).

**entren4 — additional script removal (`limpiar_entren4.py`):**
Thai, Georgian, Armenian, Tibetan, Syriac, Khmer, Lao, Malayalam, Gurmukhi, Unicode private area, emojis, remaining invisible controls.

**entren5 — IPA and exotic Latin character removal (`limpiar_entren6.py`):**
International Phonetic Alphabet and exotic Latin letters with frequency ≤7 across all corpora.

**Final audit on definitive corpora:**

```
Total docs audited: 901,186
Total contaminated: 0
VERDICT: ✓ ALL CORPORA CLEAN — 0 contaminated
```

Corpora removed from epoch 2 training set:
- `ruby_es_limpio_clean.jsonl` — structural contamination
- `stackoverflow_final.jsonl` — design decision (technical code in English)

---

### 4. Tokenizer expansion v2 → v7

5 progressive expansions of the BPE tokenizer were performed, each preceded by an audit of the corpora against the previous version. The process used `sentencepiece_model_pb2` to add `USER_DEFINED` tokens without retraining the BPE merges.

> **Technical note:** Previous CHANGELOG entries documented the pending expansion of `ü/Ü` (tokenizer v3 = 32,065). This expansion was performed today as part of a broader process. The final result is tokenizer v7 = 32,308 tokens.

| Version | Tokens | Added | Main category |
|---|---|---|---|
| v2 | 32,063 | — | base (documented in previous CHANGELOG) |
| v3 | 32,104 | +41 | Full umlauts (ü Ü ö Ö ä Ä ë ï…), Romance vowels (è à â ê î ô…), typographic punctuation (— – " " ' '…), math operators (≥ ≤ ≠ ≡ ⁿ ⁺ ⁻ ∂ ₐ…) |
| v4 | 32,112 | +8 | Missing symbols: § ∪ ∩ ∞ ∑ ¥ © ® |
| v5 | 32,226 | +114 | Frequent Latin chars in corpora (ã ō ć ū ș š ă ł č ý ā ø ț ž ś ł ß å ń…), accented Greek (ύ ή ά ό ί έ ώ…), uppercase Greek (Α Κ Ε Μ Ν Β Τ Ι Υ Ρ), logic operators (∈ ∧ ∨ ∴ ∅ ∃ ∀ ∠ ∗ ∇ ∶ ∛ ∮ ∝), arrows (⇒ ⇔ ⇐ ← ↔ ↑ ↓ ↵), sets (⊆ ⊂ ⊕ ⊗ ⟨ ⟩), typography (¬ ¢ £ † ‡ ― ‐ ¨ ¶) |
| v6 | 32,289 | +63 | Additional Latin (ð Č Î Ş Ż ə ĭ Â ė ň ǔ ǎ İ Œ…), accented Greek (Ά Ί Ό Ξ Ζ Η), frequent symbols (△ □ ● ✔ ∆ ∉ ∣), European quotes („ ‚), currencies (₤ ₫ ₪) |
| v7 | 32,308 | +19 | Phonetic modifiers (Ș ː ˆ ˈ ʻ ʿ ʾ), combining diacritics (̄ ́ ̃), Latin (Ð þ ŏ ǐ ɪ ˚ ɔ ʃ ɛ) |

**Definitive file:** `tokenizador_bpe_32k_v7.model` (32,308 tokens)

Final audit of wiki_v23 against v7: **0 missing**.
Audit of all training corpora against v7: **0 missing**.

---

### 5. Checkpoint expansion v2→v7

#### Critical bug fixed in bridge.m

When trying to load the expanded checkpoint with the new tokenizer, ARK produced a **segmentation fault** immediately after Metal graph compilation.

**Root cause:** In `objc/bridge.m`, function `ark_mps_get_embed_ptr`, line ~558, the Metal buffer `g_buf_embed` was allocated only once using the condition:

```objc
if (!g_buf_embed)
    g_buf_embed = [g_dev newBufferWithLength:(size_t)vocab_size*G_D*2 options:sh];
```

When changing `vocab_size` from 32,063 to any larger value, the buffer was not reallocated. The pointer pointed to a memory block of size `32063×768×2` bytes while the code tried to write `new_vocab×768×2` bytes → buffer overflow → segfault.

**Fix applied with sed:**

```bash
sed -i '' 's/    if (!g_buf_embed)/    size_t needed_embed = (size_t)vocab_size*G_D*2;\n    if (!g_buf_embed || g_buf_embed.length != needed_embed)/' \
  ~/Documents/ark/rust/ark050/objc/bridge.m
```

Result in `bridge.m`:

```objc
size_t needed_embed = (size_t)vocab_size*G_D*2;
if (!g_buf_embed || g_buf_embed.length != needed_embed)
    g_buf_embed = [g_dev newBufferWithLength:needed_embed options:MTLResourceStorageModeShared];
```

With this fix, the buffer is automatically reallocated whenever `vocab_size` changes. No further modifications to `bridge.m` are needed for future tokenizer expansions.

#### Why Python was used instead of expand_checkpoint.js

The `expand_checkpoint.js` script (documented in previous versions) failed when trying to read the checkpoint with:

```
RangeError [ERR_FS_FILE_TOO_LARGE]: File size (2370084076) is greater than 2 GiB
```

Node.js v25.2.1 cannot load files larger than 2GB with `readFileSync`. The epoch 1 checkpoint weighs 2.37 GB. `expand_checkpoint.py` was created using `mmap` to process the file in chunks without loading it entirely into RAM.

#### Cascaded expansion process

Each tokenizer expansion required a corresponding checkpoint expansion. 5 chained expansions were performed:

| Step | Input | Output | vocab-old | vocab-new | Time |
|---|---|---|---|---|---|
| v2→v3 | `ckpt_ark_ep1_rot2.bin` | `ckpt_ark_ep2_v32104_expand_rot0.bin` | 32,063 | 32,104 | 5.7s |
| v3→v4 | previous | `ckpt_ark_ep2_v32112_rot0.bin` | 32,104 | 32,112 | ~6s |
| v4→v5 | previous | `ckpt_ark_ep2_v32226_rot0.bin` | 32,112 | 32,226 | ~6s |
| v5→v6 | previous | `ckpt_ark_ep2_v32289_rot0.bin` | 32,226 | 32,289 | ~6s |
| v6→v7 | previous | `ckpt_ark_ep2_v32308_rot0.bin` | 32,289 | 32,308 | ~6s |

For each expansion, the 3 embedding tensors (`embed_w` FP16, `embed_m` FP32, `embed_v` FP32) were expanded by initializing the new rows with the average of the last 8 existing rows. The 810 layer tensors were copied without modification.

**Definitive checkpoint:** `ckpt_ark_ep2_v32308_rot0.bin` — step 352,000, vocab 32,308

#### config.rs and main.rs updates

```bash
# config.rs — sequence of changes
vocab_size: 32_063 → 32_104 → 32_112 → 32_226 → 32_289 → 32_308

# main.rs — banner updated
"Tokenizador: BPE 32k+ v2 — 32063 tokens" → "Tokenizador: BPE 32k+ v7 — 32308 tokens"
```

ARK was recompiled with `cargo build --release` after each change.

#### Successful load verification

```
[checkpoint v4] FP16 native weight load complete: step=352000 | adam=352000 | layers=30
[optimizer] Adam moments restored correctly — 271 tensors, current step=352000
[ark] checkpoint v4 restored — weights + Adam moments, step 352000
vocab_size: 32308 ✓
params: 237.19603M
[ep 1  step      1  g  352001]  loss=4.5821  ppl=97.7  scale=256  skips=0
```

No segfault. No vocab mismatch. First loss of the new training run: 4.58 — reasonable given the change in data distribution.

---

### 6. Directory reorganization

The `entren/` folder was deleted and everything consolidated into `entrenamiento/`:

| Content | Description |
|---|---|
| `ckpt_ark_ep1_rot0/1/2.bin` | Epoch 1 checkpoints (backup) |
| `ckpt_ark_ep2_v32308_rot0.bin` | Definitive epoch 2 checkpoint |
| `tokenizador_bpe_32k_v2.model` | Historical reference |
| `tokenizador_bpe_32k_v7.model` | Definitive tokenizer |
| `wiki_v23.jsonl` | Clean and filtered Wikipedia (338,030 articles) |
| `wiki_disambig.jsonl` | Disambiguation corpus (63,113 docs, 0 contaminated) |
| 29 `.jsonl` corpora | Clean training corpora |
| Cleanup and expansion scripts | `expand_checkpoint.py`, `expandir_vocab_v3..v7.py`, `limpiar_*.py`, `filtrar_wiki.py` |
| Logs | `ark_ep1_seq1024.log`, `ark_ep1_razonamiento.log`, `ark_ep2_corpus_mixto.log` |

Deleted intermediate checkpoints: `ckpt_ark_ep1_rot1_expanded.bin`, `ckpt_ark_ep2_v32104_expand_rot0.bin`, `ckpt_ark_ep2_v32112_rot0.bin`, `ckpt_ark_ep2_v32226_rot0.bin`, `ckpt_ark_ep2_v32289_rot0.bin`.

---

### 7. Epoch 2 start

#### Final training configuration

| Parameter | Epoch 1 | Epoch 2 |
|---|---|---|
| Corpus | `wiki_esencial19.jsonl` | mixed reasoning corpus |
| Tokenizer | v2 — 32,063 tokens | v7 — 32,308 tokens |
| Base checkpoint | step 0 | step 352,000 |
| `lr` | 5e-5 → 2e-5 | 5e-6 |
| `clip` | 0.5 | 0.3 |
| `seq` | 128 → 512 → 1024 | 1024 |
| `batch` | 1 | 1 |

#### Epoch 2 corpora (high priority)

24 corpora in interleaved order by category:

```
identidad_eko.jsonl, nous_contexto.jsonl,
cn1_norm.jsonl, cn2_norm.jsonl, cn3_norm.jsonl,
razonamiento_profundo_v2.jsonl, debug_logico.jsonl,
primeros_principios.jsonl, inferencia_abductiva.jsonl,
pensamiento_sistemico.jsonl, teoria_mente_sesgos.jsonl,
eml_matematica_logica.jsonl, gsm8k_reasoning_es_final.jsonl,
gsm_hard_final.jsonl, mcot_math_es_final.jsonl,
corpus_abduccion_final.jsonl, aya_reasoning_es.jsonl,
curiosidades_mundo.jsonl, corpus_en_pregunta_es_respuesta.jsonl,
razonamiento_es_completo.jsonl, lenguaje_figurado_es.jsonl,
lenguaje_claro_final.jsonl, lingcomp_final.jsonl,
wiki_disambig.jsonl
```

Total: ~358,000 documents. Deferred to epoch 3: `tatoeba_es_corpus.jsonl` (184k short sentences), `conversanatural_norm.jsonl` (140k simple dialogue), `opensubs_norm.jsonl` (110k subtitles), `alpaca_es_norm.jsonl` and `somos_alpaca_final.jsonl` (generic instructions).

#### Active command

```bash
nohup caffeinate -i ./target/release/ark \
  --ckpt=../entrenamiento/ckpt_ark_ep2_v32308_rot0.bin \
  --vocab=../entrenamiento/tokenizador_bpe_32k_v7.model \
  --corpus=../entrenamiento/identidad_eko.jsonl,../entrenamiento/nous_contexto.jsonl,\
../entrenamiento/cn1_norm.jsonl,../entrenamiento/cn2_norm.jsonl,\
../entrenamiento/cn3_norm.jsonl,../entrenamiento/razonamiento_profundo_v2.jsonl,\
[...24 corpora total...] \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=1024 --batch=1 --lr=5e-6 --clip=0.3 \
  --epochs=1 >> ../entrenamiento/ark_ep2_corpus_mixto.log 2>/dev/null &
```

> **Note:** `2>/dev/null` silences ANE warnings from the operating system (incompatible element type between Metal GPU and ANE for FP32 operations). These messages are emitted by macOS directly before ARK can capture them and do not affect training — all operations execute correctly on Metal GPU.

#### Startup confirmation

```
vocab_size: 32308
step=352000 | adam=352000
params: 237.19603M
[ep 1  step  1  g  352001]  loss=3.8848  ppl=48.7  scale=256  skips=0
```

Startup loss of 3.88 with pure reasoning corpus, significantly better than the stackoverflow startup (7.40) and comparable to the previous reasoning-only startup (4.58 with seq=128).

---

### Pending — Before inference on Ryzen

| Action | Description |
|---|---|
| Export v7 JSONs | `vocab_sp_v7.json` and `vocab_scores_v7.json` from `tokenizador_bpe_32k_v7.model` using `exportar_vocab_v2.py` |
| Update `eko_infer` | Change `--vocab-size` from 32,063 to 32,308 |
| First epoch 2 inference | Validate quality with stable epoch 2 checkpoint (~step 353,000+) |
| Review pending wiki articles | From Mac with Excel — additional filtering of niche articles |

---

*ARK is developed by Benjamín Alonso Carmona Vega / IAsesoria Informática, Villarrica, Chile.*
*Development assisted by Claude Sonnet (Anthropic).*


*ARK is developed by Benjamín Alonso Carmona Vega / IAsesoria Informática, Villarrica, Chile.*
*Development assisted by Claude Sonnet (Anthropic) and Gemini Pro (Google).*
