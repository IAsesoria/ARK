# BPE Vocabulary Management — ARK / EKO
## Expansions, corrections and technical process

**NOUS Project · ARK Engine · IAsesoria Informática · Villarrica, Chile · 2026**

---

## Tokenizer version history

| Version | File | Tokens | Date | Reason |
|---------|------|--------|------|--------|
| v1 | `tokenizador_bpe_32k.model` | 32,000 | Apr-2026 | Initial BPE training on Spanish corpus |
| v2 | `tokenizador_bpe_32k_v2.model` | 32,063 | 13-May-2026 | Expansion +63 special characters |
| v3 | `tokenizador_bpe_32k_v3.model` | 32,065 | *pending* | Fix: add `ü` / `Ü` omitted in v2 |

---

## Expansion v1 → v2: +63 characters

### Why it was necessary

During Epoch 1 training (~step 300,000), inference tests on the Ryzen machine detected `<unk>` tokens in the model's output. Corpus audit revealed that mathematical, scientific, and high-frequency ASCII characters had no coverage in the base vocabulary of 32,000 tokens — producing noise in the embeddings and dispersion in learning.

### Characters added — the 63 tokens

**Lowercase Greek letters (25):**
```
α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω
```

**Uppercase Greek letters (10):**
```
Γ Δ Θ Λ Σ Φ Χ Ψ Ω Π
```

**Superscripts (5):**
```
² ³ ¹ ⁴ ⁰
```

**Subscripts (3):**
```
₀ ₁ ₂
```

**Mathematical operators (8):**
```
° × √ ≈ ∫ → ± ·
```

**Fraction / currency / ordinal (4):**
```
½ € º ª
```

**Missing ASCII (8):**
```
& # \ ~ ^ @ ` ÷
```

**Total: 25 + 10 + 5 + 3 + 8 + 4 + 8 = 63 tokens**

### Omission detected afterwards

The Spanish diaeresis characters `ü` (lowercase) and `Ü` (uppercase) **were not included** in this expansion. The error was identified when reviewing the corpus cleaning script `limpiar_wiki_v3.py`, which contained the following line in its transliteration table:

```python
'ü': 'u', 'Ü': 'u',
```

This line converted all occurrences of `ü`/`Ü` to `u` during corpus preprocessing. As a result:

- The corpus `wiki_esencial19.jsonl` contains no `ü` (verified with `grep`)
- The v2 tokenizer also lacks `ü`/`Ü` as dedicated tokens
- Words like "pingüino", "vergüenza", "lingüística", "cigüeña" became "pinguino", "verguenza", "linguistica", "ciguena"

The impact on Epoch 1 is marginal — approximately 250 low-frequency words in encyclopedic text. However, the Epoch 2 corpus (Spanish reasoning) contains these letters correctly spelled, so they must be incorporated before starting Epoch 2.

---

### Technical process — how v1 → v2 was expanded

The process uses the SentencePiece `.proto` file to add `USER_DEFINED` tokens without retraining the BPE model or altering existing merges.

```python
# expandir_vocab.py — process used for v1 → v2
import sentencepiece_model_pb2

NEW_CHARACTERS = [
    # Lowercase Greek
    'α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ',
    'ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω',
    # Uppercase Greek
    'Γ','Δ','Θ','Λ','Σ','Φ','Χ','Ψ','Ω','Π',
    # Superscripts
    '²','³','¹','⁴','⁰',
    # Subscripts
    '₀','₁','₂',
    # Mathematical operators
    '°','×','√','≈','∫','→','±','·',
    # Fraction / currency / ordinal
    '½','€','º','ª',
    # Missing ASCII
    '&','#','\\','~','^','@','`','÷',
]

m = sentencepiece_model_pb2.ModelProto()
with open('tokenizador_bpe_32k.model', 'rb') as f:
    m.ParseFromString(f.read())

for char in NEW_CHARACTERS:
    new_token = m.SentencePiece()
    new_token.piece = char
    new_token.score = 0.0   # score 0 = user token
    new_token.type  = 4     # USER_DEFINED — does not interfere with BPE merges
    m.pieces.append(new_token)

with open('tokenizador_bpe_32k_v2.model', 'wb') as f:
    f.write(m.SerializeToString())

print(f"v2 generated: {len(m.pieces)} tokens")  # → 32063
```

**Post-expansion verification:**

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('tokenizador_bpe_32k_v2.model')

print(sp.GetPieceSize())          # → 32063
print(sp.PieceToId('α'))          # → must be ≥ 32000, not 0
print(sp.PieceToId('÷'))          # → must be 32062
print(sp.IdToPiece(32000))        # → first new token
print(sp.IdToPiece(32062))        # → last new token (÷)
```

---

## Expansion v2 → v3: add `ü` / `Ü`

### When to execute

Before starting Epoch 2. The Spanish reasoning corpus contains `ü`/`Ü` correctly spelled — the tokenizer must recognize them and the embedding must have vectors for them.

### Step 1 — Generate tokenizer v3

```python
# expandir_vocab_v3.py
import sentencepiece_model_pb2

NEW_V3 = ['ü', 'Ü']  # resulting IDs: 32063 and 32064

m = sentencepiece_model_pb2.ModelProto()
with open('tokenizador_bpe_32k_v2.model', 'rb') as f:
    m.ParseFromString(f.read())

for char in NEW_V3:
    new_token = m.SentencePiece()
    new_token.piece = char
    new_token.score = 0.0
    new_token.type  = 4     # USER_DEFINED
    m.pieces.append(new_token)

with open('tokenizador_bpe_32k_v3.model', 'wb') as f:
    f.write(m.SerializeToString())

print(f"v3 generated: {len(m.pieces)} tokens")  # → 32065
```

### Step 2 — Verify IDs

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('tokenizador_bpe_32k_v3.model')

id_lower = sp.PieceToId('ü')
id_upper = sp.PieceToId('Ü')

print(f"ID ü: {id_lower}")   # must be 32063
print(f"ID Ü: {id_upper}")   # must be 32064

assert id_lower != 0, "FAIL: ü was not added"
assert id_upper != 0, "FAIL: Ü was not added"
assert id_lower == 32063, f"Wrong ID: {id_lower}"
assert id_upper == 32064, f"Wrong ID: {id_upper}"

print("Tokenizer v3 verified ✓")
```

### Step 3 — Expand the checkpoint

The model embedding has shape `[32063, 768]`. Adding 2 tokens requires expanding it to `[32065, 768]`. The 3 affected tensors are:

| Tensor | Type | Original shape | New shape |
|--------|------|---------------|-----------|
| `embed_w` | FP16 (GPU) | `[32063, 768]` | `[32065, 768]` |
| `embed_m` | FP32 (Adam m) | `[32063, 768]` | `[32065, 768]` |
| `embed_v` | FP32 (Adam v) | `[32063, 768]` | `[32065, 768]` |

The remaining 810 tensors (weights and moments of the 30 layers) are not modified.

```bash
node expand_checkpoint.js \
  --input  ~/Documents/ark/entren/ckpt_ark_ep1_rot2.bin \
  --output ~/Documents/ark/entren/ckpt_ark_ep2_v32065_rot0.bin \
  --vocab-old 32063 \
  --vocab-new 32065 \
  --d-model 768
```

The 2 new vectors are initialized with the **average of the 8 preceding tokens** in the embedding — low-frequency token zone, far better than random Xavier initialization.

### Step 4 — Update config.rs and recompile

```rust
// src/config.rs
// Before:
pub const VOCAB_SIZE: usize = 32063;
// After:
pub const VOCAB_SIZE: usize = 32065;
```

```bash
cd ~/Documents/ark/rust/ark050
cargo build --release
```

### Step 5 — Verify loading

```bash
./target/release/ark \
  --ckpt  ../entren/ckpt_ark_ep2_v32065_rot0.bin \
  --vocab ../entren/tokenizador_bpe_32k_v3.model \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=128 --batch=1 --epochs=0
```

**Signs of success:**
- `[checkpoint v4] step=XXXXXX` — the step is the last one from Epoch 1, not 0
- No `vocab mismatch` or `tensor shape error` appears
- `vocab_size: 32065` shown in the active configuration

---

## General guide — how to add future characters

This same process applies to any future vocabulary expansion. The steps are always the same:

### 1. Identify the problem
```bash
# Audit which corpus characters have no coverage
grep -oP '[^\x00-\x7F]' corpus.jsonl | sort | uniq -c | sort -rn | head -50
```

### 2. Verify before adding
```python
# Check if the character already exists in the vocab
sp.PieceToId('X')  # if it returns 0 → does not exist → add it
```

### 3. Add with USER_DEFINED type
```python
new_token.type = 4  # USER_DEFINED — never retrain full BPE
```

### 4. Expand checkpoint with the script
```bash
node expand_checkpoint.js \
  --input  current_ckpt.bin \
  --output expanded_ckpt.bin \
  --vocab-old N_CURRENT \
  --vocab-new N_NEW \
  --d-model 768
```

### 5. Update VOCAB_SIZE in config.rs and recompile

### General rule
> Never retrain the full BPE tokenizer to add individual characters.
> `USER_DEFINED` tokens are appended to the end of the vocabulary without altering
> existing merges or the IDs of the preceding 32,063 tokens.

---

## Current vocabulary state

| ID range | Content | Tokens |
|----------|---------|--------|
| 0–4 | Special tokens: `<unk>` `<s>` `</s>` `<PAD>` `<OOV>` | 5 |
| 5–12 | Control tokens: `<USR>` `<SYS>` `<FIN>` etc. | 8 |
| 13–31999 | BPE Spanish vocabulary (subwords) | ~31,987 |
| 32000–32062 | v2 expansion: +63 special characters | 63 |
| 32063–32064 | **v3 expansion (pending):** `ü` `Ü` | 2 |

---

## Reference files

```
~/Documents/ark/
├── entren/
│   ├── tokenizador_bpe_32k.model      # v1 — base 32,000 tokens (legacy)
│   ├── tokenizador_bpe_32k_v2.model   # v2 — active Epoch 1, 32,063 tokens
│   └── tokenizador_bpe_32k_v3.model   # v3 — pending Epoch 2, 32,065 tokens
└── documentacion/
    ├── expand_checkpoint.js            # checkpoint expansion script
    ├── expandir_vocab_v3.py            # tokenizer expansion script
    └── vocabulary_bpe_management.md    # this document
```

---

*Technical document — ARK Engine / NOUS Project*
*Benjamín Alonso Carmona Vega — IAsesoria Informática — Villarrica, Chile · 2026*
*Written with assistance from Claude Sonnet (Anthropic)*
