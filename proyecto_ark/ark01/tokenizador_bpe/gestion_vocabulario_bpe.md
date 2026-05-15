# Gestión del Vocabulario BPE — ARK / EKO
## Expansiones, correcciones y proceso técnico

**Proyecto NOUS · ARK Engine · iAsesoria Informática · Villarrica, Chile · 2026**

---

## Historial de versiones del tokenizador

| Versión | Archivo | Tokens | Fecha | Motivo |
|---------|---------|--------|-------|--------|
| v1 | `tokenizador_bpe_32k.model` | 32.000 | Abr-2026 | Entrenamiento inicial BPE sobre corpus español |
| v2 | `tokenizador_bpe_32k_v2.model` | 32.063 | 13-May-2026 | Expansión +63 caracteres especiales |
| v3 | `tokenizador_bpe_32k_v3.model` | 32.065 | *pendiente* | Corrección: agregar `ü` / `Ü` omitidos en v2 |

---

## Expansión v1 → v2: +63 caracteres

### Por qué fue necesaria

Durante el entrenamiento de Época 1 (~paso 300.000), pruebas de inferencia en el Ryzen detectaron tokens `<unk>` en la salida del modelo. La auditoría del corpus reveló que caracteres matemáticos, científicos y ASCII de alta frecuencia no tenían cobertura en el vocabulario base de 32.000 tokens — produciendo ruido en los embeddings y dispersión en el aprendizaje.

### Caracteres agregados — los 63 tokens

**Letras griegas minúsculas (25):**
```
α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω
```

**Letras griegas mayúsculas (10):**
```
Γ Δ Θ Λ Σ Φ Χ Ψ Ω Π
```

**Superíndices (5):**
```
² ³ ¹ ⁴ ⁰
```

**Subíndices (3):**
```
₀ ₁ ₂
```

**Operadores matemáticos (8):**
```
° × √ ≈ ∫ → ± ·
```

**Fracción / moneda / ordinal (4):**
```
½ € º ª
```

**ASCII faltantes (8):**
```
& # \ ~ ^ @ ` ÷
```

**Total: 25 + 10 + 5 + 3 + 8 + 4 + 8 = 63 tokens**

### Omisión detectada posteriormente

La diéresis española `ü` (minúscula) y `Ü` (mayúscula) **no fueron incluidas** en esta expansión. El error fue identificado al revisar el script de limpieza del corpus `limpiar_wiki_v3.py`, que contenía la siguiente línea en su tabla de transliteración:

```python
'ü': 'u', 'Ü': 'u',
```

Esta línea convirtió todas las apariciones de `ü`/`Ü` en `u` durante el preprocesado del corpus. Como resultado:

- El corpus `wiki_esencial19.jsonl` no contiene ninguna `ü` (verificado con `grep`)
- El tokenizador v2 tampoco tiene `ü`/`Ü` como tokens propios
- Palabras como "pingüino", "vergüenza", "lingüística", "cigüeña" quedaron escritas como "pinguino", "verguenza", "linguistica", "ciguena"

El impacto en Época 1 es marginal — son aproximadamente 250 palabras de baja frecuencia en texto enciclopédico. Sin embargo, el corpus de Época 2 (razonamiento en español) contiene estas letras correctamente escritas, por lo que deben incorporarse antes de iniciar Época 2.

---

### Proceso técnico — cómo se expandió v1 → v2

El proceso usa el archivo `.proto` de SentencePiece para agregar tokens de usuario (`USER_DEFINED`) sin reentrenar el modelo BPE ni alterar los merges existentes.

```python
# expandir_vocab.py — proceso usado para v1 → v2
import sentencepiece_model_pb2

NUEVOS_CARACTERES = [
    # Griegas minúsculas
    'α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ',
    'ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω',
    # Griegas mayúsculas
    'Γ','Δ','Θ','Λ','Σ','Φ','Χ','Ψ','Ω','Π',
    # Superíndices
    '²','³','¹','⁴','⁰',
    # Subíndices
    '₀','₁','₂',
    # Operadores matemáticos
    '°','×','√','≈','∫','→','±','·',
    # Fracción / moneda / ordinal
    '½','€','º','ª',
    # ASCII faltantes
    '&','#','\\','~','^','@','`','÷',
]

m = sentencepiece_model_pb2.ModelProto()
with open('tokenizador_bpe_32k.model', 'rb') as f:
    m.ParseFromString(f.read())

for char in NUEVOS_CARACTERES:
    nuevo = m.SentencePiece()
    nuevo.piece = char
    nuevo.score = 0.0   # score 0 = token de usuario
    nuevo.type  = 4     # USER_DEFINED — no interfiere con BPE merges
    m.pieces.append(nuevo)

with open('tokenizador_bpe_32k_v2.model', 'wb') as f:
    f.write(m.SerializeToString())

print(f"v2 generado: {len(m.pieces)} tokens")  # → 32063
```

**Verificación post-expansión:**

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('tokenizador_bpe_32k_v2.model')

print(sp.GetPieceSize())          # → 32063
print(sp.PieceToId('α'))          # → debe ser ≥ 32000, no 0
print(sp.PieceToId('÷'))          # → debe ser 32062
print(sp.IdToPiece(32000))        # → primer token nuevo
print(sp.IdToPiece(32062))        # → último token nuevo (÷)
```

---

## Expansión v2 → v3: agregar `ü` / `Ü`

### Cuándo ejecutar

Antes de iniciar Época 2. El corpus de razonamiento en español contiene `ü`/`Ü` correctamente escritas — el tokenizador debe reconocerlas y el embedding debe tener vectores para ellas.

### Paso 1 — Generar tokenizador v3

```python
# expandir_vocab_v3.py
import sentencepiece_model_pb2

NUEVOS_V3 = ['ü', 'Ü']  # IDs resultantes: 32063 y 32064

m = sentencepiece_model_pb2.ModelProto()
with open('tokenizador_bpe_32k_v2.model', 'rb') as f:
    m.ParseFromString(f.read())

for char in NUEVOS_V3:
    nuevo = m.SentencePiece()
    nuevo.piece = char
    nuevo.score = 0.0
    nuevo.type  = 4     # USER_DEFINED
    m.pieces.append(nuevo)

with open('tokenizador_bpe_32k_v3.model', 'wb') as f:
    f.write(m.SerializeToString())

print(f"v3 generado: {len(m.pieces)} tokens")  # → 32065
```

### Paso 2 — Verificar IDs

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('tokenizador_bpe_32k_v3.model')

id_min = sp.PieceToId('ü')
id_may = sp.PieceToId('Ü')

print(f"ID ü: {id_min}")   # debe ser 32063
print(f"ID Ü: {id_may}")   # debe ser 32064

assert id_min != 0, "FALLO: ü no fue agregado"
assert id_may != 0, "FALLO: Ü no fue agregado"
assert id_min == 32063, f"ID incorrecto: {id_min}"
assert id_may == 32064, f"ID incorrecto: {id_may}"

print("Tokenizador v3 verificado ✓")
```

### Paso 3 — Expandir el checkpoint

El embedding del modelo tiene forma `[32063, 768]`. Al agregar 2 tokens, debe expandirse a `[32065, 768]`. Los 3 tensores afectados son:

| Tensor | Tipo | Forma original | Forma nueva |
|--------|------|---------------|-------------|
| `embed_w` | FP16 (GPU) | `[32063, 768]` | `[32065, 768]` |
| `embed_m` | FP32 (Adam m) | `[32063, 768]` | `[32065, 768]` |
| `embed_v` | FP32 (Adam v) | `[32063, 768]` | `[32065, 768]` |

Los 810 tensores restantes (pesos y momentos de las 30 capas) no se modifican.

```bash
node expand_checkpoint.js \
  --input  ~/Documents/ark/entren/ckpt_ark_ep1_rot2.bin \
  --output ~/Documents/ark/entren/ckpt_ark_ep2_v32065_rot0.bin \
  --vocab-old 32063 \
  --vocab-new 32065 \
  --d-model 768
```

Los 2 vectores nuevos se inicializan con el **promedio de los 8 tokens anteriores** del embedding — zona de tokens de baja frecuencia, mucho mejor que Xavier aleatorio.

### Paso 4 — Actualizar config.rs y recompilar

```rust
// src/config.rs
// Antes:
pub const VOCAB_SIZE: usize = 32063;
// Después:
pub const VOCAB_SIZE: usize = 32065;
```

```bash
cd ~/Documents/ark/rust/ark050
cargo build --release
```

### Paso 5 — Verificar carga

```bash
./target/release/ark \
  --ckpt  ../entren/ckpt_ark_ep2_v32065_rot0.bin \
  --vocab ../entren/tokenizador_bpe_32k_v3.model \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=128 --batch=1 --epochs=0
```

**Señales de éxito:**
- `[checkpoint v4] step=XXXXXX` — el paso es el último de Época 1, no 0
- No aparece `vocab mismatch` ni `tensor shape error`
- `vocab_size: 32065` en la configuración activa

---

## Guía general — cómo agregar caracteres futuros

Este mismo proceso aplica para cualquier expansión futura del vocabulario. Los pasos son siempre los mismos:

### 1. Identificar el problema
```bash
# Auditar qué caracteres del corpus no tienen cobertura
grep -oP '[^\x00-\x7F]' corpus.jsonl | sort | uniq -c | sort -rn | head -50
```

### 2. Verificar antes de agregar
```python
# Comprobar si el caracter ya existe en el vocab
sp.PieceToId('X')  # si devuelve 0 → no existe → agregar
```

### 3. Agregar con tipo USER_DEFINED
```python
nuevo.type = 4  # USER_DEFINED — nunca reentrenar BPE completo
```

### 4. Expandir checkpoint con el script
```bash
node expand_checkpoint.js \
  --input  ckpt_actual.bin \
  --output ckpt_expandido.bin \
  --vocab-old N_ACTUAL \
  --vocab-new N_NUEVO \
  --d-model 768
```

### 5. Actualizar VOCAB_SIZE en config.rs y recompilar

### Regla general
> Nunca reentrenar el tokenizador BPE completo para agregar caracteres individuales.
> Los tokens `USER_DEFINED` se agregan al final del vocabulario sin alterar los merges
> existentes ni los IDs de los 32.063 tokens anteriores.

---

## Estado actual del vocabulario

| Rango de IDs | Contenido | Tokens |
|-------------|-----------|--------|
| 0–4 | Tokens especiales: `<unk>` `<s>` `</s>` `<PAD>` `<OOV>` | 5 |
| 5–12 | Tokens de control: `<USR>` `<SYS>` `<FIN>` etc. | 8 |
| 13–31999 | Vocabulario BPE español (subpalabras) | ~31.987 |
| 32000–32062 | Expansión v2: +63 caracteres especiales | 63 |
| 32063–32064 | **Expansión v3 (pendiente):** `ü` `Ü` | 2 |

---

## Archivos de referencia

```
~/Documents/ark/
├── entren/
│   ├── tokenizador_bpe_32k.model      # v1 — base 32.000 tokens (legacy)
│   ├── tokenizador_bpe_32k_v2.model   # v2 — activo Época 1, 32.063 tokens
│   └── tokenizador_bpe_32k_v3.model   # v3 — pendiente Época 2, 32.065 tokens
└── documentacion/
    ├── expand_checkpoint.js            # script de expansión de checkpoint
    ├── expandir_vocab_v3.py            # script de expansión de tokenizador
    └── gestion_vocabulario_bpe.md      # este documento
```

---

*Documento técnico — ARK Engine / Proyecto NOUS*
*Benjamín Alonso Carmona Vega — iAsesoria Informática — Villarrica, Chile · 2026*
*Redactado con asistencia de Claude Sonnet (Anthropic)*
