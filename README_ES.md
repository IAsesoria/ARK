# ARK — Motor de Entrenamiento LLM desde Cero

### EKO · Proyecto NOUS · IAsesoria Informática · Villarrica, Chile · 2026

**Dashboard de entrenamiento en vivo:** [iasesoria.github.io/ARK](https://iasesoria.github.io/ARK/)

> 🇬🇧 [English version](README.md)

> Sin PyTorch. Sin TensorFlow. Sin GPU en la nube.

---

## Qué es

ARK es un motor de entrenamiento de modelos de lenguaje grande escrito íntegramente desde cero en **Rust, Objective-C y ensamblador NEON AArch64**. No depende de PyTorch ni TensorFlow. Cada capa del stack de cómputo — forward en GPU, backward en GPU vía MPSGraph AutoGrad, optimizador en ensamblador, kernels matemáticos propios — está escrita para exprimir al máximo Apple Silicon, concretamente M1.

El primer modelo que ARK entrena se llama **EKO**, parte del **Proyecto NOUS**. EKO es un transformer de 237 millones de parámetros entrenado principalmente con corpus en español — contenido enciclopédico, matemático y conversacional — todo construido y depurado específicamente para este proyecto.

Lo que pretendo es crear una IA desde cero, con hardware de consumo, en español, desde Chile, yendo a contracorriente al no usar frameworks típicos, y además en este proceso aprender y perseguir la curiosidad que inició meses atrás tratando de responder; "¿Es posible crear inteligencia artificial con más control y autonomía desde sus bases?".

---

## Por qué importa

La mayoría de los modelos de lenguaje se entrenan en clústeres de GPU NVIDIA que cuestan decenas o cientos de miles de dólares. ARK busca tres cosas:

**1. Eficiencia real en hardware.**
Un MacBook Air M1 de 8GB es hardware de consumo masivo. ARK lo aprovecha al máximo: arquitectura Zero-Copy sobre memoria unificada, forward en GPU vía MPSGraph con grafos compilados AOT, backward nativo en GPU vía AutoGrad con `gradientForPrimaryTensor:withTensors:`, optimizador AdamW en ensamblador SIMD. Sin copias innecesarias entre CPU y GPU. Sin overhead de frameworks.

**2. Soberanía tecnológica.**
El corpus está en español. El proyecto es chileno. El modelo es completamente propio. Busco no depender dentro de lo posible de APIs externas, sin modelos de lenguaje ni bases de terceros, sin infraestructura en la nube. El entrenamiento y a posterior la inferencia son 100% locales en mi Mac M1.

**3. Reproducibilidad total.**
ARK compila y corre en cualquier Mac con Apple Silicon. Los únicos requisitos son Rust y Xcode Command Line Tools. El código es transparente: lo que lees es exactamente lo que ejecuta.

---

## Arquitectura del sistema

El pipeline divide el trabajo entre las tres unidades de cómputo disponibles en el chip M1:

### GPU — MPSGraph (forward completo + backward AutoGrad nativo)

El forward pass completo corre en GPU vía Metal Performance Shaders Graph (MPSGraph), con grafos compilados AOT antes del primer paso de entrenamiento:

- **Arquitectura de inferencia (EKO):** Transformer de 30 capas (RMSNorm → Atención SDPA multi-cabeza → FFN SwiGLU). Implementa Rotary Position Embedding (RoPE) con tablas sin/cos precalculadas y LM Head con weight tying al embedding.
- **Forward pass acelerado (GPU):** Ejecutado en Apple Silicon vía MPSGraph, con grafos compilados AOT antes del primer paso y máscara causal fusionada nativamente en el SDPA.
- **Backward pass (AutoGrad):** El backward de las 30 capas y el LM Head corre en GPU vía AutoGrad nativo de MPSGraph (`gradientForPrimaryTensor:withTensors:`). La diferenciación simbólica genera automáticamente los gradientes para los 10 tensores de cada capa (dx, dwq, dwk, dwv, dwo, dw1, dw2, dw3, dg1, dg2). Sin implementación manual de reglas de la cadena.
- **Optimización de memoria (UMA):** Optimizador AdamW Zero-Copy. Aprovecha la arquitectura de memoria unificada del M1 escribiendo directamente en los `MTLBuffer` compartidos en FP16, eliminando cuellos de botella por `memcpy`.
- **Pipeline de entrenamiento:** Mixed Precision Training (AMP) con loss scaling dinámico, streaming de corpus zero-copy sin archivos intermedios, y sistema de checkpoints rotativos a prueba de fallos (3 slots en disco).

La cross-entropy (log-sum-exp numéricamente estable) y el scatter-add del embedding corren en CPU vía Accelerate.

### CPU — Accelerate / AMX

- Cross-entropy numéricamente estable con `vDSP_maxv` y log-sum-exp
- Scatter-acumulación de gradientes en filas del embedding (secuencial, maneja tokens repetidos)
- Norma L2 global de gradientes con `vDSP_svesq` antes del paso Adam

### Ensamblador AArch64 NEON — optimizador y kernels matemáticos

El optimizador AdamW y los kernels matemáticos están escritos directamente en ensamblador AArch64:

- **`asm/opti.s`** — AdamW completo: fórmula Adam vectorizada, 4 floats por ciclo SIMD, corrección de bias, weight decay, grad clip L2 global.
- **`asm/kern.s`** — RMSNorm FP32/FP16 (forward + backward v0.62 con 4 bugs corregidos), softmax FP32/FP16 (3 pasadas, segura ante underflow), SwiGLU forward FP16, gather embedding FP16→FP32, dequant/quant FP16↔FP32 (ancho 8), dot product FP16 con acumulación FP32.
- **`asm/aten.metal`** — Kernels Metal de atención: `attention_scores_f16`, `attention_softmax_f32`, `attention_weighted_sum_f16`, kernel fusionado `attention_full_f16_causal`.

### Zero-Copy sobre memoria unificada

Todos los pesos del modelo viven en `MTLBuffer` con `storageModeShared`. CPU y GPU acceden al mismo bloque de memoria física — sin copias intermedias. El optimizador en ensamblador recibe un puntero directo a la VRAM, lee los pesos FP16, los actualiza in-place, y los escribe de vuelta en la misma dirección que leerá la GPU en el siguiente forward.

---

## EKO — Especificaciones del modelo

| Parámetro                 | Valor                                         |
|----------------------     |-----------------------------------------------|
| Parámetros totales        | 237M                                          |
| Capas transformer         | 30                                            |
| d_model                   | 768                                           |
| Cabezas de atención       | 12 (head_dim = 64)                            |
| FFN hidden                | 2048 (activación SwiGLU)                      |
| Vocabulario               | BPE 32.063 tokens (SentencePiece, español)    |
| Codificación posicional   | RoPE                                          |
| Normalización             | RMSNorm (gamma excluido de weight decay)      |
| Precisión                 | AMP: pesos FP16 / gradientes y momentos FP32  |

> En M1 8GB, seq y batch más grandes están limitados por la memoria unificada disponible. Por eso es especialmente valioso que la comunidad pruebe en hardware con más RAM — ver la sección de pruebas de hardware más abajo.

---

## Precisión mixta automática (AMP)

Los pesos se almacenan en FP16 en VRAM. Los gradientes y momentos Adam (m, v) se mantienen en FP32. El scaler dinámico arranca en 256 y puede subir hasta 8192 en pasos de ×2 cada 2.000 pasos limpios. Si aparece NaN/Inf, el paso se descarta, el scale se divide a la mitad, y el entrenamiento continúa desde el checkpoint intacto sin corrupción del estado del optimizador.

---

## Tokenizador

EKO utiliza un tokenizador BPE entrenado con SentencePiece sobre corpus en español.

**Archivo activo:** `tokenizador_bpe_32k_v2.model` — 32.063 tokens

El vocabulario base de 32.000 tokens fue ampliado en +63 tokens para cubrir símbolos matemáticos, científicos y caracteres ASCII que aparecen con alta frecuencia en el corpus pero no estaban representados:

- Letras griegas minúsculas (25): α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω
- Letras griegas mayúsculas (10): Γ Δ Θ Λ Σ Φ Χ Ψ Ω Π
- Superíndices (5): ² ³ ¹ ⁴ ⁰
- Subíndices (3): ₀ ₁ ₂
- Operadores matemáticos (8): ° × √ ≈ ∫ → ± ·
- Fracción / moneda / ordinal (4): ½ € º ª
- ASCII faltantes (8): & # \ ~ ^ @ ` ÷

> **Pendiente v3:** La diéresis española `ü`/`Ü` no fue incluida en la expansión y fue transliterada a `u` durante el preprocesado del corpus (pingüino → pinguino). El impacto en Época 1 es marginal dado que son ~250 palabras de baja frecuencia en texto enciclopédico. Se evaluará corregir para Época 2: quitar la transliteración del limpiador y añadir `ü`/`Ü` al vocabulario (+2 tokens → 32.065).
---

## Corpus y curriculum de entrenamiento

**Época 1:** `wiki_esencial19.jsonl` — Wikipedia en español filtrada y depurada.

| Dato               | Valor                |
|--------------------|----------------------|
| Artículos          | 340.275              |
| Tokens estimados   | ~518M                |
| Tamaño en disco    | ~2,1 GB              |
| Tokens por doc     | ~1.524 (promedio)    |

El corpus fue procesado para eliminar caracteres no latinos (cirílico, CJK, árabe) que no tenían cobertura en el vocabulario BPE y causaban tokens `<unk>` en inferencia. Los tokens se cuentan dinámicamente al arrancar ARK muestreando 1.000 documentos — sin estimaciones hardcodeadas.

El corpus no se distribuye con el repositorio por tamaño. Para usar el tuyo propio, cualquier archivo JSONL con un campo `"title"` y `"text"` por línea funciona directamente con `--corpus`.

**Época 2 (planificada):** corpus de razonamiento en español (~176K ejemplos sintéticos) orientado a mejorar la coherencia lógica y la calidad de respuesta.

---

## Compilar y ejecutar

**Requisitos:**
- macOS con Apple Silicon (M1 o superior)
- Rust toolchain (`rustup`)
- Xcode Command Line Tools

**Compilar:**
```bash
cd ark01
cargo build --release
```

**Iniciar entrenamiento desde cero:**

Pasar un nombre de checkpoint que no exista. ARK detecta la ausencia del archivo e inicializa los pesos con Xavier automáticamente.

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

**Reanudar desde checkpoint:**

Pasar el checkpoint más reciente disponible. ARK restaura pesos FP16 y momentos Adam FP32 y continúa desde el paso guardado.

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

**Sistema de checkpoints rotativo:**

ARK guarda automáticamente cada 500 pasos (época 1) o 1000 pasos (época 2) usando 3 slots rotativos: `_rot0.bin`, `_rot1.bin`, `_rot2.bin`. El slot se determina por `(step / ckpt_every) % 3`. Esto garantiza que siempre haya al menos dos checkpoints recientes disponibles ante cualquier fallo.

**Monitorear entrenamiento:**
```bash
tail -f ../entren/ark_ep1_seq1024.log
```

**Inferencia (Ryzen / Windows):**
```powershell
$env:RUSTFLAGS="-C target-cpu=native"
cargo run --release --bin eko_infer -- \
  --ckpt ckpt_ark_ep1_rot2.bin \
  --vocab tokenizador_bpe_32k_v2.model \
  --prompt "La vida es"
```

---

## Hardware probado

| Hardware            | Rol           | Estado      |
|---------------------|---------------|-------------|
| MacBook Air M1 8 GB | Entrenamiento | ✅ Activo   |
| Ryzen 7 (Windows)   | Inferencia    | ✅ Probado  |

Se aceptan contribuciones de pruebas en otro hardware ARM. Si lograste compilar y ejecutar ARK en un equipo distinto, abre un issue con los resultados.

---

## Investigación activa

El diseño de ARK está informado por los siguientes trabajos:

- Vaswani et al. (2017) — *Attention Is All You Need*
- Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Touvron et al. (2023) — *LLaMA: Open and Efficient Foundation Language Models*
- Zhang & Sennrich (2019) — *Root Mean Square Layer Normalization*
- Micikevicius et al. (2018) — *Mixed Precision Training*

---

## Estructura del proyecto

```
proyecto_ark/
├── ark01/                                # Código fuente
│   ├── asm/
│   │   ├── kern.s                        # RMSNorm, softmax, SwiGLU, embed, dequant
│   │   ├── opti.s                        # AdamW FP16/FP32, grad clip L2 (NEON)
│   │   └── aten.metal                    # Kernels Metal de atención
│   ├── eko_infer/                        # Código fuente inferencia (CPU/Windows)
│   ├── entren/                           # Directorio ejemplo (corpus y checkpoints no en repo)
│   ├── objc/
│   │   └── bridge.m                      # MPSGraph fwd/bwd AutoGrad, cross-entropy
│   ├── src/
│   │   ├── main.rs                       # Punto de entrada y parser de argumentos
│   │   ├── config.rs                     # Hiperparámetros y validación
│   │   ├── training.rs                   # Loop de entrenamiento, AMP, checkpointing
│   │   ├── optimizer.rs                  # AdamW Zero-Copy, clip global vDSP
│   │   ├── memory.rs                     # ModelWeights, AlignedVec, f16↔f32
│   │   ├── io.rs                         # CorpusStream JSONL, CheckpointV4, BPE
│   │   └── ffi.rs                        # Bindings seguros Rust↔ObjC↔ASM
│   ├── tokenizador_bpe/                  # Tokenizador BPE (incluido en repo)
│   │   ├── tokenizador_bpe_32k_v2.model  (SentencePiece BPE 32.063 tokens — activo)
│   │   ├── tokenizador_bpe_32k.model     (versión original 32.000 tokens — legacy)
│   │   ├── sentencepiece_model.proto     (definición del modelo SentencePiece)
│   │   ├── tokenizador_bpe_32k.vocab     (vocabulario legible)
│   │   ├── tokenizer_hf.json             (formato HuggingFace)
│   │   └── vocab_bpe_32k.json            (vocab JSON para Rust)
│   ├── build.rs                          # Compila bridge.m + kern.s + opti.s
│   └── cargo.toml
├── LICENSE
├── README.md                             # Documentación en inglés
└── README_ES.md                          # Documentación en español
```

---

## Contacto y colaboración

El código es libre (MIT), el tiempo no lo es. Si necesitas adaptación, integración, consultoría técnica o co-desarrollo:

[benjaminalonsocarmona@gmail.com](mailto:benjaminalonsocarmona@gmail.com)

Para empresas que facturen usando este código, ofrezco soporte formal con contrato y factura.

---

## Sobre el proyecto

Desarrollado por **Benjamín Alonso Carmona Vega**, fundador de [IAsesoria Informática](https://iasesoria.cl).

Villarrica, Chile · 2026

*Desarrollado con asistencia de Claude Sonnet (Anthropic) y Gemini Pro (Google) para documentación, depuración y revisión arquitectónica.*

---

*El paso 1 registró loss=10.47. El valor teórico para distribución uniforme sobre 32.063 tokens es log(32.063) ≈ 10.37. La diferencia es la inicialización Xavier. Eso es todo — el modelo no sabe nada todavía. Lo que viene después es lo interesante.*
