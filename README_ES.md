# ARK — Motor de Entrenamiento LLM desde Cero

### EKO · Proyecto NOUS · IAsesoria Informática · Villarrica, Chile · 2026

## **Actualización entrenamiento:** [iasesoria.github.io/ARK](https://iasesoria.github.io/ARK/)

> 🇬🇧 [English version](README.md)

> Sin PyTorch. Sin TensorFlow. Sin GPU en la nube.

---

## Qué es esto

ARK es un motor de entrenamiento de modelos de lenguaje grande escrito íntegramente desde cero en **Rust, Objective-C y ensamblador NEON AArch64**. No depende de PyTorch, TensorFlow ni de ningún framework de deep learning. Cada capa del stack de cómputo — forward en GPU, backward en GPU vía MPSGraph AutoGrad, optimizador en ensamblador, kernels matemáticos propios — está escrita para exprimir al máximo Apple Silicon, concretamente M1.

El primer modelo que ARK entrena se llama **EKO**, parte del **Proyecto NOUS**. EKO es un transformer de 237 millones de parámetros entrenado principalmente con corpus en español — contenido enciclopédico, matemático y conversacional — todo construido y depurado específicamente para este proyecto.

Lo que pretendo es crear una IA desde cero, con hardware de consumo, en español, desde Chile, yendo a contracorriente al no usar frameworks típicos, y además en este proceso aprender y calmar la curiosidad que dio inicio a todo esto meses atrás.

> **Estado actual:** La época 1 está entrenando activamente. El modo de inferencia aún no está implementado. Este es un proyecto vivo — transparente sobre lo que funciona, lo que se está mejorando, y lo que viene.

---

## Por qué importa

La mayoría de los modelos de lenguaje se entrenan en clústeres de GPU NVIDIA que cuestan decenas o cientos de miles de dólares. ARK busca tres cosas:

**1. Eficiencia real en hardware limitado.**
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

- Lookup de embedding (FP16)
- 30 capas transformer (EKO): RMSNorm → atención SDPA multi-cabeza → FFN SwiGLU → RMSNorm
- Rotary Position Embedding (RoPE) con tablas sin/cos precalculadas en CPU
- Máscara causal fusionada nativamente en SDPA
- LM head con weight tying al embedding

El backward de las 30 capas y el LM head corre en GPU vía AutoGrad nativo de MPSGraph (`gradientForPrimaryTensor:withTensors:`). Cada grafo de capa define un `loss_proxy` como la suma de reducción de `t_out × d_out`, y la diferenciación simbólica genera automáticamente los gradientes para los 10 tensores de cada capa (dx, dwq, dwk, dwv, dwo, dw1, dw2, dw3, dg1, dg2). Sin implementación manual de reglas de la cadena.

La cross-entropy (log-sum-exp numéricamente estable) y el scatter-add del embedding corren en CPU vía Accelerate.

### CPU — Accelerate / AMX

- Cross-entropy numéricamente estable con `vDSP_maxv` y log-sum-exp
- Scatter-acumulación de gradientes en filas del embedding (secuencial, maneja tokens repetidos)
- Norma L2 global de gradientes con `vDSP_svesq` antes del paso Adam

### Ensamblador AArch64 NEON (optimizador y kernels matemáticos)

El optimizador AdamW y los kernels matemáticos están escritos directamente en ensamblador AArch64:

- **`asm/opti.s`** — AdamW completo: fórmula Adam vectorizada, 4 floats por ciclo SIMD, corrección de bias, weight decay, grad clip L2 global.
- **`asm/kern.s`** — RMSNorm FP32/FP16 (forward + backward v0.62 con 4 bugs corregidos), softmax FP32/FP16 (3 pasadas, segura ante underflow), SwiGLU forward FP16, gather embedding FP16→FP32, dequant/quant FP16↔FP32 (ancho 8), dot product FP16 con acumulación FP32.
- **`asm/aten.metal`** — Kernels Metal de atención: `attention_scores_f16`, `attention_softmax_f32`, `attention_weighted_sum_f16`, kernel fusionado `attention_full_f16_causal`.

### Zero-Copy sobre memoria unificada

Todos los pesos del modelo viven en `MTLBuffer` con `storageModeShared`. CPU y GPU acceden al mismo bloque de memoria física — sin copias intermedias. El optimizador en ensamblador recibe un puntero directo a la VRAM, lee los pesos FP16, los actualiza in-place, y los escribe de vuelta en la misma dirección que leerá la GPU en el siguiente forward.

---

## EKO — Especificaciones del modelo

| Parámetro | Valor |
|---|---|
| Parámetros totales | 237M |
| Capas transformer | 30 |
| d_model | 768 |
| Cabezas de atención | 12 (head_dim = 64) |
| FFN hidden | 2048 (activación SwiGLU) |
| Contexto (época 1) | 128 tokens (inició en 1024 → 512 → 128, ajustado por velocidad en M1 8GB) |
| Contexto (época 2+) | 1024 tokens |
| Vocabulario | BPE 32.000 tokens (SentencePiece, español) |
| Codificación posicional | RoPE |
| Normalización | RMSNorm (gamma excluido de weight decay) |
| Precisión | AMP: pesos FP16 / gradientes y momentos Adam FP32 |

---

## Configuración de entrenamiento — cómo fue evolucionando

La configuración de la época 1 se fue ajustando a medida que el entrenamiento reveló cuellos de botella en el M1 8GB. Este es el historial real:

| Fase | Pasos | seq | batch | lr | Motivo del cambio |
|---|---|---|---|---|---|
| Inicio | 0 – ~15k | 1024 | 1 | 1e-4 | Configuración inicial |
| Ajuste 1 | ~15k – ~79k | 512 | 1 | 1e-4 | Reducido por velocidad |
| Actual | ~79k – presente | 128 | 2 | 5e-5 | Reducido más; lr bajado para estabilidad |

**Configuración activa actual:**
```
Época 1 (base lingüística):
  --corpus=wiki_esencial14.jsonl,wiki_disambig.jsonl
  --layers=30 --heads=12 --d-model=768 --hidden=2048
  --seq=128 --batch=2 --lr=5e-5 --clip=1.0
  --epochs=1

AdamW: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
AMP:   loss_scale_init=256, max=8192, step_up_every=2000
Checkpoint: rotativo de 3 slots, cada 500 pasos
Scheduler: cosine decay de lr a lr/10 en 250k pasos
```

> En M1 8GB, seq y batch más grandes están limitados por la memoria unificada disponible. Por eso es especialmente valioso que la comunidad pruebe en hardware con más RAM — ver la sección de pruebas de hardware más abajo.

---

## Precisión mixta automática (AMP)

Los pesos se almacenan en FP16 en VRAM. Los gradientes y momentos Adam (m, v) se mantienen en FP32. El scaler dinámico arranca en 256 y puede subir hasta 8192 en pasos de ×2 cada 2.000 pasos limpios. Si aparece NaN/Inf, el paso se descarta, el scale se divide a la mitad, y el entrenamiento continúa desde el checkpoint intacto sin corrupción del estado del optimizador.

---

## Corpus y curriculum de entrenamiento

Corpus de época 1: **~617M tokens reales** (calculados muestreando 1.000 documentos al arrancar — sin estimaciones hardcodeadas):

| Corpus | Contenido |
|---|---|
| `wiki_esencial14.jsonl` | Wikipedia en español — 341.147 documentos filtrados (2,1 GB) — **no incluido en el repo** |
| `wiki_disambig.jsonl` | Páginas de desambiguación de Wikipedia — 63.113 documentos — **no incluido en el repo** |

El corpus no se distribuye con el repositorio por tamaño. Para usar el tuyo propio, cualquier archivo JSONL con un campo `"text"` por línea funciona directamente con `--corpus`.

**Épocas planificadas:**

| Época | Enfoque | Corpus principales |
|---|---|---|
| **Época 1** | Base lingüística | Wikipedia + desambiguación |
| **Época 2** | Razonamiento y lógica | GSM8K-ES, GSM-Hard, MCOT-Math, Aya-Reasoning, corpus de abducción. seq=1024, lr=5e-5 |
| **Época 3+** | Diálogo e instrucción | Alpaca-ES, Orca-ES, OpenSubtitles, Tatoeba, StackOverflow, corpus de identidad NOUS |

---

## Tokenizador

El tokenizador BPE está incluido en el repositorio bajo `tokenizador/`:

| Archivo | Tamaño | Descripción |
|---|---|---|
| `tokenizador_bpe_32k.model` | 748 KB | Modelo SentencePiece — requerido para entrenar |
| `tokenizador_bpe_32k.vocab` | 471 KB | Vocabulario legible |
| `tokenizer_hf.json` | 753 KB | Formato compatible con HuggingFace |
| `vocab_bpe_32k.json` | 627 KB | Vocab en JSON para herramientas Rust |

Entrenado en Wikipedia en español con BPE, 32.000 tokens, usando SentencePiece con codificación Metaspace.

---

## Formato de checkpoint

Formato v4 (magic bytes `ARK4`). Almacena pesos en FP16 nativo + momentos Adam m y v en FP32. Para 237M parámetros: **~2,2 GB por slot**.

Sistema rotativo de 3 slots. Siempre hay al menos dos copias válidas disponibles simultáneamente. Al reanudar, restaura pesos y estado completo del optimizador (271 tensores) para continuar con la inercia Adam acumulada intacta. Compatible con formatos anteriores v2 (pesos FP32) y v3 (pesos + momentos FP32).

Los checkpoints no se distribuyen con el repositorio por tamaño (~2,2 GB cada uno).

---

## Estructura del proyecto

```
proyecto_ark/
├── tokenizador/                     # Tokenizador BPE (incluido en repo)
│   ├── tokenizador_bpe_32k.model    (SentencePiece BPE 32k — requerido)
│   ├── tokenizador_bpe_32k.vocab    (vocabulario legible)
│   ├── tokenizer_hf.json            (formato HuggingFace)
│   └── vocab_bpe_32k.json           (vocab JSON para Rust)
├── entren/                          # Corpus y checkpoints (no en repo)
│   ├── wiki_esencial14.jsonl        (2,1 GB — debes aportarlo)
│   ├── wiki_disambig.jsonl          (37 MB — debes aportarlo)
│   └── ckpt_ark_ep1_rot*.bin        (~2,2 GB cada uno — generados por entrenamiento)
└── ark01/                           # Código fuente
    ├── src/
    │   ├── main.rs                  # Punto de entrada y parser de argumentos
    │   ├── config.rs                # Hiperparámetros y validación
    │   ├── training.rs              # Loop de entrenamiento, AMP, checkpointing
    │   ├── optimizer.rs             # AdamW Zero-Copy, clip global vDSP
    │   ├── memory.rs                # ModelWeights, AlignedVec, f16↔f32
    │   ├── io.rs                    # CorpusStream JSONL, CheckpointV4, BPE
    │   └── ffi.rs                   # Bindings seguros Rust↔ObjC↔ASM
    ├── objc/
    │   └── bridge.m                 # MPSGraph fwd/bwd AutoGrad, cross-entropy
    ├── asm/
    │   ├── kern.s                   # RMSNorm, softmax, SwiGLU, embed, dequant
    │   ├── opti.s                   # AdamW FP16/FP32, grad clip L2 (NEON)
    │   └── aten.metal               # Kernels Metal de atención
    ├── build.rs                     # Compila bridge.m + kern.s + opti.s
    └── Cargo.toml
```

---

## Compilar y ejecutar

**Requisitos:** macOS con Apple Silicon (M1/M2/M3/M4/M5), toolchain Rust, Xcode Command Line Tools.

> **Windows / Linux:** no soportado. ARK requiere frameworks exclusivos de macOS (Metal, MPSGraph, Accelerate). La dependencia `sentencepiece` también requiere `cmake` y solo está probada en macOS vía Homebrew (`brew install cmake sentencepiece`).

```bash
git clone https://github.com/IAsesoria/ARK.git
cd ARK/ark01
cargo build --release

caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial14.jsonl,../entren/wiki_disambig.jsonl \
  --vocab=../tokenizador/tokenizador_bpe_32k.model \
  --ckpt=../entren/ckpt_ark_ep1_rot0.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=128 --batch=2 --lr=5e-5 --clip=1.0 \
  --epochs=1

tail -f ../entren/ark_ep1.log
```

> `caffeinate -i` previene que macOS suspenda la CPU o la GPU durante sesiones largas de entrenamiento.

**Sobre los avisos ANE:** Al arrancar, MPSGraph intenta despachar algunas operaciones al Apple Neural Engine e imprime avisos `Incompatible element type for ANE`. Esto es normal — ARK no usa ANE, y el entrenamiento corre correctamente en GPU en todo momento. Silenciar estos avisos está en el roadmap.

---

## Pruebas en hardware — convocatoria abierta

ARK está desarrollado y probado exclusivamente en **M1 8GB**. Los valores de seq y batch que usa la época 1 están limitados por la memoria unificada disponible en esa máquina. Genuinamente no sabemos cómo se comporta ARK en Apple Silicon con más RAM — y ahí es donde puedes ayudar.

**Si tienes un Mac con Apple Silicon y quieres contribuir:** compila ARK, prueba correr con valores más grandes de `--seq` y `--batch`, y abre un Issue con la etiqueta `hardware-report` incluyendo:

- Chip y RAM
- Tiempo de compilación
- Tiempo por paso (s/step) y configuración usada
- El `--seq` / `--batch` más grande que corrió sin OOM
- Errores o comportamiento inesperado

Esto ayuda directamente a optimizar para hardware al que no tenemos acceso. Los Pull Requests con mejoras son igual de bienvenidos — ARK es un proyecto de aprendizaje y el aporte de la comunidad lo hace mejor para todos.

Chips de los que necesitamos reportes:
- M1 Pro / Max / Ultra
- M2 / M2 Pro / Max / Ultra
- M3 / M3 Pro / Max
- M4 / M4 Pro / Max / Ultra
- M5 / M5 Pro / Max

---

## Problemas conocidos y próximos pasos

Lista honesta de lo que aún no está hecho o no está pulido:

- **Modo de inferencia** — no implementado aún. El checkpoint existe; el binario de inferencia (generación token a token desde un checkpoint entrenado) es el próximo hito de desarrollo principal.
- **Avisos ANE** — `Incompatible element type for ANE` aparece al arrancar. No afecta el entrenamiento, pero ensucia el log. Será silenciado o filtrado.
- **Pasos hardcodeados** — el total de pasos estimados mostrado en el dashboard es actualmente un valor fijo. Se calculará dinámicamente desde el corpus al arrancar.
- **Estimación de tiempo restante** — no se muestra porque aún no hay suficiente promedio de velocidad por paso confiable. Se agregará cuando haya datos suficientes para promediar con sentido.
- **Historial de batch/seq en el dashboard** — los cambios de configuración en el tiempo aún no se reflejan en el gráfico de entrenamiento. Planificado para correlacionar cambios de config con la curva de loss.

Issues y pull requests son bienvenidos.

---

## Investigación activa

### Think-Anywhere — razonamiento bajo demanda durante la generación
*(Jiang et al., Peking University / Alibaba, marzo 2026 — arXiv:2603.29957)*

Demuestra que los LLMs concentran el razonamiento en el thinking previo a la respuesta, lo cual es ineficiente cuando la complejidad del problema solo se revela durante la generación. Think-Anywhere propone insertar bloques de razonamiento en posiciones de alta entropía durante la generación misma. Resultado: mejor rendimiento con menos tokens de razonamiento totales.

**Aplicación a EKO época 2:** construir ejemplos de entrenamiento donde el razonamiento emerge en posiciones de alta incertidumbre dentro de la respuesta, no solo en el thinking inicial. *(Experimental.)*

### EML — un operador para todas las funciones elementales
*(Odrzywolek, Jagiellonian University, abril 2026 — arXiv:2603.21852)*

`eml(x, y) = exp(x) − ln(y)` junto con la constante 1 genera la base completa de funciones elementales estándar. Análogo a la puerta NAND en electrónica digital.

**Aplicación a ARK:** regresión simbólica post-entrenamiento con árboles EML para descubrir qué función elemental exacta emergió en cada capa de EKO. Análisis de interpretabilidad alineado con el principio de emergencia sin declaración del Proyecto NOUS.

---

## Relación con el Proyecto NOUS

ARK es el motor de entrenamiento de EKO, la capa de verbalización y generación del **Proyecto NOUS** — un sistema de grafos emergentes, amplio y dinámico donde los conceptos se organizan en un hipergrafo por co-activación estadística, y las relaciones emergen de la física estadística del grafo sin reglas explícitas declaradas por el programador.

NOUS no tendrá reglas semánticas hardcodeadas — el significado emerge de las relaciones topológicas, de la física del hipergrafo. La estructura del hipergrafo es el próximo componente a construir.

**Objetivo a largo plazo:** NOUS proporciona conocimiento estructurado, ARK proporciona la capacidad de expresarlo. Comprensión y generación con raíces completamente distintas pero complementarias.

---

## Roadmap

- [x] Motor de entrenamiento Zero-Copy en Rust/ObjC/ASM
- [x] Tokenizador BPE 32k entrenado en español
- [x] Corpus Wikipedia en español (~617M tokens reales)
- [x] Checkpoints rotativos v4 con estado completo del optimizador
- [x] AMP dinámico estable (scale 256 → 8192, cero skips)
- [x] Kernels de ensamblador AArch64 v0.62 (4 bugs del backward corregidos)
- [x] AutoGrad nativo en GPU vía MPSGraph (bridge v1.3)
- [x] Cosine decay del LR (lr → lr/10 en 250k pasos)
- [ ] Modo de inferencia — generación token a token desde checkpoint
- [ ] Silenciar avisos ANE al arrancar
- [ ] Cálculo dinámico de pasos totales (sin hardcode)
- [ ] Estimación de tiempo restante basada en promedio real
- [ ] Historial de batch/seq superpuesto en el dashboard
- [ ] Completar época 1
- [ ] Época 2 — razonamiento matemático y abducción
- [ ] Épocas 3+ — instrucción y diálogo
- [ ] Evaluación en benchmarks en español (HellaSwag-ES, XCOPA-ES, XQuAD)

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

*El paso 1 registró loss=10.47. El valor teórico para distribución uniforme sobre 32.000 tokens es log(32.000) ≈ 10.37. La diferencia es la inicialización Xavier. Eso es todo — el modelo no sabe nada todavía. Lo que viene después es lo interesante.*
