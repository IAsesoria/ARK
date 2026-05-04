# ARK — Motor de Entrenamiento LLM desde Cero
### EKO · Proyecto NOUS · IAsesoria Informática · Villarrica, Chile · 2026

> **Entrenando ahora mismo.** Época 1, paso ~14.700 de ~1.206.463. Loss: 10.47 → 3.61. Sin PyTorch. Sin TensorFlow. Sin GPU en la nube.

---

## Qué es esto

ARK es un motor de entrenamiento de modelos de lenguaje grande escrito íntegramente desde cero en **Rust, Objective-C y ensamblador NEON AArch64**. No depende de PyTorch, TensorFlow ni de ningún framework de deep learning. Cada capa del stack de cómputo — forward en GPU, backward en GPU vía MPSGraph AutoGrad, optimizador en ensamblador, kernels matemáticos propios — está escrita para ser lo más optimizada posible para Apple Silicon, concretamente para M1.

El primer modelo que ARK entrena se llama **EKO**, parte del **Proyecto NOUS**. EKO es un transformer de 237 millones de parámetros entrenado principalmente con corpus en español, variados por ejemplo contenido enciclopédico, matemático y conversacional, todo construido-depurado específicamente para este proyecto.

Lo que pretendo es crear una IA desde cero, con hardware de consumo, en español, desde Chile, yendo a contra corriente al no usar framework tipicos, y además en este proceso aprender, calmar la curiosidad que dio inicio meses atrás.

---

## Por qué importa

La mayoría de los modelos de lenguaje se entrenan en clústeres de GPU NVIDIA que cuestan decenas o cientos de miles de dólares. ARK busca tres cosas:

**1. Eficiencia real en hardware limitado.**
Un MacBook Air M1 de 8GB es hardware de consumo masivo. ARK lo aprovecha al máximo: trata de usar arquitectura Zero-Copy sobre memoria unificada, forward en GPU vía MPSGraph con grafos compilados AOT, backward nativo en GPU vía AutoGrad con `gradientForPrimaryTensor:withTensors:`, optimizador AdamW en ensamblador SIMD. Sin copias innecesarias entre CPU y GPU. Sin overhead de frameworks.

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
- 30 capas transformer(Eko): RMSNorm → atención SDPA multi-cabeza → FFN SwiGLU → RMSNorm
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
| Contexto (época 1) | 512 tokens |
| Contexto (época 2+) | 1024 tokens |
| Vocabulario | BPE 32.000 tokens (SentencePiece, español) |
| Codificación posicional | RoPE |
| Normalización | RMSNorm (gamma excluido de weight decay) |
| Precisión | AMP: pesos FP16 / gradientes y momentos Adam FP32 |

### Configuración de entrenamiento activa

```
Época 1 (base lingüística):
  --corpus=wiki_esencial14.jsonl,wiki_disambig.jsonl
  --layers=30 --heads=12 --d-model=768 --hidden=2048
  --seq=512 --batch=1 --lr=1e-4 --warmup=2000 --clip=0.5
  --epochs=1

AdamW: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
AMP:   loss_scale_init=256, max=8192, step_up_every=2000
Checkpoint: rotativo de 3 slots, cada 500 pasos
```

---

## Precisión mixta automática (AMP)

Los pesos se almacenan en FP16 en VRAM. Los gradientes y momentos Adam (m, v) se mantienen en FP32. El scaler dinámico arranca en 256 y puede subir hasta 8192 en pasos de ×2 cada 2000 pasos limpios. Si aparece NaN/Inf, el paso se descarta, el scale se divide a la mitad, y el entrenamiento continúa desde el checkpoint intacto sin corrupción del estado del optimizador.

---

## Corpus y curriculum de entrenamiento

Corpus de época 1: **~617M tokens reales** (calculados muestreando 1.000 documentos al arrancar — sin estimaciones hardcodeadas):

| Corpus | Contenido |
|---|---|
| `wiki_esencial14.jsonl` | Wikipedia en español — 341.147 documentos filtrados (2,1 GB) |
| `wiki_disambig.jsonl` | Páginas de desambiguación de Wikipedia — 63.113 documentos |

**Épocas planificadas:**

- **Época 1** — Base lingüística. Wikipedia + desambiguación. ~1.206.463 pasos a seq=512.
- **Época 2** — Razonamiento y lógica. GSM8K-ES, GSM-Hard, MCOT-Math, Aya-Reasoning, corpus de abducción, thinking multilingüe. seq=1024, lr=5e-5. (Sujeta a cambios).
- **Época 3+** — Diálogo e instrucción. Alpaca-ES, Somos Alpaca, Orca-ES, conversación natural, OpenSubtitles, Tatoeba, StackOverflow, lenguaje claro, corpus de identidad NOUS. (Sujeta a cambios).

---

## Formato de checkpoint

Formato v4 (magic bytes `ARK4`). Almacena pesos en FP16 nativo + momentos Adam m y v en FP32. Para 237M parámetros: **2.369,6 MB por slot**.

Sistema rotativo de 3 slots. Siempre hay al menos dos copias válidas disponibles simultáneamente. Al reanudar, restaura pesos y estado completo del optimizador (271 tensores) para continuar con la inercia Adam acumulada intacta. Compatible con formatos anteriores v2 (pesos FP32) y v3 (pesos + momentos FP32).

---

## Estructura del proyecto

```
ark/
└── rust/
    ├── entren/                          # Corpus y artefactos
    │   ├── wiki_esencial14.jsonl        (2,1 GB — 341.147 docs)
    │   ├── wiki_disambig.jsonl          (37 MB — 63.113 docs)
    │   ├── tokenizador_bpe_32k.model    (SentencePiece BPE 32k)
    │   ├── ckpt_ark_ep1_rot*.bin        (checkpoints rotativos, ~2,2 GB cada uno)
    │   └── [corpus época 2-3+]          (razonamiento, instrucción, diálogo)
    └── ark050/                          # Código fuente
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
        └── cargo.toml
```

---

## Compilar y ejecutar

**Requisitos:** macOS con Apple Silicon (M1/M2/M3), toolchain Rust, Xcode Command Line Tools.

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

`caffeinate -i` previene que macOS suspenda la CPU o la GPU durante sesiones largas de entrenamiento.

---

## Pruebas en hardware — convocatoria abierta

ARK está desarrollado y probado en M1 8GB. **Necesitamos reportes de:**

- M1 Pro / Max / Ultra
- M2 / M2 Pro / Max / Ultra
- M3 / M3 Pro / Max

Si tienes un Mac con Apple Silicon y quieres contribuir: compila ARK, ejecuta un paso de entrenamiento, y abre un Issue con la etiqueta `hardware-report` incluyendo tu chip, RAM, tiempo de compilación, tiempo por paso (s/step) y uso de memoria. Esto ayuda directamente a optimizar para hardware al que no tenemos acceso.

---

## Investigación activa

### Think-Anywhere — razonamiento bajo demanda durante la generación
*(Jiang et al., Peking University / Alibaba, marzo 2026 — arXiv:2603.29957)*

Demuestra que los LLMs concentran el razonamiento en el thinking previo a la respuesta, lo cual es ineficiente cuando la complejidad del problema solo se revela durante la generación. Think-Anywhere propone insertar bloques de razonamiento en posiciones de alta entropía durante la generación misma. Resultado: mejor rendimiento con menos tokens de razonamiento totales.

Aplicación a EKO época 2: construir ejemplos de entrenamiento donde el razonamiento emerge en posiciones de alta incertidumbre dentro de la respuesta, no solo en el thinking inicial.

### EML — un operador para todas las funciones elementales
*(Odrzywolek, Jagiellonian University, abril 2026 — arXiv:2603.21852)*

`eml(x, y) = exp(x) − ln(y)` junto con la constante 1 genera la base completa de funciones elementales estándar. Análogo a la puerta NAND en electrónica digital. Las identidades clave están verificadas en código.

Aplicación a ARK: regresión simbólica post-entrenamiento con árboles EML para descubrir qué función elemental exacta emergió en cada capa de EKO. Análisis de interpretabilidad alineado con el principio de emergencia sin declaración del Proyecto NOUS.

---

## Relación con el Proyecto NOUS

ARK es el motor de entrenamiento de EKO, la capa de verbalización y generación del Proyecto NOUS — un sistema de cognición emergente más amplio donde los conceptos se organizan en un hipergrafo cognitivo por co-activación estadística, y las relaciones emergen de la física estadística del grafo sin reglas declaradas por el programador.

La coherencia filosófica entre NOUS y ARK es completa: ninguno define comportamientos lingüísticos explícitamente. NOUS no tiene reglas semánticas hardcodeadas — las relaciones emergen de la física del hipergrafo. ARK no tiene heurísticas de lenguaje — el modelo aprende distribuciones sobre tokens puramente de los datos. El significado emerge de la estructura, en ambos sistemas.

Objetivo a largo plazo: NOUS proporciona conocimiento estructurado, ARK proporciona la capacidad de expresarlo. Comprensión y generación con raíces completamente distintas pero complementarias, ambas sin intervención declarativa del programador.

---

## Roadmap

- [x] Motor de entrenamiento Zero-Copy en Rust/ObjC/ASM
- [x] Tokenizador BPE 32k entrenado en español
- [x] Corpus Wikipedia en español (~617M tokens reales)
- [x] Checkpoints rotativos v4 con estado completo del optimizador
- [x] AMP dinámico estable (scale 256 → 8192, cero skips)
- [x] Kernels de ensamblador AArch64 v0.62 (4 bugs del backward corregidos)
- [x] AutoGrad nativo en GPU vía MPSGraph (bridge v1.3)
- [ ] Completar época 1 (~1.206.463 pasos, estimado ~14 días totales desde inicio)
- [ ] Modo de inferencia (generación token a token desde checkpoint)
- [ ] Época 2 — razonamiento matemático y abducción
- [ ] Épocas 3+ — instrucción y diálogo
- [ ] Evaluación en benchmarks en español (HellaSwag-ES, XCOPA-ES, XQuAD)
- [ ] Escalar a 1B parámetros

---

## Patrocinio

ARK y el Proyecto NOUS son investigación independiente financiada de forma privada desde Villarrica, Chile. El entrenamiento corre en hardware propio sin costo de nube.

Si eres una institución, empresa o persona interesada en IA eficiente en hardware de consumo, modelos soberanos en español, o sistemas cognitivos emergentes sin dependencia de grandes proveedores — el apoyo es bienvenido vía **GitHub Sponsors**.

### Tiers de patrocinio

| Tier | Monto | Qué incluye |
|---|---|---|
| **Seguidor** | $2/mes | Actualización mensual del progreso del entrenamiento — curva de loss, pasos completados, hitos alcanzados. Nombre en la lista de sponsors del README. |
| **Colaborador** | $8/mes | Todo lo anterior + reporte mensual detallado: análisis de loss, estabilidad de gradientes, cobertura del corpus, comportamiento del AMP. |
| **Contribuidor** | $25/mes | Todo lo anterior + acceso a checkpoints intermedios a medida que sean evaluables. Mención en publicaciones técnicas del proyecto. |
| **Técnico** | $75/mes | Todo lo anterior + asistencia técnica si estás construyendo algo sobre ARK o EKO. Respuesta prioritaria a issues. Orientación para adaptar el código a casos de uso específicos. |
| **Institucional** | $250/mes | Todo lo anterior + consideración de co-autoría en publicaciones de investigación. Adaptación personalizada de ARK o EKO para el caso de uso de tu organización. Acceso a documentación técnica completa y decisiones de arquitectura. |

Todo el apoyo va directamente a hardware (almacenamiento para corpus y checkpoints, Apple Silicon de mayor capacidad para entrenamiento más rápido) y tiempo de investigación.

---

## Sobre el proyecto

Desarrollado por **Benjamín Alonso Carmona Vega**, fundador de IAsesoria Informática, con colaboración de **Sonia**.

Villarrica, Chile · 2026

---

*El paso 1 registró loss=10.47. El valor teórico para distribución uniforme sobre 32.000 tokens es log(32.000)≈10.37. La diferencia es la inicialización Xavier. Eso es todo — el modelo no sabe nada todavía. Lo que viene después es lo interesante.*
