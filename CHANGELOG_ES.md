# CHANGELOG — ARK Training Engine

Todos los cambios significativos, correcciones y decisiones tomadas durante el proyecto están documentados aquí en orden cronológico. Este es un registro honesto — incluyendo errores encontrados y corregidos.

---

## [2026-05-13 / 2026-05-15] — Corrección de corpus, expansión de vocabulario, aumento de contexto, reanudación del entrenamiento

**Paso global al inicio del evento:** ~300.000 (época 1)
**Paso global al cierre del evento:** ~312.300 (época 1, en curso)

### Contexto

Esta entrada cubre tres días consecutivos de trabajo correctivo desencadenado por la detección de tokens `<unk>` en la salida de inferencia aproximadamente en el paso 300.000. La secuencia de cambios no fue planificada como una operación única — se desarrolló progresivamente a medida que cada corrección revelaba el siguiente problema. La cronología está documentada con la mayor precisión posible.

---

### 2026-05-13 — Causa raíz identificada: corpus y vocabulario

#### Qué ocurrió

Las pruebas de inferencia en la máquina Ryzen usando el checkpoint activo producían tokens `<unk>` en la salida generada. La investigación reveló que el problema estaba en el corpus de entrenamiento, no en el modelo.

`wiki_esencial14.jsonl` no había sido completamente limpiado de caracteres no latinos (escrituras cirílica, CJK, árabe). Estos caracteres pasaron por el pipeline de filtrado sin ser detectados. Como el vocabulario BPE (`tokenizador_bpe_32k.model`) fue entrenado sobre español latino limpio, no tenía cobertura para estos caracteres — produciendo `<unk>` durante la inferencia e introduciendo ruido y dispersión en los embeddings aprendidos durante el entrenamiento.

Este fue un error de preparación del corpus que debería haberse detectado antes. No se detectó.

#### Acción 1 — Corpus reconstruido: wiki_esencial19.jsonl

El corpus de Wikipedia fue reconstruido desde cero con filtrado estricto de caracteres no latinos aplicado a nivel de carácter (`limpiar_wiki_v3.py`). El orden de serialización del campo `title` también fue corregido (anteriormente `{"text":..., "title":...}`, ahora consistentemente `{"title":..., "text":...}`).

| Métrica                    | Valor                                       |
|--------------------------- |-------------------------------------------- |
| Artículos                  | 340.275                                     |
| Tamaño en disco            | 2,1 GB                                      |
| Tokens estimados           | ~518M                                       |
| Tokens promedio por doc    | ~1.524 (muestreado dinámicamente al inicio) |
| Errores JSON               | 0                                           |
| Caracteres fuera del vocab | solo U+0020 (espacio — esperado)            |

`wiki_esencial14.jsonl` fue eliminado para recuperar espacio en disco.

#### Acción 2 — Vocabulario expandido: tokenizador_bpe_32k_v2.model

El vocabulario BPE fue expandido de 32.000 a 32.063 tokens para cubrir caracteres de alta frecuencia que causaban `<unk>` o que estaban ausentes del vocabulario base. La expansión se realizó añadiendo tokens `USER_DEFINED` al modelo SentencePiece — sin reentrenar los merges BPE.

|          | Antes                       | Después                        |
|--------- |---------------------------- |------------------------------- |
| Archivo  | `tokenizador_bpe_32k.model` | `tokenizador_bpe_32k_v2.model` |
| Tokens   | 32.000                      | 32.063                         |
| Añadidos | —                           | +63 caracteres (ver abajo)     |

**Los 63 caracteres añadidos (por categoría):**

| Categoría                   | Caracteres                                          | Cantidad |
|---------------------------- |---------------------------------------------------- |--------- |
| Griegas minúsculas          | α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω | 25       |
| Griegas mayúsculas          | Γ Δ Θ Λ Σ Φ Χ Ψ Ω Π                                | 10       |
| Superíndices                | ² ³ ¹ ⁴ ⁰                                          | 5        |
| Subíndices                  | ₀ ₁ ₂                                              | 3        |
| Operadores matemáticos      | ° × √ ≈ ∫ → ± ·                                    | 8        |
| Fracción / moneda / ordinal | ½ € º ª                                            | 4        |
| ASCII faltantes             | & # \ ~ ^ @ ` ÷                                    | 8        |
| **Total**                   |                                                     | **63**   |

**Omisión conocida — `ü` / `Ü`:**

Los caracteres de diéresis española `ü` (minúscula) y `Ü` (mayúscula) no fueron incluidos en esta expansión. Fue un descuido. El script de limpieza `limpiar_wiki_v3.py` contenía la transliteración `'ü': 'u', 'Ü': 'u'`, que convirtió silenciosamente todas las ocurrencias antes de que se realizara la auditoría del vocabulario. Como resultado, el corpus contiene "pinguino" en lugar de "pingüino", "verguenza" en lugar de "vergüenza", y aproximadamente 250 palabras afectadas de baja frecuencia en texto enciclopédico.

El impacto en Época 1 se considera marginal. La corrección está programada antes de Época 2, que contendrá corpus de razonamiento con español correctamente escrito. El proceso de corrección completo está documentado en:

> `tokenizador_bpe/gestion_vocabulario_bpe.md`

El proceso técnico completo de esta expansión — incluyendo el script Python utilizado, los pasos de verificación y el procedimiento general para agregar caracteres en versiones futuras — está documentado en:

> `tokenizador_bpe/vocabulary_bpe_management.md`

#### Acción 3 — Checkpoint expandido para coincidir con el nuevo vocabulario

Los tensores de embedding fueron expandidos de `[32000, 768]` a `[32063, 768]` para coincidir con el nuevo tamaño del vocabulario. Los 3 tensores afectados (`embed_w` FP16, `embed_m` FP32, `embed_v` FP32) fueron expandidos usando la misma estrategia de inicialización por promedio de vecinos documentada en `gestion_vocabulario_bpe.md`. Los 810 tensores de capas restantes se preservaron sin cambios.

#### Acción 4 — Contexto aumentado: seq 128/512 → 1024

El mismo día, la longitud de contexto de entrenamiento fue aumentada en dos etapas:

- Primera reanudación: `seq=512 batch=1` (~4.000 pasos, desde el paso global ~300.500)
- Segunda reanudación: `seq=1024 batch=1` (desde el paso global ~304.000 en adelante)

Motivación: `seq=128` y `seq=512` eran compromisos de velocidad que habían alcanzado su meseta de aprendizaje. Un contexto más largo permite al modelo aprender dependencias de mayor alcance en texto enciclopédico español. `batch=1` es la única configuración que permite procesar `seq=1024` a través de 30 capas transformer dentro de los 8GB de memoria unificada del MacBook Air M1 sin recurrir a swap.

El conteo de pasos del corpus en `seq=1024` es ~506.424 pasos por época (340.275 docs × ~1.524 tokens/doc ÷ 1.024 tokens/paso).

#### Reanudación del entrenamiento — confirmada limpia

```
[checkpoint v4] step=304000 | adam=304000 | layers=30
[ep 1  paso    1  g  304001]  loss=3.7324  ppl=41.8  scale=256  skips=0
```

Loss de 3,73 en el primer paso reanudado en seq=1024 confirma la integridad del checkpoint y la compatibilidad del vocabulario.

**Comando activo desde este punto:**

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

### 2026-05-13 al 2026-05-15 — Estabilización AMP en seq=1024

El escalador de pérdida AMP progresó por sus fases normalmente durante los primeros ~8.300 pasos locales en seq=1024 (pasos globales ~304.001–312.300):

| Pasos locales | Pasos globales  | Scale AMP     |
|-------------- |---------------- |-------------- |
| 1–1.999       | 304.001–305.999 | 256           |
| 2.000–3.999   | 306.000–307.999 | 512           |
| 4.000–5.999   | 308.000–309.999 | 1.024         |
| 6.000–7.999   | 310.000–311.999 | 2.048         |
| 8.000+        | 312.000+        | 4.096 → 8.192 |

Sin eventos NaN/Inf, sin pasos omitidos. AMP avanza hacia la fase estable de `scale=8192`.

---

### Qué significa este evento para el historial de entrenamiento

El contador de pasos (`g`) es acumulativo y continuo a través de todos los cambios. El flujo del corpus se reinició desde el principio con `wiki_esencial19.jsonl` en el paso ~300.500. Los ~300.000 pasos entrenados en `wiki_esencial14.jsonl` no se pierden — el modelo aprendió estructura real del idioma español — pero el ruido de embedding de los caracteres `<unk>` será progresivamente sobreescrito por el corpus limpio. Como resultado, completar Época 1 requiere alcanzar aproximadamente el paso global ~806.000 (300.500 de reinicio + 506.424 pasos para cubrir el corpus completo una vez en seq=1024). Los pasos anteriores contribuyeron aprendizaje real del lenguaje pero no cuentan hacia la cobertura del corpus de la época actual.

---

### Línea base de rendimiento en la reanudación a seq=1024 (paso global ~304.000)

| Ventana           | Loss promedio | PPL promedio |
|------------------ |-------------- |------------- |
| Últimos 10k pasos | 4,032         | ~56          |
| Últimos 50k pasos | 3,877         | ~48          |

Estas sirven como línea base para medir la mejora durante los próximos 50k–100k pasos en seq=1024.

---

## [2026-05-13] — Corrección de corpus, expansión de vocabulario, aumento de contexto

**Paso al momento del descubrimiento:** ~300.000 (época 1, fase seq=128/batch=2)

### Qué ocurrió

Aproximadamente en el paso 300.000, se ejecutaron pruebas de inferencia en la máquina Ryzen usando el checkpoint activo. La salida contenía tokens `<unk>` — el modelo producía marcadores de token desconocido durante la generación.

### Causa raíz

La investigación reveló que el problema no estaba en el modelo, sino en el corpus de entrenamiento. `wiki_esencial14.jsonl` no había sido completamente limpiado de caracteres no latinos (escrituras cirílica, CJK, árabe). Estos caracteres pasaron por el pipeline de filtrado sin ser detectados. Como el vocabulario BPE (`tokenizador_bpe_32k.model`) fue entrenado sobre español latino limpio, no tenía cobertura para estos caracteres — produciendo `<unk>` durante la inferencia e introduciendo ruido y dispersión en los embeddings aprendidos durante el entrenamiento.

Este fue un error de preparación del corpus que debería haberse detectado antes. No se detectó.

### Acciones tomadas

**1. Corpus regenerado — wiki_esencial19.jsonl**

El corpus de Wikipedia fue reconstruido con filtrado estricto de caracteres no latinos aplicado a nivel de carácter. El orden de serialización del campo `title` también fue corregido (anteriormente `{"text":..., "title":...}`, ahora consistentemente `{"title":..., "text":...}`).

| Métrica                    | Valor                                       |
|--------------------------- |-------------------------------------------- |
| Artículos                  | 340.275                                     |
| Tamaño                     | 2,1 GB                                      |
| Errores JSON               | 0                                           |
| Artículos sin `title`      | 0                                           |
| Artículos sin `text`       | 0                                           |
| Caracteres fuera del vocab | solo U+0020 (espacio — esperado y correcto) |

`wiki_esencial14.jsonl` fue eliminado para recuperar espacio en disco en la máquina M1 8GB.

**2. Vocabulario expandido — tokenizador_bpe_32k_v2.model**

El vocabulario BPE fue expandido para cubrir los caracteres que causaban `<unk>`. Los tensores de embedding y el checkpoint fueron expandidos en consecuencia.

|         | Antes                       | Después                        |
|-------- |---------------------------- |------------------------------- |
| Tokens  | 32.000                      | 32.063                         |
| Archivo | `tokenizador_bpe_32k.model` | `tokenizador_bpe_32k_v2.model` |

**3. Longitud de contexto aumentada — seq 128 → 512 → 1024**

El entrenamiento fue reanudado con `seq=512 batch=1` por aproximadamente 4.000 pasos, luego aumentado a `seq=1024 batch=1`. Este cambio fue realizado al mismo tiempo que la corrección del corpus. Motivación: un contexto más largo permite al modelo aprender dependencias de mayor alcance; el seq=128 anterior era un compromiso de velocidad que había alcanzado su meseta.

**4. Entrenamiento reanudado desde el paso 300.500**

El checkpoint `ckpt_ark_ep1_rot1_expanded.bin` fue cargado con el estado completo del optimizador Adam (271 tensores). Reanudación confirmada limpia:

```
[checkpoint v4] step=300500 | adam=300500 | layers=30
[ep 1  paso  1  g 300501]  loss=3.1333  ppl=22.9  scale=256  skips=0
```

Loss de 3,13 en el primer paso reanudado confirma que el checkpoint y el vocabulario expandido son compatibles.

**Comando activo desde este punto:**

```bash
nohup caffeinate -i ./target/release/ark \
  --corpus=../entren/wiki_esencial19.jsonl \
  --vocab=../entren/tokenizador_bpe_32k_v2.model \
  --ckpt=../entren/ckpt_ark_ep1_rot1_expanded.bin \
  --layers=30 --heads=12 --d-model=768 --hidden=2048 \
  --seq=512 --batch=1 --lr=5e-5 --clip=0.5 \
  --epochs=1 >> ../entren/ark_ep1_seq512.log 2>&1 &
```

### Qué significa para el historial de entrenamiento

El contador de pasos (`g`) continúa desde 300.501 y es acumulativo. Sin embargo, el flujo del corpus se reinició desde el principio con el nuevo archivo. Los ~300.000 pasos entrenados en `wiki_esencial14.jsonl` no se pierden — el modelo aprendió estructura real del idioma español — pero el ruido de embedding introducido por los caracteres `<unk>` necesitará ser sobreescrito por el corpus limpio. Pasos estimados para cubrir el corpus completo una vez en seq=512: ~1.012.849. El contexto fue posteriormente aumentado a seq=1024, reduciendo eso a ~506.424 pasos para completar Época 1 — ver entrada [2026-05-13 / 2026-05-15] más arriba. El contador de pasos global marcará aproximadamente ~806.000 cuando Época 1 finalice (300.500 punto de reinicio + 506.424 pasos del corpus).

### Línea base de rendimiento en la reanudación (promedios de loss sobre los últimos N pasos)

| Ventana             | Loss promedio | PPL promedio |
|-------------------- |-------------- |------------- |
| Últimos 10k pasos   | 3,871         | 55,00        |
| Últimos 50k pasos   | 3,854         | 54,71        |
| Últimos 100k pasos  | 3,853         | 54,60        |

Estas sirven como línea base para medir si la corrección del corpus y el aumento de contexto producen una mejora medible durante los próximos 10k–50k pasos.

### Validación pendiente

La inferencia en la máquina Ryzen con el primer checkpoint estable después del paso ~310.000 confirmará si los tokens `<unk>` han sido eliminados de la salida de generación.

---

## Pendiente — Antes de Época 2

Todos los procedimientos de expansión (tokenizador y checkpoint) están cubiertos paso a paso en `tokenizador_bpe/gestion_vocabulario_bpe.md` y `tokenizador_bpe/expand_checkpoint.js`.

Las siguientes acciones son necesarias antes de iniciar el entrenamiento de Época 2:

| Acción                   | Descripción                                                             | Referencia                                   |
|------------------------- |------------------------------------------------------------------------ |--------------------------------------------- |
| Tokenizador v3           | Agregar `ü` / `Ü` → 32.065 tokens                                       | `tokenizador_bpe/gestion_vocabulario_bpe.md` |
| Expansión del checkpoint | Expandir embedding `[32063,768]` → `[32065,768]`                        | `expand_checkpoint.js`                       |
| Actualizar config.rs     | `VOCAB_SIZE: 32063` → `32065`                                           | `src/config.rs`                              |
| Auditoría del corpus     | Verificar que los corpus de Época 2 contengan `ü` correctamente escrita | `grep -l 'ü' *.jsonl`                        |
| Actualizar README        | Actualizar sección del tokenizador para reflejar v3                     | `README_ES.md` / `README.md`                 |

---

## [2026-05-17] — Inferenciador Ryzen: migración vocab v1→v2, primeras inferencias sin <unk>

**Paso global de referencia:** ~321.000 (época 1, en curso)

### Contexto

Durante las pruebas de inferencia del 13 de mayo (paso ~297.000) con el vocabulario
original v1 (`vocab_sp.json`, 32.000 tokens), los resultados mostraban tokens `<unk>`
frecuentes — evidencia directa del problema de cobertura documentado en la entrada
anterior. Esas pruebas quedan registradas en:

> `inferencias/EKO_paso297k_130526.txt`

El 17 de mayo se completó la migración del inferenciador en Ryzen para trabajar con
el vocabulario expandido v2, y se registraron las primeras inferencias limpias:

> `inferencias/EKO_paso321k_170526.txt`

---

### Problema detectado

`eko_infer` en Windows carga el vocabulario desde JSON plano (`vocab_sp.json` +
`vocab_scores.json`), no desde el archivo `.model` de SentencePiece directamente.
Al expandir el tokenizador de 32.000 a 32.063 tokens, los JSONs del Ryzen quedaron
desactualizados. El checkpoint activo (paso 321.000) fue entrenado con v2 pero el
inferenciador seguía cargando v1 — produciendo un mismatch silencioso de vocab_size.

---

### Acciones

**1. Tokenizador v2 copiado al Ryzen desde el Mac**

```bash
scp usuario@xxx.xxx.x.xxx:/Users/benjamin/Documents/ark/rust/entren/tokenizador_bpe_32k_v2.model D:\proyecto-nwin\llm\eko\
```

**2. JSONs v2 generados en Ryzen con Python**

Sin interrumpir el entrenamiento en el Mac. Se instaló sentencepiece en Windows
y se exportaron los JSONs directamente desde el `.model`:

```powershell
python -m pip install sentencepiece
python exportar_vocab_v2.py tokenizador_bpe_32k_v2.model
```

Salida confirmada:

**3. Comando de inferencia activo**

```powershell
cargo run --release --bin eko_infer -- `
  --ckpt ..\ckpt_ark_ep1_rot017mayo.bin `
  --vocab ..\vocab_sp_v2.json `
  --scores ..\vocab_scores_v2.json `
  --vocab-size 32063 `
  --prompt "El agua es"
```

**4. Archivos subidos al repositorio**

Carpeta `tokenizador_bpe/` — commit `ece67bb`:

| Archivo                | Descripción                                         |
|------------------------|-----------------------------------------------------|
| `vocab_sp_v2.json`     | Vocabulario v2 en formato JSON (32.063 tokens)      |
| `vocab_scores_v2.json` | Scores Viterbi v2 en formato JSON (32.063 tokens)   |
| `exportar_vocab_v2.py` | Script para regenerar los JSONs desde cualquier `.model` |

Carpeta `inferencias/` — commit `a877ff4`:

| Archivo                      | Descripción                                    |
|------------------------------|------------------------------------------------|
| `EKO_paso297k_130526.txt`    | Inferencias 13-may, paso ~297.000, vocab v1    |
| `EKO_paso321k_170526.txt`    | Inferencias 17-may, paso ~321.000, vocab v2    |

---

### Resultados — comparación directa

| Métrica                  | 13-may paso ~297.000 vocab v1 | 17-may paso ~321.000 vocab v2 |
|--------------------------|-------------------------------|-------------------------------|
| Tokens `<unk>`           | Frecuentes                    | Ninguno                       |
| Velocidad inferencia     | ~4 tok/s                      | ~11 tok/s                     |
| Coherencia geográfica    | Baja                          | Media                         |
| Corpus activo            | wiki_esencial14 (con ruido)   | wiki_esencial19 (limpio)      |

La mejora de velocidad (~3×) se debe a la transposición de pesos implementada
en `eko_infer`, no al cambio de vocabulario.

**Observaciones sobre la calidad a paso 321.000:**

- Cero tokens `<unk>` en toda la salida
- Español gramaticalmente coherente a nivel de frase corta
- Atractor temático dominante: etimología / toponimia / lenguas — consistente
  con el contenido de Wikipedia letra A (~10% del corpus procesado a seq=1024)
- Prompts geográficos producen las salidas más coherentes en esta etapa

---

## [2026-05-22] — Decaimiento manual de tasa de aprendizaje, estabilización de longitud de secuencia, diagnóstico de swap

**Paso global del evento:** ~343,500 (época 1)

### Qué sucedió

Se observó una tendencia alcista constante en la pérdida (loss) promedio y la perplejidad (PPL) durante 5 días consecutivos:
- Punto mínimo: Loss `3.491` | PPL `34.25` (aprox. paso 320,000)
- Punto máximo: Loss `3.841` | PPL `54.04` (aprox. paso 343,500)

Aunque el modelo ocasionalmente experimentó picos locales debido a bloques de datos densos y no lingüísticos (tablas y listas de Wikipedia), el promedio diario general no logró retornar a la base. Esto se diagnosticó como **inestabilidad por falta de decaimiento (divergencia lenta)**. Una tasa de aprendizaje constante de `5e-5` resultó demasiado agresiva para la fase de ajuste fino de la ventana de contexto de 1024, impidiendo que los pesos se asentaran en el mínimo local.

Se intentó realizar una prueba experimental a `seq=2048`, pero fue abortada rápidamente. El diagnóstico a través de `htop` mostró que la memoria swap activa subió a 1.87 GB, reduciendo la utilización de la CPU al 18.1% (saturación severa de disco/thrashing). Esto confirmó que `seq=1024` es el límite físico absoluto para el entrenamiento en una máquina M1 de 8 GB.

### Acciones tomadas

**1. Proceso finalizado de forma segura**
El proceso de entrenamiento fue detenido en el paso global 343,600. El checkpoint `ckpt_ark_ep1_rot0.bin` (guardado a las 07:48 AM, correspondiente al paso 343,500) fue verificado como el estado limpio más reciente.

**2. Decaimiento manual de la tasa de aprendizaje aplicado (5e-5 → 2e-5)**
Para estabilizar la convergencia y evitar la oscilación de los pesos, la tasa de aprendizaje se redujo en un 60%, de `5e-5` a `2e-5` (0.00002). El recorte de gradiente (gradient clipping) se mantuvo en `--clip=0.5` para proteger al modelo de la varianza restante de los gradientes.

**3. Reanudación del entrenamiento a seq=1024**
El proceso se reinició utilizando el checkpoint limpio más reciente con los parámetros actualizados.

**Comando activo a partir de este punto:**

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

### 4. Optimización de la interfaz del panel (Dashboard)

Junto con el ajuste del motor principal, se resolvieron dos fallos heredados en la interfaz de visualización del entrenamiento:

- **Corrección del fallo de renderizado del Canvas:** Se modificó `drawChart()` para leer los valores de dimensión directamente desde `canvas.clientWidth/clientHeight` en lugar de analizar el string `canvas.style.width`. Esto resolvió un fallo de maquetación donde las líneas del gráfico no se renderizaban a menos que se redujera el zoom de la ventana del navegador.
- **Actualización tolerante a fallos de las tarjetas KPI:** Se ajustó `updateDashboardCards()` para evitar colapsos en la ejecución de JavaScript. Anteriormente, la eliminación de la tarjeta KPI "Loss Mínimo" (no utilizada) del HTML causaba un error de puntero nulo (null pointer exception) fatal cuando el script de actualización intentaba escribir en su elemento inexistente, deteniendo silenciosamente las actualizaciones posteriores de la interfaz (como la píldora dinámica de AMP Scale). El script y la maquetación HTML han sido limpiados y desacoplados.

*ARK es desarrollado por Benjamín Alonso Carmona Vega / IAsesoria Informática, Villarrica, Chile.*
*Desarrollo asistido por Claude Sonnet (Anthropic) y Gemini Pro (Google).*
