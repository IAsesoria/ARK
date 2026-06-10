// src/io.rs — ARK v1.0 "METAL-REASONER"
//
// I/O completamente en Rust optimizado para Zero-Copy y Modelos Razonadores.
// Tokenización: BPE real vía SentencePiece (e.g., tokenizador_bpe_32k.model).
// Corpus: CorpusStream lee archivos JSONL/TXT directamente on-the-fly, sin archivos .bin intermediarios.
//
// Mejoras v1.0 (Checkpoint V4):
//   - Almacena los pesos nativamente en FP16 (reduciendo el almacenamiento a la mitad).
//   - Mantiene los momentos del optimizador Adam (m, v) en FP32 para máxima precisión.
//   - Compatibilidad retroactiva transparente para cargar checkpoints V2 y V3.
//
// Formato binario v4 (little-endian):
//   [4 bytes]  MAGIC        = 0x4B_34_52_41  ("ARK4")
//   [8 bytes]  global_step  u64
//   [8 bytes]  adam_step    u64
//[8 bytes]  n_tensors    u64
//   Por cada tensor:
//     [8 bytes]  n_elementos  u64  (cantidad de elementos, no de bytes)
//[8 bytes]  tipo         u64  (0 = F16, 1 = F32)
//[n * tipo_size bytes] datos (alineados)
//
// Orden de escritura: 
//   embed_w(F16), embed_m(F32), embed_v(F32)
//   luego por capa: 9×pesos(F16) + 9×m(F32) + 9×v(F32)

use std::fs::File;
use std::io::{BufRead, Read, Seek, Write, BufReader, BufWriter};
use std::path::Path;
use anyhow::{Context, Result};
use sentencepiece::SentencePieceProcessor;

// ── Tokenizador BPE ───────────────────────────────────────────────

pub struct Tokenizador {
    spp: SentencePieceProcessor,
}

impl Tokenizador {
    pub fn cargar(path: &str) -> Result<Self> {
        let spp = SentencePieceProcessor::open(path)
            .with_context(|| format!("[tokenizador] no se pudo cargar el modelo BPE: {}", path))?;
        println!("[tokenizador] Modelo BPE cargado correctamente: {}", path);
        Ok(Self { spp })
    }

    pub fn encode(&self, texto: &str) -> Vec<u32> {
        // En v0.13 encode devuelve Result<Vec<common::SentencePiece>> o similar
        // Usamos bos/eos según necesites, aquí extraemos los IDs:
        match self.spp.encode(texto) {
            Ok(pieces) => pieces.iter().map(|p| p.id).collect(),
            Err(_) => Vec::new(),
        }
    }
}

// ── Limpieza de texto ─────────────────────────────────────────────

const TOKENS_ESPECIALES: &[(&str, &str)] = &[
    ("<USR>",        "\n"),
    ("<SYS>",        "\n"),
    ("<FIN>",        "\n"),
    ("<ASST>",       "\n"),
    ("<BOT>",        "\n"),
    ("<EOT>",        "\n"),
    ("<thinking>",   ""),
    ("</thinking>",  "\n"),
    ("<reflexión>",  ""),
    ("</reflexión>", "\n"),
    ("<reflection>", ""),
    ("</reflection>","\n"),
    ("<ajuste>",     ""),
    ("</ajuste>",    "\n"),
    ("<salida>",     ""),
    ("</salida>",    "\n"),
    ("<o>",          ""),
    ("</o>",         "\n"),
    ("‹reflexión>",      ""),
    ("‹/reflexión>",     "\n"),
    ("‹pensamiento>",    ""),
    ("‹/pensamiento>",   "\n"),
    ("\r",           ""),
];

/// Limpia las etiquetas de control del dataset y normaliza saltos de línea
fn limpiar(texto: &str) -> String {
    let mut s = texto.to_string();
    for (patron, reemplazo) in TOKENS_ESPECIALES {
        s = s.replace(patron, reemplazo);
    }
    
    let mut resultado = String::with_capacity(s.len());
    let mut linea_vacia_previa = false;
    
    for linea in s.lines() {
        let trimmed = linea.trim();
        if trimmed.is_empty() {
            if !linea_vacia_previa { 
                resultado.push('\n'); 
            }
            linea_vacia_previa = true;
        } else {
            resultado.push_str(trimmed);
            resultado.push('\n');
            linea_vacia_previa = false;
        }
    }
    resultado
}

// ── Extracción JSONL ──────────────────────────────────────────────

/// Extrae texto útil desde diversas estructuras JSONL conocidas
fn extraer_texto_jsonl(linea: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(linea.trim()).ok()?;
    let obj = v.as_object()?;

    // Formato estándar {"text": "..."}
    if let Some(t) = obj.get("text").and_then(|x| x.as_str()) {
        let limpio = limpiar(t);
        if !limpio.trim().is_empty() { return Some(limpio); }
    }

    // Formato Alpaca {"instruction": "...", "input": "...", "output": "..."}
    if let (Some(instr), Some(out)) = (
        obj.get("instruction").and_then(|x| x.as_str()),
        obj.get("output").and_then(|x| x.as_str()),
    ) {
        let entrada = obj.get("input").and_then(|x| x.as_str()).unwrap_or("");
        let combinado = if entrada.trim().is_empty() {
            format!("{}\n{}", instr, out)
        } else {
            format!("{}\n{}\n{}", instr, entrada, out)
        };
        let limpio = limpiar(&combinado);
        if !limpio.trim().is_empty() { return Some(limpio); }
    }

    // Formato QA {"question": "...", "answer": "..."}
    if let (Some(q), Some(a)) = (
        obj.get("question").and_then(|x| x.as_str()),
        obj.get("answer").and_then(|x| x.as_str()),
    ) {
        let limpio = limpiar(&format!("{}\n{}", q, a));
        if !limpio.trim().is_empty() { return Some(limpio); }
    }

    // Formatos genéricos alternativos
    if let Some(c) = obj.get("content").and_then(|x| x.as_str()) {
        let limpio = limpiar(c);
        if !limpio.trim().is_empty() { return Some(limpio); }
    }

    for campo in &["response", "completion", "generated_text"] {
        if let Some(t) = obj.get(*campo).and_then(|x| x.as_str()) {
            let limpio = limpiar(t);
            if !limpio.trim().is_empty() { return Some(limpio); }
        }
    }

    None
}

// ── CorpusStream ──────────────────────────────────────────────────

pub struct CorpusStream {
    rutas:       Vec<String>,
    tok:         Tokenizador,
    buffer:      Vec<u32>,
    archivo_idx: usize,
    lector:      Option<std::io::Lines<BufReader<File>>>,
    pub tokens_entregados: u64,
    pub docs_procesados:   u64,
    pub docs_sin_campo:    u64,
    sep_token:   u32,
}

impl CorpusStream {
    pub fn abrir(rutas: &[&str], vocab_path: &str) -> Result<Self> {
        anyhow::ensure!(!rutas.is_empty(), "[stream] no se proporcionaron archivos fuente");

        let tok = Tokenizador::cargar(vocab_path)?;
        let rutas_owned: Vec<String> = rutas.iter().map(|s| s.to_string()).collect();

        println!("[stream] Inicializando fuentes: {} archivo(s)", rutas_owned.len());
        for r in &rutas_owned {
            println!("[stream]  └─ {}", r);
        }

        let mut stream = Self {
            rutas:              rutas_owned,
            tok,
            buffer:             Vec::with_capacity(8192),
            archivo_idx:        0,
            lector:             None,
            tokens_entregados:  0,
            docs_procesados:    0,
            docs_sin_campo:     0,
            sep_token:          1,
        };

        stream.abrir_siguiente_archivo()?;
        Ok(stream)
    }

    fn abrir_siguiente_archivo(&mut self) -> Result<bool> {
        if self.archivo_idx >= self.rutas.len() {
            return Ok(false);
        }
        let ruta = &self.rutas[self.archivo_idx];
        let file = File::open(ruta)
            .with_context(|| format!("[stream] error abriendo corpus: {}", ruta))?;
        println!("[stream] Leyendo dataset: {}", ruta);
        self.lector = Some(BufReader::new(file).lines());
        Ok(true)
    }

    fn llenar_buffer(&mut self, minimo: usize) -> Result<bool> {
        while self.buffer.len() < minimo {
                        let linea_opt = match &mut self.lector {
                Some(lector) => lector.next(),
                None => {
                    self.archivo_idx += 1;
                    if self.archivo_idx >= self.rutas.len() {
                        return Ok(false);
                    }
                    self.abrir_siguiente_archivo()?;
                    continue;
                }
            };

            match linea_opt {
                Some(Ok(linea)) => {
                    if linea.trim().is_empty() { continue; }

                    let ext = self.rutas[self.archivo_idx]
                        .rsplit('.')
                        .next()
                        .unwrap_or("");

                    let texto_opt = match ext {
                        "jsonl" => extraer_texto_jsonl(&linea),
                        "txt"   => {
                            let limpio = limpiar(linea.trim());
                            if limpio.trim().is_empty() { None } else { Some(limpio) }
                        }
                        _       => extraer_texto_jsonl(&linea),
                    };

                    match texto_opt {
                        Some(texto) => {
                            let ids = self.tok.encode(&texto);
                            if !ids.is_empty() {
                                self.buffer.extend_from_slice(&ids);
                                self.buffer.push(self.sep_token);
                                self.docs_procesados += 1;
                            }
                        }
                        None => {
                            self.docs_sin_campo += 1;
                        }
                    }
                }

                Some(Err(e)) => {
                    eprintln!("[stream] WARN: error leyendo línea: {}", e);
                    continue;
                }

                None => {
                    self.archivo_idx += 1;
                    if self.archivo_idx >= self.rutas.len() {
                        return Ok(false);
                    }
                    self.lector = None;
                    self.abrir_siguiente_archivo()?;
                }
            }
        }
        Ok(true)
    }

    pub fn next_batch(&mut self, batch_tokens: usize) -> Result<Option<Vec<u32>>> {
        let hay_datos = self.llenar_buffer(batch_tokens)?;

        if !hay_datos && self.buffer.len() < batch_tokens {
            return Ok(None);
        }

        let batch: Vec<u32> = self.buffer.drain(..batch_tokens).collect();
        self.tokens_entregados += batch_tokens as u64;
        Ok(Some(batch))
    }

    pub fn reiniciar(&mut self) -> Result<()> {
        self.archivo_idx = 0;
        self.lector      = None;
        self.buffer.clear();
        self.abrir_siguiente_archivo()?;
        println!("[stream] Stream reiniciado exitosamente para nueva época.");
        Ok(())
    }

    pub fn stats(&self) {
        println!("[stream] Documentos procesados : {}", self.docs_procesados);
        println!("[stream] Tokens entregados     : {}", self.tokens_entregados);
        if self.docs_sin_campo > 0 {
            println!("[stream] Docs sin campo útil : {} (Sugerencia: revisar formato)", self.docs_sin_campo);
        }
    }

    pub fn contar_lineas_corpus(rutas: &[String]) -> u64 {
        let mut total: u64 = 0;
        for ruta in rutas {
            let ext = ruta.rsplit('.').next().unwrap_or("");
            let file = match File::open(ruta) {
                Ok(f)  => f,
                Err(_) => {
                    eprintln!("[stream] WARN: no se puede abrir archivo para conteo: {}", ruta);
                    continue;
                }
            };
            let reader = BufReader::new(file);
            for linea in reader.lines() {
                let linea = match linea { Ok(l) => l, Err(_) => continue };
                if linea.trim().is_empty() { continue; }
                let tiene_campo = match ext {
                    "jsonl" => extraer_texto_jsonl(&linea).is_some(),
                    "txt"   => !linea.trim().is_empty(),
                    _       => extraer_texto_jsonl(&linea).is_some(),
                };
                if tiene_campo { total += 1; }
            }
        }
        total
    }

    /// Muestrea `n` documentos del corpus y retorna el promedio real de tokens por doc.
    /// Usa el tokenizador ya cargado en el stream — sin instanciar uno nuevo.
    /// Llamar DESPUÉS de `abrir()` y ANTES de `reiniciar()`.
    pub fn muestrear_tokens_promedio_real(&self, n: usize) -> u64 {
        let mut total_tokens: u64 = 0;
        let mut count: u64 = 0;

        'outer: for ruta in &self.rutas {
            let ext = ruta.rsplit('.').next().unwrap_or("");
            let file = match File::open(ruta) {
                Ok(f)  => f,
                Err(_) => continue,
            };
            let reader = BufReader::new(file);

            for linea in reader.lines() {
                if count >= n as u64 { break 'outer; }
                let linea = match linea { Ok(l) => l, Err(_) => continue };
                if linea.trim().is_empty() { continue; }

                let texto_opt = match ext {
                    "jsonl" => extraer_texto_jsonl(&linea),
                    "txt"   => {
                        let l = limpiar(linea.trim());
                        if l.trim().is_empty() { None } else { Some(l) }
                    }
                    _ => extraer_texto_jsonl(&linea),
                };

                if let Some(texto) = texto_opt {
                    let ids = self.tok.encode(&texto);
                    if !ids.is_empty() {
                        total_tokens += ids.len() as u64;
                        count += 1;
                    }
                }
            }
        }

        if count == 0 { 256 } else { total_tokens / count }
    }
}

// ── Utilidades I/O genéricas ──────────────────────────────────────

#[allow(dead_code)]
pub fn load_weights(path: &str, expected_floats: usize) -> Result<Vec<f32>> {
    let mut file = BufReader::new(
        File::open(path).with_context(|| format!("[io] fallo cargando pesos desde: {}", path))?
    );
    let mut buf = Vec::with_capacity(expected_floats * 4);
    file.read_to_end(&mut buf)?;

    anyhow::ensure!(buf.len() == expected_floats * 4,
        "[io] mismatch de tamaño: esperaba {} floats ({} bytes), recibió {} bytes",
        expected_floats, expected_floats * 4, buf.len());

    let floats: Vec<f32> = bytemuck::cast_slice(&buf).to_vec();
    println!("[io] load_weights: {} floats OK", floats.len());
    Ok(floats)
}

#[allow(dead_code)]
pub fn save_weights(path: &str, weights: &[f32]) -> Result<()> {
    let mut file = BufWriter::new(
        File::create(path).with_context(|| format!("[io] fallo guardando pesos en: {}", path))?
    );
    let bytes: &[u8] = bytemuck::cast_slice(weights);
    file.write_all(bytes)?;
    println!("[io] save_weights: guardados {} floats en {}", weights.len(), path);
    Ok(())
}

// ── Constantes Magic de Checkpoints ──────────────────────────────

const MAGIC_V2: u32 = 0x4B_32_52_41; // "ARK2" — solo pesos FP32
const MAGIC_V3: u32 = 0x4B_33_52_41; // "ARK3" — pesos + momentos en FP32
const MAGIC_V4: u32 = 0x4B_34_52_41; // "ARK4" — pesos FP16 + momentos FP32

// ── CheckpointV4 y V3 (Unified Struct) ───────────────────────────

pub struct CheckpointV3 {
    pub global_step:    u64,
    pub adam_step:      u64,
    pub pesos:          Vec<Vec<f32>>, // Descomprimidos a FP32 para consistencia
    pub momentos_m:     Vec<Vec<f32>>,
    pub momentos_v:     Vec<Vec<f32>>,
    pub tiene_momentos: bool,
}

impl CheckpointV3 {
    /// Guarda un checkpoint V4 ultra-eficiente en almacenamiento.
    ///
    /// Escribe los tensores de los pesos nativamente en FP16 (ocupando la mitad de RAM en disco)
    /// y los tensores de los momentos (m, v) en FP32 para mantener estricta precisión de optimización.
    pub fn save_fp16(
        path: &str,
        global_step: u64,
        adam_step:   u64,
        embed_w_f16: &[u16],
        embed_m_f32: &[f32],
        embed_v_f32: &[f32],
        gamma_f_f16: &[u16],         // Nuevo
        gamma_f_m_f32: &[f32],       // Nuevo
        gamma_f_v_f32: &[f32],       // Nuevo
        layer_w_f16: &[Vec<&[u16]>],
        layer_m_f32: &[Vec<&[f32]>],
        layer_v_f32: &[Vec<&[f32]>],
    ) -> Result<()> {
        let n_layers = layer_w_f16.len();
        // sumamos 3 tensores más al total (peso, momento m, momento v)
        let n_tensors: u64 = (3 + n_layers * 27 + 3) as u64; 

        let mut f = BufWriter::new(File::create(path)?);

        // Header v4
        f.write_all(&MAGIC_V4.to_le_bytes())?;
        f.write_all(&global_step.to_le_bytes())?;
        f.write_all(&adam_step.to_le_bytes())?;
        f.write_all(&n_tensors.to_le_bytes())?;

        // Lambdas auxiliares
        let write_tensor_f16 = |f: &mut BufWriter<File>, data: &[u16]| -> Result<()> {
            f.write_all(&(data.len() as u64).to_le_bytes())?;
            f.write_all(&0u64.to_le_bytes())?; // Flag 0 = F16
            f.write_all(bytemuck::cast_slice(data))?;
            Ok(())
        };
        
        let write_tensor_f32 = |f: &mut BufWriter<File>, data: &[f32]| -> Result<()> {
            f.write_all(&(data.len() as u64).to_le_bytes())?;
            f.write_all(&1u64.to_le_bytes())?; // Flag 1 = F32
            f.write_all(bytemuck::cast_slice(data))?;
            Ok(())
        };

        // 1. Escritura Embedding
        write_tensor_f16(&mut f, embed_w_f16)?;
        write_tensor_f32(&mut f, embed_m_f32)?;
        write_tensor_f32(&mut f, embed_v_f32)?;

        // 2. Escritura Capas Transformer (Primero los pesos F16, luego momentos M y V F32)
        for l in 0..n_layers {
            for tensor in &layer_w_f16[l] {
                write_tensor_f16(&mut f, tensor)?;
            }
            for tensor in &layer_m_f32[l] {
                write_tensor_f32(&mut f, tensor)?;
            }
            for tensor in &layer_v_f32[l] {
                write_tensor_f32(&mut f, tensor)?;
            }
        }

        // 3. Escritura Capa de Normalización Final
        write_tensor_f16(&mut f, gamma_f_f16)?;
        write_tensor_f32(&mut f, gamma_f_m_f32)?;
        write_tensor_f32(&mut f, gamma_f_v_f32)?;

        let size_mb = f.stream_position()? as f32 / 1_000_000.0;
        println!("[checkpoint v4] Ckpt guardado exitosamente: step={} | adam_step={} | {:.1} MB",
                 global_step, adam_step, size_mb);
        Ok(())
    }

    /// Carga inteligente de Checkpoints.
    /// Auto-detecta la versión (V2, V3, V4). Extrae todo consistentemente en FP32.
    pub fn load(path: &str) -> Result<Self> {
        if !Path::new(path).exists() {
            anyhow::bail!("[checkpoint] Archivo de checkpoint no encontrado: {}", path);
        }
        
        let mut f = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        f.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);

        f.read_exact(&mut buf8)?;
        let global_step = u64::from_le_bytes(buf8);

        match magic {
            MAGIC_V4 => {
                f.read_exact(&mut buf8)?;
                let adam_step = u64::from_le_bytes(buf8);
                f.read_exact(&mut buf8)?;
                let n_tensors = u64::from_le_bytes(buf8) as usize;

                let mut all_tensors: Vec<Vec<f32>> = Vec::with_capacity(n_tensors);

                for _ in 0..n_tensors {
                    f.read_exact(&mut buf8)?;
                    let n_elements = u64::from_le_bytes(buf8) as usize;
                    f.read_exact(&mut buf8)?;
                    let tipo = u64::from_le_bytes(buf8);

                    if tipo == 0 {
                        // Es FP16: Leemos como bytes -> Cast a u16 -> Iteramos y convertimos a f32
                        let byte_len = n_elements * 2;
                        let mut raw = vec![0u8; byte_len];
                        f.read_exact(&mut raw)?;
                        
                        let f16_slice: &[u16] = bytemuck::cast_slice(&raw);
                        let f32_vec: Vec<f32> = f16_slice
                            .iter()
                            .map(|&bits| crate::memory::f16_to_f32(bits))
                            .collect();
                        
                        all_tensors.push(f32_vec);
                    } else {
                        // Es FP32
                        let mut raw = vec![0u8; n_elements * 4];
                        f.read_exact(&mut raw)?;
                        all_tensors.push(bytemuck::cast_slice::<u8, f32>(&raw).to_vec());
                    }
                }

                let embed_w = all_tensors[0].clone();
                let embed_m = all_tensors[1].clone();
                let embed_v = all_tensors[2].clone();

                // Detección dinámica de la RMSNorm final (3 tensores extra al final)
                let has_gamma_final = (n_tensors - 3) % 27 == 3;
                let n_layers = if has_gamma_final {
                    (n_tensors - 6) / 27
                } else {
                    (n_tensors - 3) / 27
                };

                let remaining = &all_tensors[3..3 + n_layers * 27];

                let mut pesos      = vec![embed_w];
                let mut momentos_m = vec![embed_m];
                let mut momentos_v = vec![embed_v];

                for l in 0..n_layers {
                    let base = l * 27;
                    for i in 0..9 { pesos.push(remaining[base + i].clone()); }
                    for i in 0..9 { momentos_m.push(remaining[base + 9 + i].clone()); }
                    for i in 0..9 { momentos_v.push(remaining[base + 18 + i].clone()); }
                }

                // Si contiene la RMSNorm final, se añade al final de los vectores correspondientes
                if has_gamma_final {
                    let gf_offset = 3 + n_layers * 27;
                    pesos.push(all_tensors[gf_offset + 0].clone());
                    momentos_m.push(all_tensors[gf_offset + 1].clone());
                    momentos_v.push(all_tensors[gf_offset + 2].clone());
                }

                println!("[checkpoint v4] Carga de pesos FP16 nativa finalizada: step={} | adam={} | capas={}",
                         global_step, adam_step, n_layers);
                Ok(Self {
                    global_step, adam_step,
                    pesos, momentos_m, momentos_v,
                    tiene_momentos: true,
                })
            }

            MAGIC_V3 => {
                f.read_exact(&mut buf8)?;
                let adam_step = u64::from_le_bytes(buf8);
                f.read_exact(&mut buf8)?;
                let n_tensors = u64::from_le_bytes(buf8) as usize;

                let mut all_tensors = Vec::with_capacity(n_tensors);
                for _ in 0..n_tensors {
                    f.read_exact(&mut buf8)?;
                    let n = u64::from_le_bytes(buf8) as usize;
                    let mut raw = vec![0u8; n * 4];
                    f.read_exact(&mut raw)?;
                    all_tensors.push(bytemuck::cast_slice::<u8, f32>(&raw).to_vec());
                }

                let embed_w = all_tensors[0].clone();
                let embed_m = all_tensors[1].clone();
                let embed_v = all_tensors[2].clone();
                let remaining = &all_tensors[3..];
                let n_layers = remaining.len() / 27;

                let mut pesos      = vec![embed_w];
                let mut momentos_m = vec![embed_m];
                let mut momentos_v = vec![embed_v];

                for l in 0..n_layers {
                    let base = l * 27;
                    for i in 0..9 { pesos.push(remaining[base + i].clone()); }
                    for i in 0..9 { momentos_m.push(remaining[base + 9 + i].clone()); }
                    for i in 0..9 { momentos_v.push(remaining[base + 18 + i].clone()); }
                }

                println!("[checkpoint v3] Carga FP32 completa finalizada: step={} | adam={} | capas={}",
                         global_step, adam_step, n_layers);
                Ok(Self {
                    global_step, adam_step,
                    pesos, momentos_m, momentos_v,
                    tiene_momentos: true,
                })
            }

            MAGIC_V2 => {
                f.read_exact(&mut buf8)?;
                let n_tensors = u64::from_le_bytes(buf8) as usize;
                let mut tensors = Vec::with_capacity(n_tensors);
                
                for _ in 0..n_tensors {
                    f.read_exact(&mut buf8)?;
                    let n = u64::from_le_bytes(buf8) as usize;
                    let mut raw = vec![0u8; n * 4];
                    f.read_exact(&mut raw)?;
                    tensors.push(bytemuck::cast_slice::<u8, f32>(&raw).to_vec());
                }
                
                println!("[checkpoint v2] Migración en memoria: step={} | tensores={} \
                          (Optimizer iniciará desde cero para este checkpoint)", 
                          global_step, n_tensors);
                
                Ok(Self {
                    global_step,
                    adam_step: 0,
                    pesos: tensors,
                    momentos_m: Vec::new(),
                    momentos_v: Vec::new(),
                    tiene_momentos: false,
                })
            }

            _ => anyhow::bail!(
                "[checkpoint] FALLO CRÍTICO: Magic signature 0x{:08X} inválida. Archivo corrupto.",
                magic
            ),
        }
    }
}

// ── Checkpoint Legacy (V1/V2 compatibilidad directa) ─────────────────────────

pub struct Checkpoint {
    pub global_step: u64,
    pub tensors:     Vec<Vec<f32>>,
}

impl Checkpoint {
    /// Carga compatibilidad rápida, extrayendo solamente los pesos (omitiendo momentos si existen).
    pub fn load(path: &str) -> Result<Self> {
        let ckpt_full = CheckpointV3::load(path)?;
        Ok(Self {
            global_step: ckpt_full.global_step,
            tensors:     ckpt_full.pesos,
        })
    }

    pub fn is_full(&self) -> bool {
        self.tensors.len() > 1
    }
}

// ── CorpusBuilder (Constructor Offline de Binarios) ───────────────────────────

pub struct CorpusBuilder;

impl CorpusBuilder {
    pub fn build(fuentes: &[&str], vocab_path: &str, salida: &str) -> Result<()> {
        println!("[corpus-build] Construyendo dataset .bin → {}", salida);
        let tok = Tokenizador::cargar(vocab_path)?;
        let mut tokens: Vec<u32> = Vec::new();
        let mut docs_ok:  u64 = 0;
        let mut docs_err: u64 = 0;
        const SEP_TOKEN: u32 = 1;

        for ruta in fuentes {
            let ext = Path::new(ruta).extension().and_then(|e| e.to_str()).unwrap_or("");
            let file = File::open(ruta)
                .with_context(|| format!("[corpus-build] Error abriendo archivo: {}", ruta))?;
            let reader = BufReader::new(file);

            match ext {
                "jsonl" => {
                    println!("[corpus-build] Analizando JSONL: {}", ruta);
                    for (n, linea) in reader.lines().enumerate() {
                        let linea = linea?;
                        if linea.trim().is_empty() { continue; }
                        
                        match extraer_texto_jsonl(&linea) {
                            Some(texto) => {
                                let ids = tok.encode(&texto);
                                if !ids.is_empty() {
                                    tokens.extend_from_slice(&ids);
                                    tokens.push(SEP_TOKEN);
                                    docs_ok += 1;
                                }
                            }
                            None => {
                                docs_err += 1;
                                if docs_err <= 3 {
                                    eprintln!("[corpus-build] WARN: Línea {} sin campo útil: {}…",
                                        n + 1, &linea[..linea.len().min(80)]);
                                }
                            }
                        }
                    }
                }
                "txt" => {
                    println!("[corpus-build] Analizando TXT: {}", ruta);
                    for linea in reader.lines() {
                        let linea = linea?;
                        let trimmed = linea.trim();
                        if trimmed.is_empty() { continue; }
                        
                        let ids = tok.encode(&limpiar(trimmed));
                        if !ids.is_empty() {
                            tokens.extend_from_slice(&ids);
                            tokens.push(SEP_TOKEN);
                            docs_ok += 1;
                        }
                    }
                }
                otro => {
                    eprintln!("[corpus-build] WARN: Extensión '{}' no reconocida, omitida: {}", otro, ruta);
                }
            }
        }

        anyhow::ensure!(!tokens.is_empty(), "[corpus-build] FALLO: No se extrajeron tokens de las fuentes proporcionadas.");

        let mut out = BufWriter::new(File::create(salida)?);
        out.write_all(bytemuck::cast_slice(&tokens))?;

        println!("[corpus-build] Resumen:");
        println!("  ├─ Docs OK       : {}", docs_ok);
        if docs_err > 0 { 
            println!("  ├─ Docs Ignorados: {}", docs_err); 
        }
        println!("  ├─ Total Tokens  : {}", tokens.len());
        println!("  └─ Tamaño Output : {} MB", tokens.len() * 4 / 1_000_000);
        
        Ok(())
    }

    #[allow(dead_code)]
    pub fn build_uno(fuente: &str, vocab_path: &str, salida: &str) -> Result<()> {
        Self::build(&[fuente], vocab_path, salida)
    }
}