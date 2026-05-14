// eko_infer/src/main.rs — EKO Inferenciador ARK v1.3
// CPU puro. Sin sentencepiece. Sin cmake. Sin dependencias de sistema.
// Tokenizador Viterbi sobre vocab_sp.json + vocab_scores.json.

use anyhow::{bail, Context, Result};
use half::f16;
use rayon::prelude::*; // <-- Importamos rayon para usar todos los núcleos
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

// ── Args ──────────────────────────────────────────────────────────────────────

struct Args {
    ckpt:       String,
    vocab:      String,
    scores:     String,
    n_layers:   usize,
    n_heads:    usize,
    d_model:    usize,
    hidden:     usize,
    vocab_size: usize,
    prompt:     Option<String>,
    max_tokens: usize,
    temp:       f32,
    top_p:      f32,
    rep_penalty: f32,
    seed:       u64,
}

impl Args {
    fn parse() -> Result<Self> {
        let argv: Vec<String> = std::env::args().collect();
        let mut a = Args {
            ckpt:        String::new(),
            vocab:       String::new(),
            scores:      String::new(),
            n_layers:    30,
            n_heads:     12,
            d_model:     768,
            hidden:      2048,
            vocab_size:  32000,
            prompt:      None,
            max_tokens:  200,
            temp:        0.8,
            top_p:       0.92,
            rep_penalty: 1.15,
            seed:        42,
        };
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--ckpt"        => { i+=1; a.ckpt        = argv[i].clone(); }
                "--vocab"       => { i+=1; a.vocab       = argv[i].clone(); }
                "--scores"      => { i+=1; a.scores      = argv[i].clone(); }
                "--layers"      => { i+=1; a.n_layers    = argv[i].parse()?; }
                "--heads"       => { i+=1; a.n_heads     = argv[i].parse()?; }
                "--d-model"     => { i+=1; a.d_model     = argv[i].parse()?; }
                "--hidden"      => { i+=1; a.hidden      = argv[i].parse()?; }
                "--vocab-size"  => { i+=1; a.vocab_size  = argv[i].parse()?; }
                "--prompt"      => { i+=1; a.prompt      = Some(argv[i].clone()); }
                "--max-tokens"  => { i+=1; a.max_tokens  = argv[i].parse()?; }
                "--temp"        => { i+=1; a.temp        = argv[i].parse()?; }
                "--top-p"       => { i+=1; a.top_p       = argv[i].parse()?; }
                "--rep-penalty" => { i+=1; a.rep_penalty = argv[i].parse()?; }
                "--seed"        => { i+=1; a.seed        = argv[i].parse()?; }
                // compat: ignorar --merges si alguien lo pasa todavía
                "--merges"      => { i+=1; }
                other => bail!("Argumento desconocido: {}", other),
            }
            i += 1;
        }
        if a.ckpt.is_empty()  { bail!("Falta --ckpt"); }
        if a.vocab.is_empty() { bail!("Falta --vocab"); }
        if a.scores.is_empty() { bail!("Falta --scores"); }
        Ok(a)
    }
}

// ── Tokenizador Viterbi ───────────────────────────────────────────────────────
//
// SentencePiece BPE no guarda los merges en el .model — solo piezas con scores.
// Implementamos segmentación Viterbi: para cada posición del texto encontramos
// la segmentación de máxima log-probabilidad según los scores del vocab.
// Resultado idéntico al tokenizador original para texto normal.

const METASPACE: char = '\u{2581}';
const NEG_INF: f64 = f64::NEG_INFINITY;

struct Tokenizer {
    vocab:    HashMap<String, u32>,
    id2piece: Vec<String>,
    scores:   HashMap<String, f64>,
    unk_id:   u32,
    eos_id:   u32,
}

impl Tokenizer {
    fn load(vocab_path: &str, scores_path: &str) -> Result<Self> {
        // vocab_sp.json: {"pieza": id, ...}
        let raw = std::fs::read_to_string(vocab_path)
            .with_context(|| format!("No se puede leer: {}", vocab_path))?;
        let jv: serde_json::Value = serde_json::from_str(&raw)?;
        let vocab_obj = jv.as_object().context("vocab_sp.json: no es objeto")?;
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (k, v) in vocab_obj {
            vocab.insert(k.clone(), v.as_u64().unwrap_or(0) as u32);
        }
        let mut id2piece: Vec<String> = vec![String::new(); vocab.len()];
        for (k, &id) in &vocab {
            if (id as usize) < id2piece.len() {
                id2piece[id as usize] = k.clone();
            }
        }

        // vocab_scores.json: {"pieza": score_float, ...}
        let raw2 = std::fs::read_to_string(scores_path)
            .with_context(|| format!("No se puede leer: {}", scores_path))?;
        let js: serde_json::Value = serde_json::from_str(&raw2)?;
        let scores_obj = js.as_object().context("vocab_scores.json: no es objeto")?;
        let mut scores: HashMap<String, f64> = HashMap::new();
        for (k, v) in scores_obj {
            scores.insert(k.clone(), v.as_f64().unwrap_or(NEG_INF));
        }

        let unk_id = vocab.get("<OOV>").or_else(|| vocab.get("<unk>")).copied().unwrap_or(0);
        let eos_id = vocab.get("</s>").copied().unwrap_or(2);

        eprintln!("[tok] vocab={} scores={} unk={} eos={}",
                  vocab.len(), scores.len(), unk_id, eos_id);
        Ok(Self { vocab, id2piece, scores, unk_id, eos_id })
    }

    /// Segmentación Viterbi de una palabra (ya tiene ▁ prefijado si aplica)
    fn viterbi_word(&self, word: &str) -> Vec<u32> {
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();
        if n == 0 { return vec![]; }

        // Precomputar offsets de bytes por posición de char
        let mut byte_off = vec![0usize; n + 1];
        for i in 0..n {
            byte_off[i + 1] = byte_off[i] + chars[i].len_utf8();
        }

        // best[i] = (score_acumulado, posicion_inicio_ultimo_token)
        let mut best: Vec<(f64, usize)> = vec![(NEG_INF, 0); n + 1];
        best[0] = (0.0, 0);

        for i in 0..n {
            if best[i].0 == NEG_INF { continue; }
            for j in (i + 1)..=n {
                let piece = &word[byte_off[i]..byte_off[j]];
                let score: f64 = if let Some(&s) = self.scores.get(piece) {
                    s
                } else if self.vocab.contains_key(piece) {
                    // En vocab pero sin score (token especial) — penalizar levemente
                    -10000.0
                } else if j == i + 1 {
                    // Carácter único no en vocab — UNK forzado
                    -50000.0
                } else {
                    continue; // span no existe en vocab — saltar
                };
                let candidate = best[i].0 + score;
                if candidate > best[j].0 {
                    best[j] = (candidate, i);
                }
            }
        }

        // Reconstruir camino desde n hasta 0
        let mut pieces: Vec<String> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let prev = best[pos].1;
            pieces.push(word[byte_off[prev]..byte_off[pos]].to_string());
            pos = prev;
        }
        pieces.reverse();

        // Convertir a IDs
        pieces.iter().map(|p| {
            self.vocab.get(p.as_str()).copied().unwrap_or(self.unk_id)
        }).collect()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let prefixed = format!("{}{}", METASPACE, word);
            ids.extend(self.viterbi_word(&prefixed));
        }
        ids
    }

    fn decode_one(&self, id: u32) -> String {
        if (id as usize) >= self.id2piece.len() { return String::new(); }
        self.id2piece[id as usize].replace(METASPACE, " ")
    }
}

// ── Checkpoint ────────────────────────────────────────────────────────────────

const MAGIC_V4: u32 = 0x4B_34_52_41;
const MAGIC_V3: u32 = 0x4B_33_52_41;
const MAGIC_V2: u32 = 0x4B_32_52_41;

struct CheckpointWeights {
    global_step: u64,
    n_layers:    usize,
    embed_w:     Vec<f32>,
    layers:      Vec<LayerW>,
}

struct LayerW {
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
    w1: Vec<f32>, w2: Vec<f32>, w3: Vec<f32>,
    g1: Vec<f32>, g2: Vec<f32>,
}

fn f16_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let u16s: &[u16] = bytemuck::cast_slice(bytes);
    u16s.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

fn read_n(f: &mut BufReader<File>, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn load_checkpoint(path: &str) -> Result<CheckpointWeights> {
    let mut f = BufReader::new(File::open(path)
        .with_context(|| format!("No se puede abrir: {}", path))?);
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];

    f.read_exact(&mut b4)?;
    let magic = u32::from_le_bytes(b4);
    f.read_exact(&mut b8)?;
    let global_step = u64::from_le_bytes(b8);

    let tensors: Vec<Vec<f32>> = match magic {
        MAGIC_V4 => {
            f.read_exact(&mut b8)?; let adam = u64::from_le_bytes(b8);
            f.read_exact(&mut b8)?; let n    = u64::from_le_bytes(b8) as usize;
            eprintln!("[ckpt] v4 paso={global_step} adam={adam} tensores={n}");

            // Leer todos los tensores del disco
            let mut all: Vec<Vec<f32>> = Vec::with_capacity(n);
            for _ in 0..n {
                f.read_exact(&mut b8)?; let ne   = u64::from_le_bytes(b8) as usize;
                f.read_exact(&mut b8)?; let tipo = u64::from_le_bytes(b8);
                if tipo == 0 {
                    all.push(f16_to_f32_vec(&read_n(&mut f, ne * 2)?));
                } else {
                    all.push(bytemuck::cast_slice::<u8, f32>(&read_n(&mut f, ne * 4)?).to_vec());
                }
            }

            // Extraer solo pesos — formato por capa: [9 pesos][9 m][9 v]
            // tensor 0 = embed_w, 1 = embed_m, 2 = embed_v
            // luego bloques de 27 por capa (9 pesos + 9 m + 9 v)
            let n_layers = (n - 3) / 27;
            let mut out = Vec::with_capacity(3 + n_layers * 9);
            out.push(all[0].clone()); // embed_w
            // lm_head usa weight tying — no hay tensor separado
            for l in 0..n_layers {
                let base = 3 + l * 27;
                for t in 0..9 {
                    out.push(all[base + t].clone()); // solo pesos, saltar m y v
                }
            }
            out
        }
        MAGIC_V3 => {
            f.read_exact(&mut b8)?;
            f.read_exact(&mut b8)?; let n = u64::from_le_bytes(b8) as usize;
            eprintln!("[ckpt] v3 paso={global_step} tensores={n}");
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                f.read_exact(&mut b8)?; let ne = u64::from_le_bytes(b8) as usize;
                out.push(bytemuck::cast_slice::<u8, f32>(&read_n(&mut f, ne * 4)?).to_vec());
            }
            out
        }
        MAGIC_V2 => {
            f.read_exact(&mut b8)?; let n = u64::from_le_bytes(b8) as usize;
            eprintln!("[ckpt] v2 paso={global_step} tensores={n}");
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                f.read_exact(&mut b8)?; let ne = u64::from_le_bytes(b8) as usize;
                out.push(bytemuck::cast_slice::<u8, f32>(&read_n(&mut f, ne * 4)?).to_vec());
            }
            out
        }
        _ => bail!("[ckpt] Magic inválido: 0x{:08X}", magic),
    };

    let embed_w = tensors[0].clone();
    let rest    = &tensors[1..]; // solo embed_w + capas (sin lm_head separado)
    let n_layers = rest.len() / 9;
    eprintln!("[ckpt] capas={n_layers} embed={} elems", embed_w.len());

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let b = l * 9;
        layers.push(LayerW {
            wq: rest[b].clone(),     wk: rest[b+1].clone(),
            wv: rest[b+2].clone(),   wo: rest[b+3].clone(),
            w1: rest[b+4].clone(),   w2: rest[b+5].clone(),
            w3: rest[b+6].clone(),
            g1: rest[b+7].clone(),   g2: rest[b+8].clone(),
        });
    }
    Ok(CheckpointWeights { global_step, n_layers, embed_w, layers })
}

// ── Primitivas CPU ────────────────────────────────────────────────────────────

#[inline]
fn rms_norm(x: &[f32], g: &[f32], out: &mut [f32]) {
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + 1e-6).sqrt();
    let inv = 1.0 / rms;
    for i in 0..x.len() { out[i] = x[i] * inv * g[i]; }
}

#[inline]
fn matvec(a: &[f32], b: &[f32], out: &mut [f32], _n: usize, k: usize) {
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = a[i*k..(i+1)*k].iter().zip(b).map(|(&x, &y)| x * y).sum();
    });
}

#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); s += *v; }
    let inv = 1.0 / s;
    for v in x.iter_mut() { *v *= inv; }
}

fn apply_rope(x: &mut [f32], pos: usize) {
    // RoPE Entrelazado (Pares e impares)
    for i in (0..x.len()).step_by(2) {
        let theta = pos as f32 / 10000f32.powf(i as f32 / x.len() as f32);
        let (s, c) = theta.sin_cos();
        let (x0, x1) = (x[i], x[i+1]);
        x[i]   = x0 * c - x1 * s;
        x[i+1] = x0 * s + x1 * c;
    }
}

// ── Modelo ────────────────────────────────────────────────────────────────────

struct ArkModel {
    n_layers: usize, n_heads: usize, d_model: usize,
    hidden:   usize, head_dim: usize, vocab:   usize,
    embed_w:  Vec<f32>,
    layers:   Vec<LayerW>,
    kv_k:     Vec<Vec<Vec<f32>>>,
    kv_v:     Vec<Vec<Vec<f32>>>,
}

impl ArkModel {
    fn new(mut ckpt: CheckpointWeights, n_heads: usize, d_model: usize,
           hidden: usize, vocab: usize) -> Self {
        let nl = ckpt.n_layers;

        eprintln!("[ark] Transponiendo pesos para máxima velocidad CPU...");
        let transpose = |w: &mut Vec<f32>, rows: usize, cols: usize| {
            let mut out = vec![0.0; w.len()];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = w[i * cols + j];
                }
            }
            *w = out;
        };

        for lw in &mut ckpt.layers {
            transpose(&mut lw.wq, d_model, d_model);
            transpose(&mut lw.wk, d_model, d_model);
            transpose(&mut lw.wv, d_model, d_model);
            transpose(&mut lw.wo, d_model, d_model);
            transpose(&mut lw.w1, d_model, hidden);
            transpose(&mut lw.w3, d_model, hidden);
            transpose(&mut lw.w2, hidden, d_model);
        }

        Self {
            n_layers: nl, n_heads, d_model, hidden,
            head_dim: d_model / n_heads, vocab,
            embed_w: ckpt.embed_w,
            layers:  ckpt.layers,
            kv_k: vec![Vec::new(); nl],
            kv_v: vec![Vec::new(); nl],
        }
    }

    fn reset_cache(&mut self) {
        for l in 0..self.n_layers {
            self.kv_k[l].clear();
            self.kv_v[l].clear();
        }
    }

    fn forward<'b>(&mut self, token_id: usize, pos: usize,
                   buf: &'b mut ForwardBuf) -> &'b [f32] {
        let (d, h, nh, hd) = (self.d_model, self.hidden, self.n_heads, self.head_dim);
        buf.x.copy_from_slice(&self.embed_w[token_id * d..(token_id + 1) * d]);

        for li in 0..self.n_layers {
            let lw = &self.layers[li];

            rms_norm(&buf.x, &lw.g1, &mut buf.xn);
            matvec(&lw.wq, &buf.xn, &mut buf.q, d, d);
            matvec(&lw.wk, &buf.xn, &mut buf.k, d, d);
            matvec(&lw.wv, &buf.xn, &mut buf.v, d, d);

            for hi in 0..nh {
                let s = hi * hd;
                apply_rope(&mut buf.q[s..s + hd], pos);
                apply_rope(&mut buf.k[s..s + hd], pos);
            }

            self.kv_k[li].push(buf.k.clone());
            self.kv_v[li].push(buf.v.clone());
            let seq     = self.kv_k[li].len();
            let inv_sqrt = 1.0 / (hd as f32).sqrt();

            buf.attn_out.iter_mut().for_each(|v| *v = 0.0);
            for hi in 0..nh {
                let hs = hi * hd;
                for t in 0..seq {
                    let k_t = &self.kv_k[li][t][hs..hs + hd];
                    buf.scores[t] = buf.q[hs..hs + hd].iter().zip(k_t)
                        .map(|(&a, &b)| a * b).sum::<f32>() * inv_sqrt;
                }
                softmax_inplace(&mut buf.scores[..seq]);
                for t in 0..seq {
                    let v_t = &self.kv_v[li][t][hs..hs + hd];
                    let w   = buf.scores[t];
                    for i in 0..hd { buf.attn_out[hs + i] += w * v_t[i]; }
                }
            }

            matvec(&lw.wo, &buf.attn_out, &mut buf.tmp_d, d, d);
            for i in 0..d { buf.x[i] += buf.tmp_d[i]; }

            rms_norm(&buf.x, &lw.g2, &mut buf.xn);
            matvec(&lw.w1, &buf.xn, &mut buf.gate, h, d);
            matvec(&lw.w3, &buf.xn, &mut buf.up,   h, d);
            for i in 0..h { buf.gate[i] = silu(buf.gate[i]) * buf.up[i]; }
            matvec(&lw.w2, &buf.gate, &mut buf.tmp_d, d, h);
            for i in 0..d { buf.x[i] += buf.tmp_d[i]; }
        }

        for v in 0..self.vocab {
            buf.logits[v] = buf.x.iter()
                .zip(&self.embed_w[v * d..(v + 1) * d])
                .map(|(&a, &b)| a * b).sum();
        }
        &buf.logits
    }
}

struct ForwardBuf {
    x:        Vec<f32>, xn:       Vec<f32>,
    q:        Vec<f32>, k:        Vec<f32>, v:    Vec<f32>,
    attn_out: Vec<f32>, tmp_d:    Vec<f32>,
    gate:     Vec<f32>, up:       Vec<f32>,
    scores:   Vec<f32>, logits:   Vec<f32>,
}

impl ForwardBuf {
    fn new(d: usize, h: usize, vocab: usize, max_seq: usize) -> Self {
        Self {
            x:        vec![0.0; d],
            xn:       vec![0.0; d],
            q:        vec![0.0; d],
            k:        vec![0.0; d],
            v:        vec![0.0; d],
            attn_out: vec![0.0; d],
            tmp_d:    vec![0.0; d],
            gate:     vec![0.0; h],
            up:       vec![0.0; h],
            scores:   vec![0.0; max_seq],
            logits:   vec![0.0; vocab],
        }
    }
}

// ── Sampler ───────────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(s: u64) -> Self { Self(s) }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32
    }
}

fn sample(logits: &mut Vec<f32>, temp: f32, top_p: f32,
          rng: &mut Rng, recent: &[u32], rep: f32) -> u32 {
    if rep != 1.0 {
        for &t in &recent[recent.len().saturating_sub(64)..] {
            let idx = t as usize;
            if logits[idx] < 0.0 {
                logits[idx] *= rep; // Si es negativo, multiplicar lo hace MÁS negativo (penaliza)
            } else {
                logits[idx] /= rep; // Si es positivo, dividir lo hace MÁS pequeño (penaliza)
            }
        }
    }
    if temp <= 0.0 {
        return logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap_or(0);
    }
    for v in logits.iter_mut() { *v /= temp; }
    softmax_inplace(logits);

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let mut cum = 0.0f32;
    let mut cut = idx.len();
    for (i, &id) in idx.iter().enumerate() {
        cum += logits[id];
        if cum >= top_p { cut = i + 1; break; }
    }
    idx.truncate(cut);
    let total: f32 = idx.iter().map(|&i| logits[i]).sum();
    let inv = 1.0 / total;
    let r   = rng.next_f32();
    let mut acc = 0.0f32;
    for &id in &idx {
        acc += logits[id] * inv;
        if r <= acc { return id as u32; }
    }
    *idx.last().unwrap() as u32
}

// ── Generación ────────────────────────────────────────────────────────────────

fn generate(
    model:      &mut ArkModel,
    buf:        &mut ForwardBuf,
    tok:        &Tokenizer,
    prompt_ids: &[u32],
    max_new:    usize,
    temp:       f32,
    top_p:      f32,
    rep:        f32,
    rng:        &mut Rng,
) -> Vec<u32> {
    model.reset_cache();
    eprint!("[gen] Prefill {} tokens... ", prompt_ids.len());
    let t0 = Instant::now();

    let mut logits = vec![0.0f32; model.vocab];
    for (pos, &tid) in prompt_ids.iter().enumerate() {
        let lg = model.forward(tid as usize, pos, buf);
        logits.copy_from_slice(lg);
    }
    eprintln!("listo en {:.2}s", t0.elapsed().as_secs_f32());

    let mut generated: Vec<u32> = Vec::new();
    let mut all_ids: Vec<u32>   = prompt_ids.to_vec();
    let mut pos = prompt_ids.len();

    for _ in 0..max_new {
        let next = sample(&mut logits, temp, top_p, rng, &all_ids, rep);
        generated.push(next);
        all_ids.push(next);

        let piece = tok.decode_one(next);
        print!("{}", piece);
        let _ = std::io::stdout().flush();

        if next == tok.eos_id { break; }

        let lg = model.forward(next as usize, pos, buf);
        logits.copy_from_slice(lg);
        pos += 1;
    }
    println!();
    generated
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse()?;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  EKO — ARK v1.3 · NOUS Project · iAsesoria.cl  ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    eprint!("[tok] Cargando {}... ", args.vocab);
    let tok = Tokenizer::load(&args.vocab, &args.scores)?;
    eprintln!("OK");

    eprintln!("[ckpt] Cargando {}...", args.ckpt);
    let t0   = Instant::now();
    let ckpt = load_checkpoint(&args.ckpt)?;
    let step = ckpt.global_step;
    eprintln!("[ckpt] Listo en {:.1}s", t0.elapsed().as_secs_f32());

    let mut model = ArkModel::new(ckpt, args.n_heads, args.d_model, args.hidden, args.vocab_size);
    let mut buf   = ForwardBuf::new(args.d_model, args.hidden, args.vocab_size, 2048);
    let mut rng   = Rng::new(args.seed);

    println!("[ark] {}L × {}H × {}D  FFN={}  vocab={}",
             model.n_layers, args.n_heads, args.d_model, args.hidden, args.vocab_size);
    println!("[ark] paso={}  temp={:.2}  top_p={:.2}  rep={:.2}\n",
             step, args.temp, args.top_p, args.rep_penalty);

    let stdin = std::io::stdin();
    let mut first = true;
    loop {
        let text = if let Some(ref p) = args.prompt {
            if !first { break; }
            first = false;
            p.clone()
        } else {
            eprint!("Prompt: ");
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            stdin.read_line(&mut line)?;
            let t = line.trim().to_string();
            if t.is_empty() || t == "salir" {
                println!("[eko] Saliendo.");
                break;
            }
            t
        };

        let ids = tok.encode(&text);
        if ids.is_empty() { eprintln!("[WARN] Sin tokens."); continue; }
        eprintln!("[tok] {} tokens", ids.len());
        print!("\nEKO: ");
        let _ = std::io::stdout().flush();

        let t0  = Instant::now();
        let out = generate(&mut model, &mut buf, &tok, &ids,
                           args.max_tokens, args.temp, args.top_p,
                           args.rep_penalty, &mut rng);
        let e = t0.elapsed().as_secs_f32();
        eprintln!("[gen] {} tokens  {:.1}s  {:.1} tok/s\n",
                  out.len(), e, out.len() as f32 / e);
        if args.prompt.is_some() { break; }
    }
    Ok(())
}