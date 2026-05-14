// src/config.rs — ARK v1.3 "METAL-REASONER"
//
// Configuración central del motor ARK.
// Todos los hiperparámetros de arquitectura, entrenamiento y optimizer
// se definen aquí. Los demás módulos importan `Config` desde este archivo.

use anyhow::{bail, Result};

// ──────────
// MARK: — Config
// ──────────

#[derive(Debug, Clone)]
pub struct Config {
    // ── Rutas ─────
    pub corpus_paths: Vec<String>,
    pub vocab_path:   String,
    pub ckpt_path:    String,

    // ── Arquitectura ───
    pub vocab_size:  usize,
    pub d_model:     usize,
    pub n_heads:     usize,
    pub head_dim:    usize,   // derivado: d_model / n_heads
    pub n_layers:    usize,
    pub hidden_dim:  usize,
    pub seq_len:     usize,

    // ── Precisión ────
    pub use_fp16: bool,

    // ── Entrenamiento ──────
    pub n_epochs:    usize,
    pub batch_size:  usize,  // secuencias por batch
    pub lr:          f32,
    pub loss_scale_max: f32,
    pub grad_clip:   f32,
    pub warmup_steps: usize,

    // ── AdamW ────────
    pub beta1:        f32,
    pub beta2:        f32,
    pub adam_eps:     f32,
    pub weight_decay: f32,
}

impl Config {
    /// Configuración por defecto optimizada para razonamiento en Apple Silicon 8 GB.
    /// Sigue la escala Chinchilla-optimum: 32 capas × d=512 × h=2048 ≈ 117M params.
    pub fn default_ark() -> Self {
        Self {
            corpus_paths: Vec::new(),
            vocab_path:   "entren/tokenizador_bpe_32k_v2.model".into(),
            ckpt_path:    "entren/modelo.bin".into(),

            vocab_size:  32_063,
            d_model:     512,
            n_heads:     8,
            head_dim:    64,    // 512 / 8 — recalculado por fix_derived()
            n_layers:    32,
            hidden_dim:  2_048,
            seq_len:     2_048,

            use_fp16:    true,

            n_epochs:    3,
            batch_size:  2,
            lr:          3e-4,
            loss_scale_max: 8192.0,
            grad_clip:   1.0,
            warmup_steps: 100,

            beta1:        0.9,
            beta2:        0.999,
            adam_eps:     1e-8,
            weight_decay: 0.01,
        }
    }

    /// Recalcula campos derivados que dependen de otros campos.
    /// Debe llamarse después de aplicar los argumentos de línea de comandos.
    #[inline]
    pub fn fix_derived(&mut self) {
        if self.n_heads > 0 {
            self.head_dim = self.d_model / self.n_heads;
        }
    }

    /// Número total de tokens por batch (batch_size × seq_len).
    #[inline]
    pub fn batch_tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }

    /// Valida que la configuración sea coherente antes de inicializar Metal.
    /// Errores aquí previenen pánicos en Objective-C que son difíciles de depurar.
    pub fn validate(&self) -> Result<()> {
        if self.corpus_paths.is_empty() {
            bail!("[config] --corpus no especificado. Usa --corpus=ruta.jsonl");
        }
        if self.vocab_path.is_empty() {
            bail!("[config] --vocab no especificado.");
        }
        if self.ckpt_path.is_empty() {
            bail!("[config] --ckpt no especificado.");
        }
        if self.d_model == 0 {
            bail!("[config] d_model debe ser > 0");
        }
        if self.n_heads == 0 {
            bail!("[config] n_heads debe ser > 0");
        }
        if self.d_model % self.n_heads != 0 {
            bail!(
                "[config] d_model ({}) debe ser divisible entre n_heads ({})",
                self.d_model, self.n_heads
            );
        }
        if self.n_layers == 0 {
            bail!("[config] n_layers debe ser > 0");
        }
        if self.hidden_dim == 0 {
            bail!("[config] hidden_dim debe ser > 0");
        }
        if self.vocab_size == 0 {
            bail!("[config] vocab_size debe ser > 0");
        }
        if self.seq_len == 0 {
            bail!("[config] seq_len debe ser > 0");
        }
        if self.batch_size == 0 {
            bail!("[config] batch_size debe ser > 0");
        }
        if self.lr <= 0.0 || !self.lr.is_finite() {
            bail!("[config] lr debe ser > 0.0 y finito, recibido: {}", self.lr);
        }
        if self.grad_clip <= 0.0 {
            bail!("[config] grad_clip debe ser > 0.0");
        }
        if self.n_epochs == 0 {
            bail!("[config] n_epochs debe ser >= 1");
        }
        Ok(())
    }

    /// Imprime la configuración activa en consola con formato legible.
    pub fn print(&self) {
        let batch_tokens = self.batch_tokens();
        let total_params = self.estimate_params();

        println!("┌───");
        println!("│  ARK v1.3 — Configuración Activa");
        println!("├───");
        println!("│  RUTAS");
        println!("│    corpus:     {}", self.corpus_paths.join(", "));
        println!("│    vocab:      {}", self.vocab_path);
        println!("│    ckpt:       {}", self.ckpt_path);
        println!("├───");
        println!("│  ARQUITECTURA");
        println!("│    vocab_size: {} ", self.vocab_size);
        println!("│    d_model:    {} ", self.d_model);
        println!("│    n_heads:    {}  head_dim: {:>6} ", self.n_heads, self.head_dim);
        println!("│    n_layers:   {} ", self.n_layers);
        println!("│    hidden_dim: {} ", self.hidden_dim);
        println!("│    seq_len:    {} ", self.seq_len);
        println!("│    params:     {}M ", total_params as f32 / 1e6);
        println!("├───");
        println!("│  ENTRENAMIENTO ");
        println!("│    epochs:     {} ", self.n_epochs);
        println!("│    batch_size: {}  tokens: {:>8} ", self.batch_size, batch_tokens);
        println!("│    lr:         {} ", self.lr);
        println!("│    grad_clip:  {} ", self.grad_clip);
        println!("│    warmup:     {}  pasos ", self.warmup_steps);
        println!("│    precisión:  {} ", if self.use_fp16 { "FP16/FP32" } else { "FP32" });
        println!("├───");
        println!("│  ADAMW");
        println!("│    beta1={:.3}  beta2={:.3}  eps={:.0e}  wd={:.3}",
            self.beta1, self.beta2, self.adam_eps, self.weight_decay);
        println!("└───");
        println!();
    }

    /// Estimación de parámetros totales del modelo.
    fn estimate_params(&self) -> usize {
        let d = self.d_model;
        let h = self.hidden_dim;
        let v = self.vocab_size;
        let l = self.n_layers;
        // Embedding + gamma_final
        let embed = v * d + d;
        // Por capa: wq wk wv wo (d×d) + w1 w3 (d×h) + w2 (h×d) + g1 g2 (d)
        let per_layer = d * d * 4 + d * h * 3 + d * 2;
        embed + per_layer * l
    }
}
