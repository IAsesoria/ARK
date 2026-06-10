use crate::config::Config;
use crate::ffi;
use crate::io::{CorpusStream, Checkpoint, CheckpointV3};
use crate::memory::ModelWeights;
use crate::optimizer::{Optimizer, AdamConfig};


// ── Constantes de Loss Scaling ────────────────────────────────────────────────

const LOSS_SCALE_INIT: f32 = 256.0;

const LOSS_SCALE_MIN:  f32 = 1.0;
const SCALE_UP_EVERY:  u32 = 2000;

// ── Buffers temporales para Backward (FP32) ───────────────────────────────────

/// Estructura temporal pre-asignada que almacena la versión FP32 de los pesos.
/// Se usa exclusivamente durante el backward pass en CPU (Accelerate/AMX) que
/// requiere matemática en precisión simple.
struct TempLayerF32 {
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
    w1: Vec<f32>, w2: Vec<f32>, w3: Vec<f32>,
    g1: Vec<f32>, g2: Vec<f32>,
}

impl TempLayerF32 {
    fn new(d: usize, h: usize) -> Self {
        Self {
            wq: vec![0.0f32; d * d], wk: vec![0.0f32; d * d],
            wv: vec![0.0f32; d * d], wo: vec![0.0f32; d * d],
            w1: vec![0.0f32; d * h], w2: vec![0.0f32; h * d], w3: vec![0.0f32; d * h],
            g1: vec![0.0f32; d],     g2: vec![0.0f32; d],
        }
    }
}

// ── Trainer ───────────────────────────────────────────────────────────────────

pub struct Trainer {
    pub cfg:     Config,
    pub weights: ModelWeights,
    pub opt:     Optimizer,
    ckpt_every: u64,

    // Buffers pre-allocados — cero allocations en el hot path del entrenamiento
    logits:      Vec<f32>,
    logits_grad: Vec<f32>,
    targets:     Vec<u32>,

    // Buffers temporales FP32 para backward (M1 AMX) y forward (LM Head / Embed)
    temp_embed_w: Vec<f32>,
    temp_layers:  Vec<TempLayerF32>,

    // Estado del loss scaler dinámico
    loss_scale:    f32,
    clean_steps:   u32,

    consecutive_nans: u32,
    skipped_steps: u64,
}

impl Trainer {
    pub fn new(cfg: Config) -> anyhow::Result<Self> {
        cfg.validate()?;

        // 1. Inicializar Hardware y compilar grafos MPSGraph PRIMERO
        // CRÍTICO: ark_mps_build_graphs() debe ejecutarse ANTES de ModelWeights::new()
        // porque get_embed_ptr / get_weight_ptrs devuelven punteros a los MTLBuffer
        // que solo existen después de que los grafos están compilados.
        // Orden incorrecto anterior → punteros nulos → bus error en el primer dequant.
        ffi::gpu_init(cfg.n_layers, cfg.d_model, cfg.hidden_dim)?;
        ffi::gpu_set_heads(cfg.n_heads);

        let batch_tokens = cfg.batch_tokens();

        // 2. Pre-compilar los grafos MPSGraph ANTES de pedir los punteros a VRAM
        ffi::gpu_build_graphs(batch_tokens, cfg.vocab_size, cfg.batch_size)?;
        println!("[ark] MPSGraph grafos compilados (batch_tokens={} vocab={})",
                 batch_tokens, cfg.vocab_size);

        // 3. Ahora sí los MTLBuffer existen — ModelWeights obtiene punteros válidos
        let weights = ModelWeights::new(
            cfg.n_layers, cfg.vocab_size, cfg.d_model, cfg.hidden_dim
        );

        let opt = Optimizer::new(AdamConfig {
            lr:           cfg.lr,
            beta1:        cfg.beta1,
            beta2:        cfg.beta2,
            eps:          cfg.adam_eps,
            weight_decay: cfg.weight_decay,
            grad_clip:    cfg.grad_clip,
            warmup_steps: cfg.warmup_steps,
        });

        // 4. Pre-allocar buffers lógicos CPU
        let logits      = vec![0.0f32; batch_tokens * cfg.vocab_size];
        let logits_grad = vec![0.0f32; batch_tokens * cfg.vocab_size];
        let targets     = vec![0u32;  batch_tokens];

        let temp_embed_w = vec![0.0f32; cfg.vocab_size * cfg.d_model];
        let temp_layers  = (0..cfg.n_layers)
            .map(|_| TempLayerF32::new(cfg.d_model, cfg.hidden_dim))
            .collect();

        println!("[ark] modelo:   {} params  ({:.1}M)",
            weights.n_params(),
            weights.n_params() as f32 / 1_000_000.0);
        println!("[ark] buffers:  logits {:.1} MB  logits_grad {:.1} MB  targets {} B",
            logits.len() as f32 * 4.0 / 1e6,
            logits_grad.len() as f32 * 4.0 / 1e6,
            targets.len() * 4);
        println!("[ark] AMP:      loss_scale_init={:.0}  max={:.0}  up_every={}",
            LOSS_SCALE_INIT, cfg.loss_scale_max, SCALE_UP_EVERY);

        Ok(Self {
            cfg, weights, opt,
            logits, logits_grad, targets,
            temp_embed_w, temp_layers,
            loss_scale:    LOSS_SCALE_INIT,
            clean_steps:   0,

            consecutive_nans: 0,
            skipped_steps: 0,
            ckpt_every: 500,
        })
    }

    // ── Escalado de gradientes ────────────────────────────────────────────────

    /// Multiplica todos los gradientes por `factor`.
    /// Utiliza `vDSP_vsmul` (NEON nativo) para procesar 4 floats por ciclo en Apple Silicon.
    fn scale_grads(&mut self, factor: f32) {
        ffi::scale_tensor_vdsp(self.weights.embed_w_grad.data.as_mut_slice(), factor);
        for l in &mut self.weights.layers {
            ffi::scale_tensor_vdsp(l.wq_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.wk_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.wv_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.wo_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.w1_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.w2_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.w3_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.g1_grad.data.as_mut_slice(), factor);
            ffi::scale_tensor_vdsp(l.g2_grad.data.as_mut_slice(), factor);
        }
    }

    // ── Train step ───────────────────────────────────────────────────────────

    /// Un paso completo de entrenamiento (Forward -> Backward -> Optimizer).
    ///
    /// Retorna:
    ///   - Ok(Some(loss)) si el paso fue válido.
    ///   - Ok(None)       si fue descartado por overflow/NaN en gradientes.
    fn train_step(&mut self, tokens: &[u32]) -> anyhow::Result<Option<f32>> {
        let cfg = &self.cfg;
        let bs  = cfg.batch_tokens() as i32;
        let v   = cfg.vocab_size as i32;
        let d   = cfg.d_model as i32;
        let h   = cfg.hidden_dim as i32;

        // ── 0. DE-CUANTIZACIÓN AL VUELO (FP16 -> FP32) ───────────────────────
        // Desempaquetamos los pesos desde la GPU (VRAM compartida FP16)
        // hacia nuestros buffers temporales FP32 usando kernels de ensamblador.
        unsafe {
            ffi::ark_dequant_f16_to_f32( // <-- Modificado para llamar vía ffi::
                self.weights.embed_w_fp16, 
                self.temp_embed_w.as_mut_ptr(), 
                (cfg.vocab_size * cfg.d_model) as u64
            );

            for l in 0..cfg.n_layers {
                let gpu_l = &self.weights.layers[l];
                let tmp_l = &mut self.temp_layers[l];
                
                ffi::ark_dequant_f16_to_f32(gpu_l.wq_fp16, tmp_l.wq.as_mut_ptr(), (d * d) as u64); // <-- Modificado
                ffi::ark_dequant_f16_to_f32(gpu_l.wk_fp16, tmp_l.wk.as_mut_ptr(), (d * d) as u64); // ...
                ffi::ark_dequant_f16_to_f32(gpu_l.wv_fp16, tmp_l.wv.as_mut_ptr(), (d * d) as u64);
                ffi::ark_dequant_f16_to_f32(gpu_l.wo_fp16, tmp_l.wo.as_mut_ptr(), (d * d) as u64);
                ffi::ark_dequant_f16_to_f32(gpu_l.w1_fp16, tmp_l.w1.as_mut_ptr(), (d * h) as u64);
                ffi::ark_dequant_f16_to_f32(gpu_l.w3_fp16, tmp_l.w3.as_mut_ptr(), (d * h) as u64);
                ffi::ark_dequant_f16_to_f32(gpu_l.w2_fp16, tmp_l.w2.as_mut_ptr(), (h * d) as u64);
                ffi::ark_dequant_f16_to_f32(gpu_l.g1_fp16, tmp_l.g1.as_mut_ptr(), d as u64);
                ffi::ark_dequant_f16_to_f32(gpu_l.g2_fp16, tmp_l.g2.as_mut_ptr(), d as u64);
            }
        }

        // ── 1. FORWARD (GPU FP16, Zero-Copy Layers) ───────────────────────────
        let fwd_ok = unsafe {
            ffi::ark_mps_forward(
                tokens.as_ptr(),
                self.temp_embed_w.as_ptr(), // Embedding dequantizado
                self.logits.as_mut_ptr(),
                bs, v, d,
            )
        };
        anyhow::ensure!(fwd_ok, "[fwd] fallo catastrófico en ark_mps_forward");

        // ── 2. TARGETS (Prevención de Alucinación de Bucle) ───────────────────
        {
            let n = tokens.len();
            debug_assert_eq!(self.targets.len(), n);
            if n > 1 {
                self.targets[..n-1].copy_from_slice(&tokens[1..]);
            }
            // El último token predice a sí mismo para no hacer wrap-around.
            self.targets[n-1] = tokens[n-1];
        }

        // ── 3. CROSS-ENTROPY Y GRADIENTE DE LOGITS ────────────────────────────
        let mut loss = 0.0f32;
        let ce_ok = unsafe {
            ffi::ark_mps_cross_entropy(
                self.logits.as_ptr(),
                self.targets.as_ptr(),
                self.logits_grad.as_mut_ptr(),
                &mut loss as *mut f32,
                bs, v,
            )
        };
        anyhow::ensure!(ce_ok, "[ce] fallo en ark_mps_cross_entropy");

        if !loss.is_finite() || loss < 0.0 {
            anyhow::bail!("[loss] valor inválido: {} — revisar cross-entropy", loss);
        }

        // ── 3b. CHECK logits_grad ANTES de escalar ───────────────────────────
        {
            let logits_finite = self.logits_grad.iter().all(|x| x.is_finite());
            if !logits_finite {
                self.loss_scale = (self.loss_scale * 0.5).max(LOSS_SCALE_MIN);
                self.clean_steps = 0;
                self.skipped_steps += 1;
                self.consecutive_nans += 1;
                if self.consecutive_nans > 200 {
                    return Err(anyhow::anyhow!("early_stop_nan"));
                }
                println!("[amp] paso descartado (NaN/Inf en logits_grad) — loss_scale → {:.0}", self.loss_scale);
                return Ok(None);
            }
        }

        // ── 4. [AMP] LOSS SCALING ─────────────────────────────────────────────
        let inv_scale = 1.0 / self.loss_scale;
        for x in self.logits_grad.iter_mut() { *x *= self.loss_scale; }

        // ── 5. BACKWARD (CPU FP32 + Accelerate/AMX) ───────────────────────────
        self.weights.zero_all_grads();

        // Preparamos los arrays de punteros para el FFI
        let wq_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.wq.as_ptr()).collect();
        let wk_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.wk.as_ptr()).collect();
        let wv_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.wv.as_ptr()).collect();
        let wo_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.wo.as_ptr()).collect();
        let w1_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.w1.as_ptr()).collect();
        let w2_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.w2.as_ptr()).collect();
        let w3_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.w3.as_ptr()).collect();
        let g1_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.g1.as_ptr()).collect();
        let g2_ptrs: Vec<*const f32> = self.temp_layers.iter().map(|l| l.g2.as_ptr()).collect();

        let wq_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.wq_grad.as_mut_ptr()).collect();
        let wk_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.wk_grad.as_mut_ptr()).collect();
        let wv_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.wv_grad.as_mut_ptr()).collect();
        let wo_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.wo_grad.as_mut_ptr()).collect();
        let w1_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.w1_grad.as_mut_ptr()).collect();
        let w2_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.w2_grad.as_mut_ptr()).collect();
        let w3_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.w3_grad.as_mut_ptr()).collect();
        let g1_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.g1_grad.as_mut_ptr()).collect();
        let g2_g: Vec<*mut f32> = self.weights.layers.iter_mut().map(|l| l.g2_grad.as_mut_ptr()).collect();

        let bwd_ok = unsafe {
            ffi::ark_mps_backward_layers(
                tokens.as_ptr(),
                self.logits_grad.as_ptr(),
                self.temp_embed_w.as_ptr(),
                self.weights.embed_w_grad.as_mut_ptr(),
                self.weights.gamma_final_grad.as_mut_ptr(),
                wq_ptrs.as_ptr(), wk_ptrs.as_ptr(), wv_ptrs.as_ptr(), wo_ptrs.as_ptr(),
                w1_ptrs.as_ptr(), w2_ptrs.as_ptr(), w3_ptrs.as_ptr(),
                g1_ptrs.as_ptr(), g2_ptrs.as_ptr(),
                wq_g.as_ptr(), wk_g.as_ptr(), wv_g.as_ptr(), wo_g.as_ptr(),
                w1_g.as_ptr(), w2_g.as_ptr(), w3_g.as_ptr(),
                g1_g.as_ptr(), g2_g.as_ptr(),
                bs, v, d, h, cfg.n_layers as i32
            )
        };
        anyhow::ensure!(bwd_ok, "[bwd] fallo en backward_layers");

        // ── 6. NaN CHECK ──────────────────────────────────────────────────────
        // Solo verificamos que sean finitos; el Optimizer se encargará del Clip Global
        let grad_ok = self.weights.all_grad_slices_mut().all(|x| x.is_finite());

        if !grad_ok {
            self.loss_scale = (self.loss_scale * 0.5).max(LOSS_SCALE_MIN);
            self.clean_steps = 0;
            self.skipped_steps += 1;
            self.consecutive_nans += 1;
            if self.consecutive_nans > 200 {
                println!("[ark] EARLY STOP — {} NaN consecutivos, checkpoint intacto", self.consecutive_nans);
                return Err(anyhow::anyhow!("early_stop_nan"));
            }
            self.weights.zero_all_grads();
            println!("[amp] paso descartado (NaN/Inf en gradientes) — loss_scale → {:.0}", self.loss_scale);
            return Ok(None);
        }

        // ── 7. [AMP] DESESCALAR GRADIENTES ────────────────────────────────────
        self.scale_grads(inv_scale);

        // ── 8. ADAM ZERO-COPY (Actualiza FP16 Directo) + CLIP GLOBAL ──────────
        // Llama a adam_step_f16 en el kernel de ensamblador, leyendo los gradientes
        // y momentos (FP32) y reescribiendo los pesos directamente en el buffer
        // de memoria unificada de la GPU.
        self.opt.step_all(&mut self.weights);

        // ── 9. [AMP] ACTUALIZAR LOSS SCALE ────────────────────────────────────
        self.clean_steps += 1;
        self.consecutive_nans = 0;
        if self.clean_steps >= SCALE_UP_EVERY {
            let nueva = (self.loss_scale * 2.0).min(self.cfg.loss_scale_max);
            if nueva > self.loss_scale {
                println!("[amp] loss_scale: {:.0} → {:.0}", self.loss_scale, nueva);
                self.loss_scale = nueva;
            }
            self.clean_steps = 0;
        }

        Ok(Some(loss))
    }

    // ── Loop principal ────────────────────────────────────────────────────────

    pub fn run(&mut self) -> anyhow::Result<()> {
        let rutas: Vec<&str> = self.cfg.corpus_paths.iter().map(|s| s.as_str()).collect();
        let mut stream = CorpusStream::abrir(&rutas, &self.cfg.vocab_path)?;
        let mut global_step: u64 = 0;

        // Cargar checkpoint y sincronizar hacia la VRAM FP16
        match CheckpointV3::load(&self.cfg.ckpt_path) {
            Ok(ckpt3) => {
                let expected_pesos = 1 + self.cfg.n_layers * 9;
                if ckpt3.pesos.len() == expected_pesos || ckpt3.pesos.len() == expected_pesos + 1 {
                    
                    // Empaquetar los pesos FP32 leídos para mandarlos a `sync_weights_to_gpu`
                    let mut layer_slices_f32 = Vec::new();
                    for li in 0..self.cfg.n_layers {
                        let base = 1 + li * 9;
                        layer_slices_f32.push(vec![
                            ckpt3.pesos[base + 0].clone(),
                            ckpt3.pesos[base + 1].clone(),
                            ckpt3.pesos[base + 2].clone(),
                            ckpt3.pesos[base + 3].clone(),
                            ckpt3.pesos[base + 4].clone(),
                            ckpt3.pesos[base + 5].clone(),
                            ckpt3.pesos[base + 6].clone(),
                            ckpt3.pesos[base + 7].clone(),
                            ckpt3.pesos[base + 8].clone(),
                        ]);
                    }

                    self.weights.sync_weights_to_gpu(&ckpt3.pesos[0], &layer_slices_f32);
                    global_step = ckpt3.global_step;

                    // Si el checkpoint cuenta con los pesos de gamma_final (índice 271), sincronizarlos
                    if ckpt3.pesos.len() == expected_pesos + 1 {
                        unsafe {
                            crate::ffi::ark_quant_f32_to_f16(
                                ckpt3.pesos[expected_pesos].as_ptr(),
                                self.weights.gamma_f_fp16,
                                self.cfg.d_model as u64,
                            );
                        }
                    }

                    if ckpt3.tiene_momentos {
                        self.opt.restaurar_momentos(
                            &mut self.weights,
                            &ckpt3.momentos_m,
                            &ckpt3.momentos_v,
                            ckpt3.adam_step as u32,
                        );
                        println!("[ark] checkpoint v4 restaurado — pesos + momentos Adam, paso {}", global_step);
                    } else {
                        println!("[ark] checkpoint v4 restaurado — sin momentos Adam (spike ~200-500 pasos), paso {}", global_step);
                    }
                } else {
                    println!("[ark] checkpoint v4 ignorado (arquitectura distinta: esperaba {} tensores, tiene {})",
                              expected_pesos, ckpt3.pesos.len());
                    self.cargar_checkpoint_legado(&mut global_step);
                }
            }
            Err(_) => {
                self.cargar_checkpoint_legado(&mut global_step);
            }
        }

        let batch_tokens = self.cfg.batch_tokens();
        let lineas_corpus = CorpusStream::contar_lineas_corpus(&self.cfg.corpus_paths);

        print!("[ark] muestreando tokens/doc (1000 docs)... ");
        let tokens_promedio_por_doc = stream.muestrear_tokens_promedio_real(1000);
        println!("{} tokens/doc (promedio real)", tokens_promedio_por_doc);

        let tokens_totales_estimados = lineas_corpus * tokens_promedio_por_doc;
        let steps_estimados = tokens_totales_estimados
            .saturating_div(batch_tokens as u64)
            .max(1);

        println!("[ark] corpus:   {} docs  ×  {} tok/doc  =  ~{}M tokens",
                 lineas_corpus,
                 tokens_promedio_por_doc,
                 tokens_totales_estimados / 1_000_000);
        println!("[ark] steps:    ~{} por época  (batch_tokens={})",
                 steps_estimados, batch_tokens);

       // Sin estimación de tiempo — varía según seq/batch/hardware

        // ── Loop de épocas ────────────────────────────────────────────────────
        for epoch in 0..self.cfg.n_epochs {
            stream.reiniciar()?;

            let mut loss_sum  = 0.0f64;
            let mut steps     = 0usize;
            let mut skips_ep  = 0u64;

            self.ckpt_every = if epoch == 0 { 500 } else { 1000 };

            println!("[ark] época {}/{} — iniciando[AMP scale={:.0}  ckpt_every={}]",
                     epoch + 1, self.cfg.n_epochs, self.loss_scale, self.ckpt_every);

            loop {
                match stream.next_batch(batch_tokens)? {
                    None => break,
                    Some(tokens) => {
                        match self.train_step(&tokens)? {
                            None => {
                                skips_ep += 1;
                            }
                            Some(loss) => {
                                loss_sum    += loss as f64;
                                steps       += 1;
                                global_step += 1;

                                if steps % 100 == 0 || steps == 1 {
                                    let ppl = (loss as f64).exp();
                                    println!(
                                        "[ep{:>2}  paso{:>7}  g{:>8}]  \
                                     loss={:.4}  ppl={:.1}  scale={:.0}  skips={}",
                                    epoch + 1, steps, global_step,
                                    loss, ppl, self.loss_scale, self.skipped_steps
                                    );
                                }

                                if global_step % self.ckpt_every == 0 {
                                    self.guardar_checkpoint(global_step)?;
                                }
                            }
                        }
                    }
                }
            }

            let loss_medio = if steps > 0 { loss_sum / steps as f64 } else { 0.0 };
            let ppl_medio  = loss_medio.exp();
            println!(
                "\n[ÉPOCA {}]  loss={:.4}  ppl={:.1}  pasos={}  tokens={}M  skips={}",
                epoch + 1, loss_medio, ppl_medio, steps,
                (steps * batch_tokens) / 1_000_000, skips_ep
            );

            stream.stats();
        }

        self.guardar_checkpoint(global_step)?;

        println!("[ark] entrenamiento completo.  pasos_totales={}  skips_totales={}",
                 global_step, self.skipped_steps);
        Ok(())
    }

    // ── Checkpoint ───────────────────────────────────────────────────────────

    fn cargar_checkpoint_legado(&mut self, global_step: &mut u64) {
        match Checkpoint::load(&self.cfg.ckpt_path) {
            Err(_) => {
                println!("[ark] sin checkpoint previo — iniciando pesos Xavier desde cero");
            }
            Ok(ckpt) => {
                if ckpt.is_full() {
                    let expected = 1 + self.cfg.n_layers * 9;
                    if ckpt.tensors.len() == expected {
                        
                        let mut layer_slices_f32 = Vec::new();
                        for li in 0..self.cfg.n_layers {
                            let base = 1 + li * 9;
                            layer_slices_f32.push(vec![
                                ckpt.tensors[base + 0].clone(),
                                ckpt.tensors[base + 1].clone(),
                                ckpt.tensors[base + 2].clone(),
                                ckpt.tensors[base + 3].clone(),
                                ckpt.tensors[base + 4].clone(),
                                ckpt.tensors[base + 5].clone(),
                                ckpt.tensors[base + 6].clone(),
                                ckpt.tensors[base + 7].clone(),
                                ckpt.tensors[base + 8].clone(),
                            ]);
                        }
                        
                        self.weights.sync_weights_to_gpu(&ckpt.tensors[0], &layer_slices_f32);
                        *global_step = ckpt.global_step;
                        println!("[ark] checkpoint v2 restaurado — paso {}", global_step);
                    } else {
                        println!("[ark] checkpoint v2 ignorado (arquitectura distinta: esperaba {} tensores, tiene {})", expected, ckpt.tensors.len());
                    }
                } else if !ckpt.tensors.is_empty() && ckpt.tensors[0].len() == (self.cfg.vocab_size * self.cfg.d_model) {
                    let mut dummy_layers = Vec::new();
                    for _ in 0..self.cfg.n_layers {
                        dummy_layers.push(vec![
                            vec![0.0; self.cfg.d_model * self.cfg.d_model], vec![0.0; self.cfg.d_model * self.cfg.d_model],
                            vec![0.0; self.cfg.d_model * self.cfg.d_model], vec![0.0; self.cfg.d_model * self.cfg.d_model],
                            vec![0.0; self.cfg.d_model * self.cfg.hidden_dim], vec![0.0; self.cfg.hidden_dim * self.cfg.d_model],
                            vec![0.0; self.cfg.d_model * self.cfg.hidden_dim], vec![1.0; self.cfg.d_model],
                            vec![1.0; self.cfg.d_model],
                        ]);
                    }
                    self.weights.sync_weights_to_gpu(&ckpt.tensors[0], &dummy_layers);
                    *global_step = ckpt.global_step;
                    println!("[ark] checkpoint v1 (legacy) restaurado — paso {}", global_step);
                } else {
                    println!("[ark] checkpoint v1 ignorado (dimensiones distintas)");
                }
            }
        }
    }

    fn guardar_checkpoint(&self, step: u64) -> anyhow::Result<()> {
        let v = self.cfg.vocab_size;
        let d = self.cfg.d_model;
        let h = self.cfg.hidden_dim;

        // Construir slices de los pesos directamente desde la VRAM FP16
        let embed_w_f16 = unsafe { std::slice::from_raw_parts(self.weights.embed_w_fp16, v * d) };
        let gamma_f_f16 = unsafe { std::slice::from_raw_parts(self.weights.gamma_f_fp16, d) };

        let mut layer_w_f16 = Vec::new();
        for l in &self.weights.layers {
            unsafe {
                layer_w_f16.push(vec![
                    std::slice::from_raw_parts(l.wq_fp16, d * d),
                    std::slice::from_raw_parts(l.wk_fp16, d * d),
                    std::slice::from_raw_parts(l.wv_fp16, d * d),
                    std::slice::from_raw_parts(l.wo_fp16, d * d),
                    std::slice::from_raw_parts(l.w1_fp16, d * h),
                    std::slice::from_raw_parts(l.w2_fp16, h * d),
                    std::slice::from_raw_parts(l.w3_fp16, d * h),
                    std::slice::from_raw_parts(l.g1_fp16, d),
                    std::slice::from_raw_parts(l.g2_fp16, d),
                ]);
            }
        }

        // Momentos Adam en FP32
        let layer_m_slices: Vec<Vec<&[f32]>> = self.weights.layers.iter().map(|l| {
            vec![
                l.wq_m.data.as_slice(), l.wk_m.data.as_slice(),
                l.wv_m.data.as_slice(), l.wo_m.data.as_slice(),
                l.w1_m.data.as_slice(), l.w2_m.data.as_slice(),
                l.w3_m.data.as_slice(), l.g1_m.data.as_slice(),
                l.g2_m.data.as_slice(),
            ]
        }).collect();

        let layer_v_slices: Vec<Vec<&[f32]>> = self.weights.layers.iter().map(|l| {
            vec![
                l.wq_v.data.as_slice(), l.wk_v.data.as_slice(),
                l.wv_v.data.as_slice(), l.wo_v.data.as_slice(),
                l.w1_v.data.as_slice(), l.w2_v.data.as_slice(),
                l.w3_v.data.as_slice(), l.g1_v.data.as_slice(),
                l.g2_v.data.as_slice(),
            ]
        }).collect();

        let slot = (step / self.ckpt_every % 3) as u8;
        let base = self.cfg.ckpt_path.trim_end_matches(".bin");
        let base = if let Some(pos) = base.rfind("_rot") { &base[..pos] } else { base };
        let ckpt_rot = format!("{}_rot{}.bin", base, slot);
        CheckpointV3::save_fp16(
            &ckpt_rot,
            step,
            self.opt.step as u64,
            embed_w_f16,
            self.weights.embed_m.data.as_slice(),
            self.weights.embed_v.data.as_slice(),
            gamma_f_f16,
            self.weights.gamma_final_m.data.as_slice(),
            self.weights.gamma_final_v.data.as_slice(),
            &layer_w_f16,
            &layer_m_slices,
            &layer_v_slices,
        )?;

        println!("[checkpoint] v4 guardado — step={}  pesos FP16 nativos + momentos Adam FP32", step);
        Ok(())
    }
}

impl Drop for Trainer {
    fn drop(&mut self) {
        ffi::gpu_shutdown();
    }
}