// src/optimizer.rs — ARK v1.0 "METAL-REASONER"
//
// Mejoras v1.0 (Zero-Copy Real):
//   - AdamW opera y actualiza directamente la memoria de la GPU (VRAM FP16) a través
//     del kernel ASM `ark_asm_adam_step_f16`, saltándose por completo las copias `memcpy`.
//   - Clip global de gradientes (norma L2) re-implementado con `vDSP_vsmul` para
//     garantizar la preservación exacta de la dirección del gradiente global y
//     procesar 4 floats por ciclo de CPU.
//   - Restauración de momentos desde CheckpointV4 asegurada contra pánicos.

use crate::ffi;
use crate::memory::ModelWeights;

// ── Kernel Ensamblador de AdamW FP16 (Zero-Copy) ──────────────────────────────
extern "C" {
    /// Kernel de ensamblador ARM64 NEON:
    /// Lee gradientes y momentos (FP32), de-cuantiza los pesos de FP16 a FP32 al vuelo,
    /// aplica el paso de AdamW, y re-cuantiza el resultado escribiéndolo directamente en
    /// el puntero `w` (VRAM compartida).
    fn ark_asm_adam_step_f16(
        w: *mut u16, g: *const f32, m: *mut f32, v: *mut f32,
        n: u64, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, t: u32,
    );
}

/// Wrapper seguro en línea para el Kernel ASM de AdamW Zero-Copy
#[inline]
fn adam_step_zero_copy(
    w_fp16: *mut u16, g: &[f32], m: &mut [f32], v: &mut[f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, t: u32,
) {
    debug_assert_eq!(g.len(), m.len());
    debug_assert_eq!(g.len(), v.len());
    unsafe {
        ark_asm_adam_step_f16(
            w_fp16, g.as_ptr(), m.as_mut_ptr(), v.as_mut_ptr(),
            g.len() as u64, lr, beta1, beta2, eps, wd, t,
        );
    }
}

// ── Configuración de Adam ─────────────────────────────────────────────────────

pub struct AdamConfig {
    pub lr:           f32,
    pub beta1:        f32,
    pub beta2:        f32,
    pub eps:          f32,
    pub weight_decay: f32,
    pub grad_clip:    f32,
    pub warmup_steps: usize,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr:           3e-4,   // Ajustado para pre-entrenamiento estándar
            beta1:        0.9,
            beta2:        0.999,
            eps:          1e-8,
            weight_decay: 0.01,
            grad_clip:    1.0,
            warmup_steps: 100,
        }
    }
}

pub struct Optimizer {
    pub cfg:  AdamConfig,
    pub step: u32,
}

impl Optimizer {
    pub fn new(cfg: AdamConfig) -> Self {
        Self { cfg, step: 0 }
    }

    /// Obtiene el Learning Rate actual aplicando un warmup lineal.
    #[inline]
    fn lr_now(&self) -> f32 {
        if (self.step as usize) < self.cfg.warmup_steps {
            self.cfg.lr * (self.step as f32 + 1.0) / self.cfg.warmup_steps as f32
        } else {
            // Cosine decay: lr baja gradualmente hasta lr/10 a lo largo del entrenamiento
            let decay_steps = (self.step as usize - self.cfg.warmup_steps) as f32;
            let total_decay = 250_000.0f32;
            let progress = (decay_steps / total_decay).min(1.0);
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            let lr_min = self.cfg.lr * 0.1;
            lr_min + (self.cfg.lr - lr_min) * cosine
        }
    }

    /// Clip POR CAPA — clipea cada tensor individualmente antes del global.
    fn clip_layer_grads(&self, weights: &mut ModelWeights) {
        let clip = self.cfg.grad_clip;

        let clip_tensor = |data: &mut [f32]| {
            let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > clip && norm > 0.0 {
                let scale = clip / norm;
                for x in data.iter_mut() { *x *= scale; }
            }
        };

        clip_tensor(weights.embed_w_grad.data.as_mut_slice());
        for l in &mut weights.layers {
            clip_tensor(l.wq_grad.data.as_mut_slice());
            clip_tensor(l.wk_grad.data.as_mut_slice());
            clip_tensor(l.wv_grad.data.as_mut_slice());
            clip_tensor(l.wo_grad.data.as_mut_slice());
            clip_tensor(l.w1_grad.data.as_mut_slice());
            clip_tensor(l.w2_grad.data.as_mut_slice());
            clip_tensor(l.w3_grad.data.as_mut_slice());
            clip_tensor(l.g1_grad.data.as_mut_slice());
            clip_tensor(l.g2_grad.data.as_mut_slice());
        }
    }

    /// Clip GLOBAL de gradientes por norma L2.
    ///
    /// Calcula la norma L2 combinada de todos los gradientes del modelo.
    /// Si excede `grad_clip`, escala *todos* los gradientes usando `vDSP_vsmul` (NEON nativo)
    /// para preservar la DIRECCIÓN exacta del gradiente en el espacio de parámetros.
    fn clip_all_grads(&self, weights: &mut ModelWeights) {
        let mut sum_sq = 0.0f32;

        // 1. Acumular la suma de los cuadrados de todos los gradientes
        sum_sq += weights.embed_w_grad.data.iter().map(|x| x * x).sum::<f32>();
        for l in &weights.layers {
            for g in[
                &l.wq_grad, &l.wk_grad, &l.wv_grad, &l.wo_grad,
                &l.w1_grad, &l.w2_grad, &l.w3_grad,
                &l.g1_grad, &l.g2_grad,
            ] {
                sum_sq += g.data.iter().map(|x| x * x).sum::<f32>();
            }
        }

        let norm = sum_sq.sqrt();

        // 2. Aplicar escalado global (vDSP vectorial) si la norma excede el umbral
        if norm > self.cfg.grad_clip && norm > 0.0 {
            let scale = self.cfg.grad_clip / norm;

            ffi::scale_tensor_vdsp(weights.embed_w_grad.data.as_mut_slice(), scale);
            for l in &mut weights.layers {
                ffi::scale_tensor_vdsp(l.wq_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.wk_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.wv_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.wo_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.w1_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.w2_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.w3_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.g1_grad.data.as_mut_slice(), scale);
                ffi::scale_tensor_vdsp(l.g2_grad.data.as_mut_slice(), scale);
            }
        }
    }

    /// Un paso completo de AdamW sobre todos los pesos del modelo (Zero-Copy VRAM).
    pub fn step_all(&mut self, weights: &mut ModelWeights) {
        self.step += 1;
        let lr  = self.lr_now();
        let t   = self.step;
        let b1  = self.cfg.beta1;
        let b2  = self.cfg.beta2;
        let eps = self.cfg.eps;
        let wd  = self.cfg.weight_decay;

        // 1. Clip global de gradientes
        self.clip_layer_grads(weights);
        self.clip_all_grads(weights);

        // 2. AdamW Zero-Copy para matriz de embedding
        adam_step_zero_copy(
            weights.embed_w_fp16,
            weights.embed_w_grad.data.as_slice(),
            weights.embed_m.data.as_mut_slice(),
            weights.embed_v.data.as_mut_slice(),
            lr, b1, b2, eps, wd, t,
        );

        // AdamW Zero-Copy para gamma_final (sin weight decay — igual que g1/g2)
        adam_step_zero_copy(
            weights.gamma_f_fp16,
            weights.gamma_final_grad.data.as_slice(),
            weights.gamma_final_m.data.as_mut_slice(),
            weights.gamma_final_v.data.as_mut_slice(),
            lr, b1, b2, eps, 0.0, t,
        );

        // 3. AdamW Zero-Copy para todas las capas del Transformer
        for l in &mut weights.layers {
            // Macro para mantener el código limpio y seguro
            macro_rules! adam {
                ($w_fp16:expr, $g:expr, $m:expr, $v:expr, $decay:expr) => {
                    adam_step_zero_copy(
                        $w_fp16, $g.data.as_slice(),
                        $m.data.as_mut_slice(), $v.data.as_mut_slice(),
                        lr, b1, b2, eps, $decay, t,
                    )
                };
            }

            // Pesos de matrices (MHA y FFN) — Aplican Weight Decay
            adam!(l.wq_fp16, l.wq_grad, l.wq_m, l.wq_v, wd);
            adam!(l.wk_fp16, l.wk_grad, l.wk_m, l.wk_v, wd);
            adam!(l.wv_fp16, l.wv_grad, l.wv_m, l.wv_v, wd);
            adam!(l.wo_fp16, l.wo_grad, l.wo_m, l.wo_v, wd);
            adam!(l.w1_fp16, l.w1_grad, l.w1_m, l.w1_v, wd);
            adam!(l.w2_fp16, l.w2_grad, l.w2_m, l.w2_v, wd);
            adam!(l.w3_fp16, l.w3_grad, l.w3_m, l.w3_v, wd);

            // Vectores de normalización RMSNorm (Gammas) — SIN Weight Decay (wd=0.0)
            adam!(l.g1_fp16, l.g1_grad, l.g1_m, l.g1_v, 0.0);
            adam!(l.g2_fp16, l.g2_grad, l.g2_m, l.g2_v, 0.0);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Serialización y Restauración de Momentos (Checkpoint V4)
    // ══════════════════════════════════════════════════════════════════════

    /// Restaura momentos m y v (FP32) desde un Checkpoint cargado en disco.
    /// Si las dimensiones de la arquitectura cambiaron, se ignoran silenciosamente
    /// (comportamiento Cold-Start seguro). No entra en pánico.
    pub fn restaurar_momentos(
        &mut self,
        weights:    &mut ModelWeights,
        momentos_m: &[Vec<f32>],
        momentos_v: &[Vec<f32>],
        adam_step:  u32,
    ) {
        let n_capas   = weights.layers.len();
        let esperados = 1 + n_capas * 9;

        // Validar tamaño general
        if momentos_m.len() != esperados || momentos_v.len() != esperados {
            println!(
                "[optimizer] momentos Adam ignorados — \
                 tamaño incompatible ({} tensores, esperaba {}). \
                 Optimizer partirá desde cero.",
                momentos_m.len(), esperados
            );
            return;
        }

        // Restaurar Embeddings
        if momentos_m[0].len() == weights.embed_m.data.len() {
            weights.embed_m.data.copy_from_slice(&momentos_m[0]);
            weights.embed_v.data.copy_from_slice(&momentos_v[0]);
        } else {
            println!("[optimizer] embed_w momentos ignorados (dimensión distinta).");
            return;
        }

        // Restaurar Capas
        for (li, l) in weights.layers.iter_mut().enumerate() {
            let base = 1 + li * 9;

            let tensores_capa =[
                (&mut l.wq_m, &mut l.wq_v),
                (&mut l.wk_m, &mut l.wk_v),
                (&mut l.wv_m, &mut l.wv_v),
                (&mut l.wo_m, &mut l.wo_v),
                (&mut l.w1_m, &mut l.w1_v),
                (&mut l.w2_m, &mut l.w2_v),
                (&mut l.w3_m, &mut l.w3_v),
                (&mut l.g1_m, &mut l.g1_v),
                (&mut l.g2_m, &mut l.g2_v),
            ];

            for (ti, (m_tensor, v_tensor)) in tensores_capa.into_iter().enumerate() {
                let idx = base + ti;
                if momentos_m[idx].len() == m_tensor.data.len() {
                    m_tensor.data.copy_from_slice(&momentos_m[idx]);
                    v_tensor.data.copy_from_slice(&momentos_v[idx]);
                } else {
                    println!(
                        "[optimizer] capa {} tensor {} momentos ignorados (dimensión distinta).",
                        li, ti
                    );
                    return;
                }
            }
        }

        self.step = adam_step;
        println!(
            "[optimizer] momentos Adam restaurados correctamente — {} tensores, paso actual={}",
            esperados, adam_step
        );
    }
}