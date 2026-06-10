// src/ffi.rs — ARK v1.0 "METAL-REASONER"
// ── Bridge Metal/MPSGraph (ark_mps_bridge.m) ─────────────────────────────────
extern "C" {
    pub fn ark_mps_init_with_config(n_layers: i32, d_model: i32, hidden_dim: i32) -> bool;

    #[allow(dead_code)]
    pub fn ark_mps_init(n_layers: i32) -> bool;

    pub fn ark_mps_set_heads(n_heads: i32);

    /// Pre-compila los grafos MPSGraph con dimensiones exactas de entrenamiento.
    /// Esto dispara la compilación MSL/AOT en background y elimina los bloqueos JIT.
    pub fn ark_mps_build_graphs(batch_tokens: i32, vocab: i32, batch_size: i32) -> bool;

    ///[v1.3] Obtiene los punteros directos al contenido de los MTLBuffer (Zero-Copy) de una capa.
    pub fn ark_mps_get_weight_ptrs(
        layer: i32,
        wq: *mut *mut std::ffi::c_void, wk: *mut *mut std::ffi::c_void,
        wv: *mut *mut std::ffi::c_void, wo: *mut *mut std::ffi::c_void,
        w1: *mut *mut std::ffi::c_void, w3: *mut *mut std::ffi::c_void,
        w2: *mut *mut std::ffi::c_void,
        g1: *mut *mut std::ffi::c_void, g2: *mut *mut std::ffi::c_void,
    );

    /// [v1.3] Obtiene los punteros directos al contenido del embedding y gamma_final (Zero-Copy).
    pub fn ark_mps_get_embed_ptr(
        embed_ptr: *mut *mut std::ffi::c_void,
        gamma_f_ptr: *mut *mut std::ffi::c_void,
        vocab_size: i32,
    );

    pub fn ark_mps_forward(
        tokens:    *const u32,
        embed_w:   *const f32,
        logits:    *mut   f32,
        batch_seq: i32,
        vocab:     i32,
        d_model:   i32,
    ) -> bool;

    pub fn ark_mps_cross_entropy(
        logits:      *const f32,
        targets:     *const u32,
        logits_grad: *mut   f32,
        loss_out:    *mut   f32,
        batch_seq:   i32,
        vocab:       i32,
    ) -> bool;

    /// Backward completo de capas procesado ultra-rápido en CPU vía Accelerate/AMX.
    pub fn ark_mps_backward_layers(
        tokens:            *const u32,    // para scatter-add dx_layers[0] → embed[token]
        logits_grad:       *const f32,
        embed_w:           *const f32,
        embed_w_grad:      *mut   f32,
        gamma_final_grad:  *mut   f32,    // FIX: gradiente de RMSNorm final
        layer_wq: *const *const f32, layer_wk: *const *const f32,
        layer_wv: *const *const f32, layer_wo: *const *const f32,
        layer_w1: *const *const f32, layer_w2: *const *const f32,
        layer_w3: *const *const f32,
        layer_g1: *const *const f32, layer_g2: *const *const f32,
        layer_wq_grad: *const *mut f32, layer_wk_grad: *const *mut f32,
        layer_wv_grad: *const *mut f32, layer_wo_grad: *const *mut f32,
        layer_w1_grad: *const *mut f32, layer_w2_grad: *const *mut f32,
        layer_w3_grad: *const *mut f32,
        layer_g1_grad: *const *mut f32, layer_g2_grad: *const *mut f32,
        batch_seq: i32, vocab: i32, d_model: i32, hidden_dim: i32, n_layers: i32,
    ) -> bool;

    pub fn ark_mps_shutdown();
}

// ── Kernels ASM NEON (ark_kernels.s) ─────────────────────────────────────────
extern "C" {
    #[allow(dead_code)]
    pub fn ark_asm_rmsnorm(x: *mut f32, gamma: *const f32, n_seq: i32, dim: i32);
    
    #[allow(dead_code)]
    pub fn ark_asm_softmax(x: *mut f32, n_seq: i32, dim: i32);
    
    #[allow(dead_code)]
    pub fn ark_asm_cross_entropy(
        logits: *const f32, targets: *const u32,
        loss: *mut f32, n_seq: i32, vocab: i32,
    );

    ///[v1.3] Cuantización ultra-rápida NEON de FP32 a FP16
    pub fn ark_quant_f32_to_f16(src: *const f32, dst: *mut u16, n: u64);

    /// [v1.3] De-cuantización ultra-rápida NEON de FP16 a FP32 (usada al vuelo para el backward)
    #[allow(dead_code)]
    pub fn ark_dequant_f16_to_f32(src: *const u16, dst: *mut f32, n: u64);
}

// ── Optimizador AdamW ASM (ark_optimizer.s) ───────────────────────────────────
extern "C" {
    #[allow(dead_code)]
    pub fn ark_asm_adam_step(
        w: *mut f32, g: *const f32, m: *mut f32, v: *mut f32,
        n: u64, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, t: u32,
    );

    /// [v1.3] Kernel Zero-Copy: AdamW leyendo FP32 (g,m,v) y actualizando VRAM FP16 (w) directamente.
    #[allow(dead_code)]
    pub fn ark_asm_adam_step_f16(
        w_fp16: *mut u16, g: *const f32, m: *mut f32, v: *mut f32,
        n: u64, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, t: u32,
    );

    /// Retorna la norma L2 pre-clip. Usado para métricas y clip global.
    pub fn ark_asm_grad_clip(
        grads:     *mut f32,
        n:         u64,
        threshold: f32,
    ) -> f32;
}

// ── Accelerate/vDSP: escalado vectorial ───────────────────────────────────────
#[cfg(target_os = "macos")]
extern "C" {
    /// void vDSP_vsmul(const float *__A, vDSP_Stride __IA,
    ///                 const float *__B, float *__C,
    ///                 vDSP_Stride __IC, vDSP_Length __N);
    pub fn vDSP_vsmul(
        a: *const f32,
        stride_a: isize,
        b: *const f32,
        c: *mut f32,
        stride_c: isize,
        n: usize,
    );
}

/// Escala un slice `f32` por `factor` utilizando `vDSP_vsmul` (NEON nativo, 4 floats/ciclo).
/// En macOS ejecuta rutinas ultra optimizadas de Apple Accelerate. 
/// Fallback seguro usando `chunks_mut(8)` para auto-vectorización en otras plataformas.
#[inline]
pub fn scale_tensor_vdsp(tensor: &mut[f32], factor: f32) {
    #[cfg(target_os = "macos")]
    unsafe {
        vDSP_vsmul(
            tensor.as_ptr(), 1,
            &factor,
            tensor.as_mut_ptr(), 1,
            tensor.len(),
        );
        return;
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        for chunk in tensor.chunks_mut(8) {
            for x in chunk {
                *x *= factor;
            }
        }
    }
}

// ── Wrappers Seguros de Rust ──────────────────────────────────────────────────

/// Inicializa el dispositivo Metal y la Command Queue.
pub fn gpu_init(n_layers: usize, d_model: usize, hidden_dim: usize) -> anyhow::Result<()> {
    let ok = unsafe {
        ark_mps_init_with_config(n_layers as i32, d_model as i32, hidden_dim as i32)
    };
    anyhow::ensure!(ok, "[gpu] Error crítico al inicializar dispositivo Metal o colas.");
    Ok(())
}

/// Pre-compila los grafos AOT de MPSGraph para las dimensiones exactas de la sesión actual.
pub fn gpu_build_graphs(batch_tokens: usize, vocab: usize, batch_size: usize) -> anyhow::Result<()> {
    let ok = unsafe {
        ark_mps_build_graphs(batch_tokens as i32, vocab as i32, batch_size as i32)
    };
    anyhow::ensure!(ok, "[gpu] Error al compilar los grafos MPSGraph AOT.");
    Ok(())
}

pub fn gpu_set_heads(n_heads: usize) {
    unsafe { ark_mps_set_heads(n_heads as i32) };
}

pub fn gpu_shutdown() {
    unsafe { ark_mps_shutdown() };
}

#[allow(dead_code)]
/// Ejecuta el chequeo NaN/Inf verificando la finitud matemática del slice con `chunks(8)`.
/// Facilita que LLVM lo resuelva con instrucciones SIMD de forma automática.
#[inline]
pub fn grads_are_finite(grads: &[f32]) -> bool {
    for chunk in grads.chunks(8) {
        for &x in chunk {
            if !x.is_finite() {
                return false;
            }
        }
    }
    true
}

/// Calcula y aplica el L2-Norm Clip si se excede el threshold en un slice individual.
/// En ARK v1.0 se recomienda el clip a nivel de Optimizer Global y no por tensor.
#[allow(dead_code)]
pub fn grad_clip_global(grads: &mut [f32], threshold: f32) -> f32 {
    if grads.is_empty() {
        return 0.0;
    }
    unsafe {
        ark_asm_grad_clip(grads.as_mut_ptr(), grads.len() as u64, threshold)
    }
}

/// Alias retrocompatible de grad clipping.
#[allow(dead_code)]
#[inline]
pub fn grad_clip(grads: &mut [f32], threshold: f32) -> f32 {
    grad_clip_global(grads, threshold)
}