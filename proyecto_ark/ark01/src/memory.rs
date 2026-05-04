// src/memory.rs — ARK v1.0
//
// Cambios v1.0: Zero-Copy Real.
// - Eliminado struct TensorF16.
// - ModelWeights y LayerWeights almacenan punteros `*mut u16` directos al MTLBuffer de la GPU.
// - Sincronización directa y optimizador AdamW operan sobre esta memoria compartida sin `memcpy`.

use std::alloc::{alloc, dealloc, Layout};

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — AlignedVec<T>: buffer alineado a 4096 bytes (página Apple Silicon)
// ─────────────────────────────────────────────────────────────────────────────

pub struct AlignedVec<T: Copy + Default> {
    ptr:    *mut T,
    len:    usize,
    layout: Layout,
}

unsafe impl<T: Copy + Default + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for AlignedVec<T> {}

impl<T: Copy + Default> AlignedVec<T> {
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "AlignedVec: len debe ser > 0");
        let size = len * std::mem::size_of::<T>();
        let padded = (size + 4095) & !4095;
        let layout = Layout::from_size_align(padded, 4096)
            .expect("AlignedVec: Layout inválido");
        let ptr = unsafe { alloc(layout) as *mut T };
        assert!(!ptr.is_null(), "AlignedVec: alloc falló (OOM)");
        unsafe { std::ptr::write_bytes(ptr, 0u8, len) };
        Self { ptr, len, layout }
    }

    #[inline] pub fn len(&self)            -> usize      { self.len }
    #[inline] pub fn as_ptr(&self)         -> *const T   { self.ptr }
    #[inline] pub fn as_mut_ptr(&mut self) -> *mut   T   { self.ptr }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[inline]
    pub fn zero(&mut self) {
        unsafe { std::ptr::write_bytes(self.ptr, 0u8, self.len); }
    }

    pub fn fill(&mut self, val: T) {
        for x in self.as_mut_slice().iter_mut() { *x = val; }
    }

    pub fn copy_from_slice(&mut self, src: &[T]) {
        assert_eq!(src.len(), self.len, "AlignedVec::copy_from_slice: tamaños distintos");
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, self.len); }
    }

    pub fn iter(&self)     -> std::slice::Iter<'_, T>    { self.as_slice().iter() }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> { self.as_mut_slice().iter_mut() }
}

impl<T: Copy + Default> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, self.layout) };
    }
}

impl<T: Copy + Default> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T { &self.as_slice()[i] }
}
impl<T: Copy + Default> std::ops::IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, i: usize) -> &mut T { &mut self.as_mut_slice()[i] }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Conversión FP32 ↔ FP16 (Rust puro, con redondeo al más cercano)
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
#[allow(dead_code)]
pub fn f32_to_f16(v: f32) -> u16 {
    let bits     = v.to_bits();
    let sign     = (bits >> 31) & 0x1;
    let exp_f32  = ((bits >> 23) & 0xFF) as i32;
    let mant_f32 =  bits & 0x7F_FFFF;

    if exp_f32 == 0xFF {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if exp_f32 == 0 {
        return (sign << 15) as u16;
    }

    let exp_f16 = exp_f32 - 127 + 15;
    if exp_f16 <= 0  { return (sign << 15) as u16; }
    if exp_f16 >= 31 { return ((sign << 15) | (0x1F << 10)) as u16; }

    // Redondeo al más cercano (round-to-nearest-even) para igualar precisión NEON (fcvtn).
    let mant_f16 = (mant_f32 + 0x1000 + ((mant_f32 >> 13) & 1)) >> 13;
    ((sign << 15) | ((exp_f16 as u32) << 10) | mant_f16) as u16
}

#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Tensor f32 (gradientes + momentos Adam)
// ─────────────────────────────────────────────────────────────────────────────

pub struct Tensor {
    pub data: AlignedVec<f32>,
    #[allow(dead_code)]
    pub dim0: usize,
    #[allow(dead_code)]
    pub dim1: usize,
}

impl Tensor {
    pub fn zeros(dim0: usize, dim1: usize) -> Self {
        Self { data: AlignedVec::new(dim0 * dim1), dim0, dim1 }
    }

    pub fn rand_uniform(dim0: usize, dim1: usize, scale: f32) -> Self {
        let mut t = Self::zeros(dim0, dim1);
        let mut seed: u64 = 12345;
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        for x in t.data.iter_mut() {
            seed = seed.wrapping_mul(a).wrapping_add(c);
            let bits = (seed >> 40) as u32;
            let f = (bits as f32) / 16_777_215.0;
            *x = f * scale * 2.0 - scale;
        }
        t
    }

    #[allow(dead_code)]
    pub fn rand_normal(dim0: usize, dim1: usize, std: f32) -> Self {
        // Box-Muller: genera distribución normal desde uniforme
        let mut t = Self::zeros(dim0, dim1);
        let mut seed: u64 = 98765;
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        let mut iter = t.data.iter_mut();
        loop {
            seed = seed.wrapping_mul(a).wrapping_add(c);
            let u1 = ((seed >> 40) as u32) as f32 / 4294967295.0 + 1e-7;
            seed = seed.wrapping_mul(a).wrapping_add(c);
            let u2 = ((seed >> 40) as u32) as f32 / 4294967295.0;
            let mag = std * (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();
            match iter.next() { Some(x) => *x = z0, None => break }
            match iter.next() { Some(x) => *x = z1, None => break }
        }
        t
    }

    pub fn kaiming(dim0: usize, dim1: usize) -> Self {
        let scale = (2.0f32 / dim0 as f32).sqrt();
        Self::rand_uniform(dim0, dim1, scale)
    }

    #[allow(dead_code)]
    pub fn kaiming_scaled(dim0: usize, dim1: usize, n_layers: usize) -> Self {
        // Escala adicional 1/sqrt(2*n_layers) para w2 y wo — previene acumulación de varianza
        let std = (2.0f32 / dim0 as f32).sqrt() / (2.0 * n_layers as f32).sqrt();
        Self::rand_normal(dim0, dim1, std)
    }

    pub fn embedding_init(vocab: usize, d_model: usize) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        Self::rand_uniform(vocab, d_model, scale)
    }

    pub fn ones(dim0: usize, dim1: usize) -> Self {
        let mut t = Self::zeros(dim0, dim1);
        t.data.fill(1.0f32);
        t
    }

    #[allow(dead_code)]
    #[inline] pub fn len(&self)            -> usize      { self.data.len() }
    #[inline] pub fn as_ptr(&self)         -> *const f32 { self.data.as_ptr() }
    #[inline] pub fn as_mut_ptr(&mut self) -> *mut   f32 { self.data.as_mut_ptr() }
    #[inline] pub fn zero_grad(&mut self) { self.data.zero(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — ModelWeights (Zero-Copy Real)
// ─────────────────────────────────────────────────────────────────────────────

pub struct ModelWeights {
    pub vocab: usize,
    pub d: usize,

    // Punteros Zero-Copy a los MTLBuffer de la GPU (en FP16)
    pub embed_w_fp16: *mut u16,
    #[allow(dead_code)]
    pub gamma_f_fp16: *mut u16,

    // Capas
    pub layers: Vec<LayerWeights>,

    // Gradientes y momentos FP32 en RAM CPU
    pub embed_w_grad:     Tensor,
    pub embed_m:          Tensor,
    pub embed_v:          Tensor,
    pub gamma_final_grad: Tensor,
    pub gamma_final_m:    Tensor,
    pub gamma_final_v:    Tensor,
}

// Hacemos que ModelWeights sea seguro de enviar entre hilos para compilación sin errores
unsafe impl Send for ModelWeights {}
unsafe impl Sync for ModelWeights {}

pub struct LayerWeights {
    pub d: usize,
    pub h: usize,

    // Punteros Zero-Copy a los MTLBuffer de la GPU (en FP16)
    pub wq_fp16: *mut u16, pub wk_fp16: *mut u16,
    pub wv_fp16: *mut u16, pub wo_fp16: *mut u16,
    pub w1_fp16: *mut u16, pub w2_fp16: *mut u16,
    pub w3_fp16: *mut u16,
    pub g1_fp16: *mut u16, pub g2_fp16: *mut u16,

    // Gradientes FP32 en RAM CPU
    pub wq_grad: Tensor, pub wk_grad: Tensor,
    pub wv_grad: Tensor, pub wo_grad: Tensor,
    pub w1_grad: Tensor, pub w2_grad: Tensor,
    pub w3_grad: Tensor,
    pub g1_grad: Tensor, pub g2_grad: Tensor,

    // Momentos Adam FP32 en RAM CPU
    pub wq_m: Tensor, pub wq_v: Tensor,
    pub wk_m: Tensor, pub wk_v: Tensor,
    pub wv_m: Tensor, pub wv_v: Tensor,
    pub wo_m: Tensor, pub wo_v: Tensor,
    pub w1_m: Tensor, pub w1_v: Tensor,
    pub w2_m: Tensor, pub w2_v: Tensor,
    pub w3_m: Tensor, pub w3_v: Tensor,
    pub g1_m: Tensor, pub g1_v: Tensor,
    pub g2_m: Tensor, pub g2_v: Tensor,
}

impl ModelWeights {
    pub fn new(n_layers: usize, vocab: usize, d: usize, h: usize) -> Self {
        // 1. Obtener punteros Zero-Copy del embedding y gamma final
        let mut embed_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut gamma_f_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            crate::ffi::ark_mps_get_embed_ptr(&mut embed_ptr, &mut gamma_f_ptr, vocab as i32);
        }
        let embed_w_fp16 = embed_ptr as *mut u16;
        let gamma_f_fp16 = gamma_f_ptr as *mut u16;

        // Validación defensiva: si ark_mps_build_graphs() no se llamó antes,
        // los punteros serán nulos y cualquier escritura causará bus error.
        assert!(!embed_w_fp16.is_null(),
            "[memory] FATAL: embed_w_fp16 es nulo — ark_mps_build_graphs() debe llamarse ANTES de ModelWeights::new()");
        assert!(!gamma_f_fp16.is_null(),
            "[memory] FATAL: gamma_f_fp16 es nulo — ark_mps_build_graphs() debe llamarse ANTES de ModelWeights::new()");

        // 2. Inicializar embedding temporal en FP32 y pasarlo a FP16 en la GPU directamente
        let temp_emb = Tensor::embedding_init(vocab, d);
        unsafe {
            crate::ffi::ark_quant_f32_to_f16(temp_emb.as_ptr(), embed_w_fp16, (vocab * d) as u64);
        }
        // (gamma_final ya está inicializado a 1.0 internamente por ark_mps_bridge.m)

        // 3. Crear capas e inicializarlas
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let mut wq: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut wk: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut wv: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut wo: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut w1: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut w3: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut w2: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut g1: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut g2: *mut std::ffi::c_void = std::ptr::null_mut();

            unsafe {
                crate::ffi::ark_mps_get_weight_ptrs(
                    l as i32,
                    &mut wq, &mut wk, &mut wv, &mut wo,
                    &mut w1, &mut w3, &mut w2,
                    &mut g1, &mut g2
                );
            }

            // Validación defensiva por capa
            assert!(!wq.is_null(), "[memory] FATAL: puntero wq nulo en capa {} — build_graphs() debe preceder a ModelWeights::new()", l);
            assert!(!wk.is_null(), "[memory] FATAL: puntero wk nulo en capa {}", l);
            assert!(!wv.is_null(), "[memory] FATAL: puntero wv nulo en capa {}", l);

            layers.push(LayerWeights::new(
                d, h, n_layers,
                wq as *mut u16, wk as *mut u16, wv as *mut u16, wo as *mut u16,
                w1 as *mut u16, w2 as *mut u16, w3 as *mut u16,
                g1 as *mut u16, g2 as *mut u16
            ));
        }

        Self {
            vocab,
            d,
            embed_w_fp16,
            gamma_f_fp16,
            embed_w_grad:     Tensor::zeros(vocab, d),
            embed_m:          Tensor::zeros(vocab, d),
            embed_v:          Tensor::zeros(vocab, d),
            gamma_final_grad: Tensor::zeros(1, d),
            gamma_final_m:    Tensor::zeros(1, d),
            gamma_final_v:    Tensor::zeros(1, d),
            layers,
        }
    }

    pub fn n_params(&self) -> usize {
        let embed_params = self.vocab * self.d;
        let gamma_f_params = self.d;
        embed_params + gamma_f_params + self.layers.iter().map(|l| l.n_params()).sum::<usize>()
    }

    pub fn zero_all_grads(&mut self) {
        self.embed_w_grad.zero_grad();
        self.gamma_final_grad.zero_grad();
        for l in &mut self.layers { l.zero_all_grads(); }
    }

    /// Sincroniza tensores FP32 hacia los buffers Zero-Copy en GPU (FP16).
    /// Utiliza el kernel en ASM `ark_quant_f32_to_f16` para máxima velocidad.
    /// Esta función se llama al cargar un modelo desde disco o restaurar checkpoints.
    pub fn sync_weights_to_gpu(&mut self, embed_w: &[f32], layers_f32: &[Vec<Vec<f32>>]) {
        unsafe {
            // Sincronizar embedding
            crate::ffi::ark_quant_f32_to_f16(
                embed_w.as_ptr(),
                self.embed_w_fp16,
                (self.vocab * self.d) as u64
            );

            // Sincronizar pesos de cada capa
            for (l_idx, l) in self.layers.iter_mut().enumerate() {
                let f32_tensors = &layers_f32[l_idx];

                let ptrs_and_lens = [
                    (f32_tensors[0].as_ptr(), l.wq_fp16, l.d * l.d),
                    (f32_tensors[1].as_ptr(), l.wk_fp16, l.d * l.d),
                    (f32_tensors[2].as_ptr(), l.wv_fp16, l.d * l.d),
                    (f32_tensors[3].as_ptr(), l.wo_fp16, l.d * l.d),
                    (f32_tensors[4].as_ptr(), l.w1_fp16, l.d * l.h),
                    (f32_tensors[5].as_ptr(), l.w2_fp16, l.h * l.d),
                    (f32_tensors[6].as_ptr(), l.w3_fp16, l.d * l.h),
                    (f32_tensors[7].as_ptr(), l.g1_fp16, l.d),
                    (f32_tensors[8].as_ptr(), l.g2_fp16, l.d),
                ];

                for (src, dst, len) in ptrs_and_lens.iter() {
                    crate::ffi::ark_quant_f32_to_f16(*src, *dst, *len as u64);
                }
            }
        }
    }

    pub fn all_grad_slices_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        let embed = self.embed_w_grad.data.as_mut_slice().iter_mut();
        let layers = self.layers.iter_mut().flat_map(|l| {
            l.wq_grad.data.as_mut_slice().iter_mut()
                .chain(l.wk_grad.data.as_mut_slice().iter_mut())
                .chain(l.wv_grad.data.as_mut_slice().iter_mut())
                .chain(l.wo_grad.data.as_mut_slice().iter_mut())
                .chain(l.w1_grad.data.as_mut_slice().iter_mut())
                .chain(l.w2_grad.data.as_mut_slice().iter_mut())
                .chain(l.w3_grad.data.as_mut_slice().iter_mut())
                .chain(l.g1_grad.data.as_mut_slice().iter_mut())
                .chain(l.g2_grad.data.as_mut_slice().iter_mut())
        });
        embed.chain(layers)
    }
}

impl LayerWeights {
    pub fn new(
        d: usize, h: usize, _n_layers: usize,
        wq_fp16: *mut u16, wk_fp16: *mut u16, wv_fp16: *mut u16, wo_fp16: *mut u16,
        w1_fp16: *mut u16, w2_fp16: *mut u16, w3_fp16: *mut u16,
        g1_fp16: *mut u16, g2_fp16: *mut u16,
    ) -> Self {
        // Inicializar los pesos en GPU (Zero-Copy) generando tensores kaiming/ones y cuantizando
        let kaiming_init = |ptr: *mut u16, dim0: usize, dim1: usize| {
            let temp = Tensor::kaiming(dim0, dim1);
            unsafe { crate::ffi::ark_quant_f32_to_f16(temp.as_ptr(), ptr, (dim0 * dim1) as u64) };
        };
        let ones_init = |ptr: *mut u16, dim0: usize, dim1: usize| {
            let temp = Tensor::ones(dim0, dim1);
            unsafe { crate::ffi::ark_quant_f32_to_f16(temp.as_ptr(), ptr, (dim0 * dim1) as u64) };
        };

        kaiming_init(wq_fp16, d, d);
        kaiming_init(wk_fp16, d, d);
        kaiming_init(wv_fp16, d, d);
        kaiming_init(wo_fp16, d, d);
        kaiming_init(w1_fp16, d, h);
        kaiming_init(w2_fp16, h, d);
        kaiming_init(w3_fp16, d, h);
        ones_init(g1_fp16, 1, d);
        ones_init(g2_fp16, 1, d);

        Self {
            d, h,

            wq_fp16, wk_fp16, wv_fp16, wo_fp16,
            w1_fp16, w2_fp16, w3_fp16,
            g1_fp16, g2_fp16,

            wq_grad: Tensor::zeros(d, d), wk_grad: Tensor::zeros(d, d),
            wv_grad: Tensor::zeros(d, d), wo_grad: Tensor::zeros(d, d),
            w1_grad: Tensor::zeros(d, h), w2_grad: Tensor::zeros(h, d),
            w3_grad: Tensor::zeros(d, h),
            g1_grad: Tensor::zeros(1, d), g2_grad: Tensor::zeros(1, d),

            wq_m: Tensor::zeros(d,d), wq_v: Tensor::zeros(d,d),
            wk_m: Tensor::zeros(d,d), wk_v: Tensor::zeros(d,d),
            wv_m: Tensor::zeros(d,d), wv_v: Tensor::zeros(d,d),
            wo_m: Tensor::zeros(d,d), wo_v: Tensor::zeros(d,d),
            w1_m: Tensor::zeros(d,h), w1_v: Tensor::zeros(d,h),
            w2_m: Tensor::zeros(h,d), w2_v: Tensor::zeros(h,d),
            w3_m: Tensor::zeros(d,h), w3_v: Tensor::zeros(d,h),
            g1_m: Tensor::zeros(1,d), g1_v: Tensor::zeros(1,d),
            g2_m: Tensor::zeros(1,d), g2_v: Tensor::zeros(1,d),
        }
    }

    pub fn n_params(&self) -> usize {
        let d = self.d;
        let h = self.h;
        d * d * 4 + d * h * 3 + d * 2
    }

    pub fn zero_all_grads(&mut self) {
        self.wq_grad.zero_grad(); self.wk_grad.zero_grad();
        self.wv_grad.zero_grad(); self.wo_grad.zero_grad();
        self.w1_grad.zero_grad(); self.w2_grad.zero_grad();
        self.w3_grad.zero_grad();
        self.g1_grad.zero_grad(); self.g2_grad.zero_grad();
    }
}