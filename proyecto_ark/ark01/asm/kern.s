// ============================================================
// asm/ark_kernels.s — ARK v0.62  DEFINITIVO (FINAL)
// macOS AArch64 (Apple Silicon) — Mach-O, prefijo _ obligatorio
//
// Símbolos exportados:
//
//   MATEMÁTICA ESCALAR
//   _ark_expf_scalar(s0) → s0              exp(x), Horner grado 6
//   _ark_logf_approx(s0) → s0             ln(x),  Horner grado 5
//
//   KERNELS VECTORIALES FP32
//   _ark_exp4_inline(v0.4s) → v0.4s       exp x4, caller-saved interno
//   _ark_asm_rmsnorm(x0,x1,w2,w3)         RMSNorm in-place FP32
//   _ark_asm_softmax(x0,w1,w2)            Softmax estable (max-shift)
//   _ark_asm_cross_entropy(x0,x1,x2,w3,w4) Cross-entropy log-sum-exp
//   _ark_add_rmsnorm(x0,x1,x2,w3,w4)     Add residual + RMSNorm fusionados
//   _ark_swiglu_fwd(x0,x1,x2,x3)         SwiGLU forward (silu*up)
//
//   KERNELS FP16/MIXTO
//   _ark_dot_f16_f32accum(x0,x1,x2) → s0  Dot FP16 acum FP32 (fmlal)
//   _ark_rmsnorm_f16(x0,x1,w2,w3)         RMSNorm FP16 in-place (gamma FP32)
//   _ark_softmax_f16(x0,w1,w2)            Softmax sobre logits FP16
//   _ark_add_rmsnorm_f16(x0,x1,x2,w3,w4) Add residual + RMSNorm FP16
//   _ark_swiglu_f16(x0,x1,x2,x3)         SwiGLU gate/up FP16 → out FP16
//
//   EMBEDDING + BACKWARD
//   _ark_embed_gather(x0,x1,x2,x3,x4)    Embedding lookup vectorial FP32
//   _ark_embed_gather_f16(x0,x1,x2,x3,x4) Embedding lookup FP16 → FP32 out
//   _ark_rmsnorm_backward(x0,x1,x2,x3,x4,x5,x6,s0) RMSNorm backward FP32
//   _ark_dequant_f16_to_f32(x0,x1,x2)    Convertir buffer FP16 → FP32
//   _ark_quant_f32_to_f16(x0,x1,x2)      Convertir buffer FP32 → FP16
//
// ABI macOS ARM64:
//   Callee-saved enteros : x19–x28, x29(fp), x30(lr)
//   Callee-saved SIMD   : d8–d15 (solo 64 bits bajos de v8–v15)
//   Caller-saved libres : x0–x18, v0–v7, v16–v31
//
// Macro MOV32: carga constante f32 arbitraria en Wn
//   (fmov s,#imm solo acepta 256 valores estándar ARM)
//
// Historial de correcciones:
//
//   v0.60  — Versión inicial con kernels FP16 y backward
//
//   v0.61  BUG #1 (softmax_f16 underflow):
//            La pasada 2 guardaba exp() como f16 temporal causando
//            underflow a 0.0 exacto. Fix: 3 pasadas donde la pasada 2
//            solo acumula suma en FP32 y la pasada 3 recalcula exp()
//            desde el f16 original.
//
//   v0.61  BUG #2 (rmsnorm_backward fórmula):
//            El dot product del Paso 1 calculaba dot(dy, x_norm) sin
//            incluir gamma. Fix: se carga gamma en el loop y se calcula
//            dy*gamma antes de multiplicar por x_norm.
//
//   v0.62  BUG #3 (rmsnorm_backward punteros):
//            Al final de cada fila, los punteros x21(gamma) y x23(dgamma)
//            no se reiniciaban a la posición base (tenían mov x21,x21 y
//            mov x23,x23 que no hacen nada). Esto causaba lectura/escritura
//            fuera de rango para n_seq > 1.
//            Fix: guardar gamma_base en x25 y dgamma_base en x26 al inicio
//            y recargar x2/x4 desde ellos al inicio de cada fila.
//
//   v0.62  BUG #4 (rmsnorm_backward Paso 2):
//            Los punteros x19(dy) y x20(x_norm) se avanzaban durante el
//            Paso 1 (dot product) y el Paso 2 (update) los usaba desde
//            la posición avanzada, procesando datos incorrectos.
//            Fix: reiniciar x0/x1 desde x19_base/x20_base al inicio del
//            Paso 2, y avanzar x19/x20 solo al final de la fila completa.
//
// ============================================================

.macro MOV32 reg, hi, lo
    movz \reg, #\hi, lsl #16
    movk \reg, #\lo
.endm

.arch armv8.2-a+fp16+fp16fml
.section __TEXT,__text,regular,pure_instructions
.align 4

// ============================================================
// _ark_expf_scalar
// Entrada : s0 = x  (cualquier f32 finito)
// Salida  : s0 = exp(x)
// Clobbers: s1–s5, w9–w10
// ============================================================
.globl _ark_expf_scalar
_ark_expf_scalar:
    MOV32 w9, 0xC2AE, 0xAC50
    fmov  s1, w9
    MOV32 w9, 0x42B1, 0x7218
    fmov  s2, w9
    fmax  s0, s0, s1
    fmin  s0, s0, s2

    MOV32 w9, 0x3FB8, 0xAA3B
    fmov  s1, w9
    fmul  s1, s0, s1
    frintn s2, s1
    fcvtzs w9, s2

    MOV32 w10, 0x3F31, 0x7218
    fmov  s3, w10
    MOV32 w10, 0x32BF, 0xBE8E
    fmov  s4, w10
    fmul  s3, s2, s3
    fmul  s4, s2, s4
    fsub  s0, s0, s3
    fsub  s0, s0, s4

    MOV32 w10, 0x3AB6, 0x0B61
    fmov  s5, w10
    MOV32 w10, 0x3C08, 0x8889
    fmov  s3, w10
    fmul  s5, s5, s0; fadd s5, s5, s3

    MOV32 w10, 0x3D2A, 0xAAAB
    fmov  s3, w10
    fmul  s5, s5, s0; fadd s5, s5, s3

    MOV32 w10, 0x3E2A, 0xAAAB
    fmov  s3, w10
    fmul  s5, s5, s0; fadd s5, s5, s3

    fmov  s3, #0.5
    fmul  s5, s5, s0; fadd s5, s5, s3

    fmul  s5, s5, s0
    fmov  s3, #1.0
    fadd  s5, s5, s3

    fmul  s5, s5, s0
    fadd  s0, s5, s3

    mov   w10, #127
    add   w9, w9, w10
    lsl   w9, w9, #23
    fmov  s1, w9
    fmul  s0, s0, s1
    ret

// ============================================================
// _ark_logf_approx
// Entrada : s0 = x  (debe ser > 0)
// Salida  : s0 = ln(x)
// ============================================================
.globl _ark_logf_approx
_ark_logf_approx:
    fmov  w9, s0
    lsr   w10, w9, #23
    sub   w10, w10, #127
    and   w9, w9, #0x007FFFFF
    movz  w11, #0x3F80, lsl #16
    orr   w9, w9, w11
    fmov  s1, w9

    MOV32 w11, 0x3FB5, 0x04F3
    fmov  s2, w11
    fcmp  s1, s2
    b.lt  .L_log_ok
    fmov  s2, #0.5
    fmul  s1, s1, s2
    add   w10, w10, #1
.L_log_ok:
    fmov  s2, #1.0
    fsub  s1, s1, s2

    MOV32 w11, 0x3E4C, 0xCCCD
    fmov  s3, w11
    fmov  s4, #-0.25
    fmul  s3, s3, s1; fadd s3, s3, s4

    MOV32 w11, 0x3EAA, 0xAAAB
    fmov  s4, w11
    fmul  s3, s3, s1; fadd s3, s3, s4

    fmov  s4, #-0.5
    fmul  s3, s3, s1; fadd s3, s3, s4

    fmov  s4, #1.0
    fmul  s3, s3, s1; fadd s3, s3, s4

    fmul  s3, s3, s1

    scvtf s0, w10
    MOV32 w11, 0x3F31, 0x7218
    fmov  s4, w11
    fmadd s0, s0, s4, s3
    ret

// ============================================================
// _ark_exp4_inline
// Entrada : v0.4s = x[0..3]
// Salida  : v0.4s = exp(x[0..3])
// Clobbers: v16–v22  (todos caller-saved)
// Sin prólogo/epílogo — llamar con bl
// ============================================================
.globl _ark_exp4_inline
_ark_exp4_inline:
    MOV32 w9, 0xC2AE, 0xAC50; dup v16.4s, w9
    MOV32 w9, 0x42B1, 0x7218; dup v17.4s, w9
    fmax  v0.4s, v0.4s, v16.4s
    fmin  v0.4s, v0.4s, v17.4s

    MOV32 w9, 0x3FB8, 0xAA3B; dup v16.4s, w9
    fmul  v18.4s, v0.4s, v16.4s
    frintn v18.4s, v18.4s
    fcvtzs v19.4s, v18.4s

    MOV32 w9, 0x3F31, 0x7218; dup v16.4s, w9
    MOV32 w9, 0x32BF, 0xBE8E; dup v17.4s, w9
    fmul  v20.4s, v18.4s, v16.4s; fsub v0.4s, v0.4s, v20.4s
    fmul  v20.4s, v18.4s, v17.4s; fsub v0.4s, v0.4s, v20.4s

    MOV32 w9, 0x3AB6, 0x0B61; dup v16.4s, w9
    MOV32 w9, 0x3C08, 0x8889; dup v17.4s, w9
    fmul  v16.4s, v16.4s, v0.4s; fadd v16.4s, v16.4s, v17.4s

    MOV32 w9, 0x3D2A, 0xAAAB; dup v17.4s, w9
    fmul  v16.4s, v16.4s, v0.4s; fadd v16.4s, v16.4s, v17.4s

    MOV32 w9, 0x3E2A, 0xAAAB; dup v17.4s, w9
    fmul  v16.4s, v16.4s, v0.4s; fadd v16.4s, v16.4s, v17.4s

    fmov  v17.4s, #0.5
    fmul  v16.4s, v16.4s, v0.4s; fadd v16.4s, v16.4s, v17.4s

    fmul  v16.4s, v16.4s, v0.4s
    fmov  v17.4s, #1.0; fadd v16.4s, v16.4s, v17.4s

    fmul  v16.4s, v16.4s, v0.4s; fadd v0.4s, v16.4s, v17.4s

    movi  v20.4s, #127
    add   v19.4s, v19.4s, v20.4s
    shl   v19.4s, v19.4s, #23
    fmul  v0.4s, v0.4s, v19.4s
    ret

// ============================================================
// _ark_asm_rmsnorm
// void ark_asm_rmsnorm(float *x, const float *gamma,
//                      int n_seq, int dim)
// ABI: x0=x  x1=gamma  w2=n_seq  w3=dim
// ============================================================
.globl _ark_asm_rmsnorm
_ark_asm_rmsnorm:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0
    mov  x20, x1
    mov  w21, w2
    mov  w22, w3

.L_rms_row:
    cbz  w21, .L_rms_done

    movi v16.4s, #0
    mov  x0, x19
    mov  w5, w22
.L_rms_sq4:
    cmp  w5, #4
    b.lt .L_rms_sqtail
    ldr  q0, [x0], #16
    fmla v16.4s, v0.4s, v0.4s
    sub  w5, w5, #4
    b    .L_rms_sq4
.L_rms_sqtail:
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    cbz  w5, .L_rms_sq_end
    ldr  s0, [x0], #4
    fmadd s16, s0, s0, s16
    sub  w5, w5, #1
    b    .L_rms_sqtail
.L_rms_sq_end:

    ucvtf s0, w22
    fdiv  s16, s16, s0
    MOV32 w9, 0x3727, 0xC5AC
    fmov  s1, w9
    fadd  s16, s16, s1
    fsqrt s16, s16
    frecpe  s17, s16
    fmul    s0,  s17, s16
    frecps  s0,  s0,  s17
    fmul    s17, s17, s0
    dup     v17.4s, v17.s[0]

    mov  x0, x19
    mov  x1, x20
    mov  w5, w22
.L_rms_norm4:
    cmp  w5, #4
    b.lt .L_rms_normtail
    ldr  q0, [x0]
    ldr  q1, [x1], #16
    fmul v0.4s, v0.4s, v17.4s
    fmul v0.4s, v0.4s, v1.4s
    str  q0, [x0], #16
    sub  w5, w5, #4
    b    .L_rms_norm4
.L_rms_normtail:
    cbz  w5, .L_rms_next
    ldr  s0, [x0]
    ldr  s1, [x1], #4
    fmul s0, s0, s17
    fmul s0, s0, s1
    str  s0, [x0], #4
    sub  w5, w5, #1
    b    .L_rms_normtail

.L_rms_next:
    lsl  x0, x22, #2
    add  x19, x19, x0
    sub  w21, w21, #1
    b    .L_rms_row

.L_rms_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_asm_softmax
// void ark_asm_softmax(float *x, int n_seq, int dim)
// ABI: x0=x  w1=n_seq  w2=dim
// ============================================================
.globl _ark_asm_softmax
_ark_asm_softmax:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0
    mov  w20, w1
    mov  w21, w2

.L_sfx_row:
    cbz  w20, .L_sfx_done

    ldr   s8, [x19]
    dup   v22.4s, v8.s[0]
    mov   x0, x19
    mov   w5, w21
.L_sfx_max4:
    cmp   w5, #4
    b.lt  .L_sfx_maxtail
    ldr   q0, [x0], #16
    fmax  v22.4s, v22.4s, v0.4s
    sub   w5, w5, #4
    b     .L_sfx_max4
.L_sfx_maxtail:
    fmaxv s22, v22.4s
    dup   v22.4s, v22.s[0]
    cbz   w5, .L_sfx_max_end
    ldr   s0, [x0], #4
    fmax  s22, s22, s0
    dup   v22.4s, v22.s[0]
    sub   w5, w5, #1
    b     .L_sfx_maxtail
.L_sfx_max_end:

    movi  v23.4s, #0
    fmov  s9, #0.0
    mov   x0, x19
    mov   w5, w21
.L_sfx_exp4:
    cmp   w5, #4
    b.lt  .L_sfx_exptail
    ldr   q0, [x0]
    fsub  v0.4s, v0.4s, v22.4s
    bl    _ark_exp4_inline
    fadd  v23.4s, v23.4s, v0.4s
    str   q0, [x0], #16
    sub   w5, w5, #4
    b     .L_sfx_exp4
.L_sfx_exptail:
    faddp v23.4s, v23.4s, v23.4s
    faddp s23, v23.2s
    fadd  s9, s9, s23
    cbz   w5, .L_sfx_exp_end
    ldr   s0, [x0]
    fsub  s0, s0, s22
    stp   x0,  x19, [sp, #-32]!
    stp   w5,  w20, [sp, #16]
    str   s9,  [sp, #-16]!
    str   s22, [sp, #4]
    bl    _ark_expf_scalar
    ldr   s22, [sp, #4]
    ldr   s9,  [sp], #16
    ldp   w5,  w20, [sp, #16]
    ldp   x0,  x19, [sp], #32
    fadd  s9, s9, s0
    str   s0, [x0], #4
    sub   w5, w5, #1
    b     .L_sfx_exptail
.L_sfx_exp_end:

    fmov  s10, #1.0
    fdiv  s10, s10, s9
    dup   v24.4s, v10.s[0]
    mov   x0, x19
    mov   w5, w21
.L_sfx_div4:
    cmp   w5, #4
    b.lt  .L_sfx_divtail
    ldr   q0, [x0]
    fmul  v0.4s, v0.4s, v24.4s
    str   q0, [x0], #16
    sub   w5, w5, #4
    b     .L_sfx_div4
.L_sfx_divtail:
    cbz   w5, .L_sfx_next
    ldr   s0, [x0]
    fmul  s0, s0, s10
    str   s0, [x0], #4
    sub   w5, w5, #1
    b     .L_sfx_divtail

.L_sfx_next:
    lsl   x0, x21, #2
    add   x19, x19, x0
    sub   w20, w20, #1
    b     .L_sfx_row

.L_sfx_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_asm_cross_entropy
// void ark_asm_cross_entropy(
//     const float *logits, const uint32_t *targets,
//     float *loss, int n_seq, int vocab)
// ABI: x0=logits  x1=targets  x2=loss  w3=n_seq  w4=vocab
// ============================================================
.globl _ark_asm_cross_entropy
_ark_asm_cross_entropy:
    stp  x29, x30, [sp, #-96]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]
    stp  x25, x26, [sp, #64]
    stp  x27, x28, [sp, #80]

    mov  x19, x0
    mov  x20, x1
    mov  x21, x2
    mov  w22, w3
    mov  w23, w4
    mov  w27, w3
    fmov s8, #0.0

.L_ce_row:
    cbz  w22, .L_ce_done

    ldr   s9, [x19]
    dup   v25.4s, v9.s[0]
    mov   x0, x19
    mov   w5, w23
.L_ce_max4:
    cmp   w5, #4
    b.lt  .L_ce_maxtail
    ldr   q0, [x0], #16
    fmax  v25.4s, v25.4s, v0.4s
    sub   w5, w5, #4
    b     .L_ce_max4
.L_ce_maxtail:
    fmaxv s25, v25.4s
    dup   v25.4s, v25.s[0]
    cbz   w5, .L_ce_max_end
    ldr   s0, [x0], #4
    fmax  s25, s25, s0
    dup   v25.4s, v25.s[0]
    sub   w5, w5, #1
    b     .L_ce_maxtail
.L_ce_max_end:

    movi  v26.4s, #0
    fmov  s10, #0.0
    mov   x0, x19
    mov   w5, w23
.L_ce_sum4:
    cmp   w5, #4
    b.lt  .L_ce_sumtail
    ldr   q0, [x0], #16
    fsub  v0.4s, v0.4s, v25.4s
    bl    _ark_exp4_inline
    fadd  v26.4s, v26.4s, v0.4s
    sub   w5, w5, #4
    b     .L_ce_sum4
.L_ce_sumtail:
    faddp v26.4s, v26.4s, v26.4s
    faddp s26, v26.2s
    fadd  s10, s10, s26
    cbz   w5, .L_ce_sum_end
    ldr   s0, [x0], #4
    fsub  s0, s0, s25
    stp   x0,  x19, [sp, #-48]!
    stp   x20, x21, [sp, #16]
    stp   w5,  w22, [sp, #32]
    str   s10, [sp, #-16]!
    str   s25, [sp, #4]
    bl    _ark_expf_scalar
    ldr   s25, [sp, #4]
    ldr   s10, [sp], #16
    ldp   w5,  w22, [sp, #32]
    ldp   x20, x21, [sp, #16]
    ldp   x0,  x19, [sp], #48
    fadd  s10, s10, s0
    sub   w5, w5, #1
    b     .L_ce_sumtail
.L_ce_sum_end:

    fmov  s0, s10
    stp   x19, x20, [sp, #-48]!
    stp   x21, x22, [sp, #16]
    stp   w23, w27, [sp, #32]
    str   s8,  [sp, #-16]!
    str   s25, [sp, #4]
    bl    _ark_logf_approx
    ldr   s25, [sp, #4]
    ldr   s8,  [sp], #16
    ldp   w23, w27, [sp, #32]
    ldp   x21, x22, [sp, #16]
    ldp   x19, x20, [sp], #48
    fadd  s0, s0, s25

    ldr   w9, [x20], #4
    uxtw  x9, w9
    lsl   x9, x9, #2
    add   x9, x19, x9
    ldr   s1, [x9]
    fsub  s0, s0, s1
    fadd  s8, s8, s0

    uxtw  x0, w23
    lsl   x0, x0, #2
    add   x19, x19, x0
    sub   w22, w22, #1
    b     .L_ce_row

.L_ce_done:
    ucvtf s0, w27
    fdiv  s0, s8, s0
    str   s0, [x21]

    ldp  x27, x28, [sp, #80]
    ldp  x25, x26, [sp, #64]
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #96
    ret

// ============================================================
// _ark_add_rmsnorm
// void ark_add_rmsnorm(float *x, const float *residual,
//                      const float *gamma, int n_seq, int dim)
// ABI: x0=x  x1=residual  x2=gamma  w3=n_seq  w4=dim
// ============================================================
.globl _ark_add_rmsnorm
_ark_add_rmsnorm:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0
    mov  x20, x1
    mov  x21, x2
    mov  w22, w3
    mov  x23, x4

.L_arnorm_row:
    cbz  w22, .L_arnorm_done

    mov  x0, x19
    mov  x1, x20
    mov  w5, w23
.L_arnorm_add4:
    cmp  w5, #4
    b.lt .L_arnorm_addtail
    ldr  q0, [x0]
    ldr  q1, [x1], #16
    fadd v0.4s, v0.4s, v1.4s
    str  q0, [x0], #16
    sub  w5, w5, #4
    b    .L_arnorm_add4
.L_arnorm_addtail:
    cbz  w5, .L_arnorm_add_end
    ldr  s0, [x0]
    ldr  s1, [x1], #4
    fadd s0, s0, s1
    str  s0, [x0], #4
    sub  w5, w5, #1
    b    .L_arnorm_addtail
.L_arnorm_add_end:
    lsl  x24, x23, #2
    add  x20, x20, x24

    movi v16.4s, #0
    mov  x0, x19
    mov  w5, w23
.L_arnorm_sq4:
    cmp  w5, #4
    b.lt .L_arnorm_sqtail
    ldr  q0, [x0], #16
    fmla v16.4s, v0.4s, v0.4s
    sub  w5, w5, #4
    b    .L_arnorm_sq4
.L_arnorm_sqtail:
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    cbz  w5, .L_arnorm_sq_end
    ldr  s0, [x0], #4
    fmadd s16, s0, s0, s16
    sub  w5, w5, #1
    b    .L_arnorm_sqtail
.L_arnorm_sq_end:

    ucvtf s0, w23
    fdiv  s16, s16, s0
    MOV32 w9, 0x3727, 0xC5AC
    fmov  s1, w9
    fadd  s16, s16, s1
    fsqrt s16, s16
    frecpe  s17, s16
    fmul    s0,  s17, s16
    frecps  s0,  s0,  s17
    fmul    s17, s17, s0
    dup     v17.4s, v17.s[0]

    mov  x0, x19
    mov  x1, x21
    mov  w5, w23
.L_arnorm_scale4:
    cmp  w5, #4
    b.lt .L_arnorm_scaletail
    ldr  q0, [x0]
    ldr  q1, [x1], #16
    fmul v0.4s, v0.4s, v17.4s
    fmul v0.4s, v0.4s, v1.4s
    str  q0, [x0], #16
    sub  w5, w5, #4
    b    .L_arnorm_scale4
.L_arnorm_scaletail:
    cbz  w5, .L_arnorm_next
    ldr  s0, [x0]
    ldr  s1, [x1], #4
    fmul s0, s0, s17
    fmul s0, s0, s1
    str  s0, [x0], #4
    sub  w5, w5, #1
    b    .L_arnorm_scaletail

.L_arnorm_next:
    add  x19, x19, x24
    sub  w22, w22, #1
    b    .L_arnorm_row

.L_arnorm_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_swiglu_fwd
// void ark_swiglu_fwd(float *gate, const float *up,
//                     float *out, uint64_t n)
// ABI: x0=gate  x1=up  x2=out  x3=n
// ============================================================
.globl _ark_swiglu_fwd
_ark_swiglu_fwd:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0
    mov  x20, x1
    mov  x21, x2
    mov  x22, x3

    fmov v30.4s, #1.0

.L_swiglu_loop4:
    cmp  x22, #4
    b.lt .L_swiglu_tail

    ldr  q0, [x19], #16
    ldr  q3, [x20], #16
    mov  v4.16b, v0.16b
    fneg v0.4s, v0.4s
    bl   _ark_exp4_inline
    fadd v5.4s, v0.4s, v30.4s
    fdiv v5.4s, v30.4s, v5.4s
    fmul v5.4s, v4.4s, v5.4s
    fmul v5.4s, v5.4s, v3.4s
    str  q5, [x21], #16
    sub  x22, x22, #4
    b    .L_swiglu_loop4

.L_swiglu_tail:
    cbz  x22, .L_swiglu_done

    ldr  s4, [x19], #4
    ldr  s3, [x20], #4
    fneg s0, s4
    stp  x19, x20,[sp, #-32]!
    stp  x21, x22, [sp, #16]
    str  s4, [sp, #-16]!
    str  s3,[sp, #4]
    bl   _ark_expf_scalar
    ldr  s3,[sp, #4]
    ldr  s4, [sp], #16
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #32
    fmov s5, #1.0
    fadd s5, s0, s5
    fmov s6, #1.0
    fdiv s5, s6, s5
    fmul s5, s4, s5
    fmul s5, s5, s3
    str  s5, [x21], #4
    sub  x22, x22, #1
    b    .L_swiglu_tail

.L_swiglu_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_dot_f16_f32accum
// float ark_dot_f16_f32accum(const __fp16 *a, const __fp16 *b, uint64_t n)
// ABI: x0=a  x1=b  x2=n  →  s0 = dot(a,b) acumulado en f32
//
// 8-wide unroll con fmlal/fmlal2 (ARMv8.2-A, todos Apple Silicon)
// ============================================================
.globl _ark_dot_f16_f32accum
_ark_dot_f16_f32accum:
    stp  x29, x30, [sp, #-32]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]

    mov  x19, x0
    mov  x20, x1

    movi v16.4s, #0
    movi v17.4s, #0

.L_dot_loop8:
    cmp  x2, #8
    b.lt .L_dot_loop4
    ldr  q0, [x19], #16
    ldr  q1, [x20], #16
    fmlal  v16.4s, v0.4h, v1.4h
    // fmlal2 reemplazado: fcvtl2 expande lanes 4-7 de v0/v1 a FP32, fmla acumula
    fcvtl2 v2.4s, v0.8h
    fcvtl2 v3.4s, v1.8h
    fmla   v17.4s, v2.4s, v3.4s
    sub  x2, x2, #8
    b    .L_dot_loop8

.L_dot_loop4:
    cmp  x2, #4
    b.lt .L_dot_tail
    ldr  d0, [x19], #8
    ldr  d1, [x20], #8
    fmlal  v16.4s, v0.4h, v1.4h
    sub  x2, x2, #4
    b    .L_dot_loop4

.L_dot_tail:
    cbz  x2, .L_dot_reduce
    ldr  h0, [x19], #2
    ldr  h1, [x20], #2
    fcvt s0, h0
    fcvt s1, h1
    fmadd s16, s0, s1, s16
    sub  x2, x2, #1
    b    .L_dot_tail

.L_dot_reduce:
    fadd v16.4s, v16.4s, v17.4s
    faddp v16.4s, v16.4s, v16.4s
    faddp s0, v16.2s

    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #32
    ret

// ============================================================
// ============================================================
//  KERNELS FP16 MIXTO — v0.60
// ============================================================
// ============================================================

// ============================================================
// _ark_dequant_f16_to_f32
// void ark_dequant_f16_to_f32(const __fp16 *src, float *dst, uint64_t n)
// ABI: x0=src  x1=dst  x2=n
//
// Convierte buffer completo de FP16 a FP32.
// 8-wide: dos fcvtl por iteración.
// Usado para dequantizar pesos antes de matmul por Accelerate cuando
// el modelo vive en FP16 pero el BLAS necesita FP32.
// ============================================================
.globl _ark_dequant_f16_to_f32
_ark_dequant_f16_to_f32:
    stp  x29, x30, [sp, #-16]!
    mov  x29, sp

.L_dq_loop8:
    cmp  x2, #8
    b.lt .L_dq_loop4
    ldr  q0, [x0], #16          // 8 × f16
    fcvtl  v1.4s, v0.4h         // low 4 → f32
    fcvtl2 v2.4s, v0.8h         // high 4 → f32
    stp  q1, q2, [x1], #32
    sub  x2, x2, #8
    b    .L_dq_loop8

.L_dq_loop4:
    cmp  x2, #4
    b.lt .L_dq_tail
    ldr  d0, [x0], #8
    fcvtl v1.4s, v0.4h
    str  q1, [x1], #16
    sub  x2, x2, #4
    b    .L_dq_loop4

.L_dq_tail:
    cbz  x2, .L_dq_done
    ldr  h0, [x0], #2
    fcvt s0, h0
    str  s0, [x1], #4
    sub  x2, x2, #1
    b    .L_dq_tail

.L_dq_done:
    ldp  x29, x30, [sp], #16
    ret

// ============================================================
// _ark_quant_f32_to_f16
// void ark_quant_f32_to_f16(const float *src, __fp16 *dst, uint64_t n)
// ABI: x0=src  x1=dst  x2=n
//
// Convierte buffer FP32 → FP16 con saturación automática del fcvtn.
// 8-wide.
// ============================================================
.globl _ark_quant_f32_to_f16
_ark_quant_f32_to_f16:
    stp  x29, x30, [sp, #-16]!
    mov  x29, sp

.L_q_loop8:
    cmp  x2, #8
    b.lt .L_q_loop4
    ldp  q0, q1, [x0], #32     // 8 × f32
    fcvtn  v2.4h, v0.4s        // low 4 → f16
    fcvtn2 v2.8h, v1.4s        // high 4 → f16 (upper half de v2)
    str  q2, [x1], #16
    sub  x2, x2, #8
    b    .L_q_loop8

.L_q_loop4:
    cmp  x2, #4
    b.lt .L_q_tail
    ldr  q0, [x0], #16
    fcvtn v1.4h, v0.4s
    str  d1, [x1], #8
    sub  x2, x2, #4
    b    .L_q_loop4

.L_q_tail:
    cbz  x2, .L_q_done
    ldr  s0, [x0], #4
    fcvt h0, s0
    str  h0, [x1], #2
    sub  x2, x2, #1
    b    .L_q_tail

.L_q_done:
    ldp  x29, x30, [sp], #16
    ret

// ============================================================
// _ark_rmsnorm_f16
// void ark_rmsnorm_f16(__fp16 *x, const float *gamma,
//                      int n_seq, int dim)
// ABI: x0=x  x1=gamma  w2=n_seq  w3=dim
//
// RMSNorm in-place sobre activaciones FP16, gamma en FP32.
// Estrategia:
//   - Acumulación de Σx² en FP32 (evita overflow/underflow f16)
//   - inv_rms en FP32
//   - Escala en FP32 y re-cuantiza a FP16 al escribir
//
// Registros callee-saved: x19-x23
// ============================================================
.globl _ark_rmsnorm_f16
_ark_rmsnorm_f16:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0            // ptr x (f16, in-place)
    mov  x20, x1            // ptr gamma (f32)
    mov  w21, w2            // n_seq
    mov  w22, w3            // dim

.L_rmsf16_row:
    cbz  w21, .L_rmsf16_done

    // Fase 1: Σx² en FP32 (4-wide fcvtl + fmla)
    movi v16.4s, #0
    mov  x0, x19
    mov  w5, w22
.L_rmsf16_sq4:
    cmp  w5, #4
    b.lt .L_rmsf16_sqtail
    ldr  d0, [x0], #8          // 4 × f16
    fcvtl v0.4s, v0.4h         // → f32
    fmla v16.4s, v0.4s, v0.4s
    sub  w5, w5, #4
    b    .L_rmsf16_sq4
.L_rmsf16_sqtail:
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    cbz  w5, .L_rmsf16_sq_end
    ldr  h0, [x0], #2
    fcvt s0, h0
    fmadd s16, s0, s0, s16
    sub  w5, w5, #1
    b    .L_rmsf16_sqtail
.L_rmsf16_sq_end:

    // inv_rms
    ucvtf s0, w22
    fdiv  s16, s16, s0
    MOV32 w9, 0x3727, 0xC5AC    // ε = 1e-5
    fmov  s1, w9
    fadd  s16, s16, s1
    fsqrt s16, s16
    frecpe  s17, s16
    fmul    s0,  s17, s16
    frecps  s0,  s0,  s17
    fmul    s17, s17, s0        // inv_rms en s17
    dup     v17.4s, v17.s[0]

    // Fase 3: x[i] = f16(x[i] * inv_rms * gamma[i])
    mov  x0, x19
    mov  x1, x20
    mov  w5, w22
.L_rmsf16_scale4:
    cmp  w5, #4
    b.lt .L_rmsf16_scaletail
    ldr  d0, [x0]               // 4 × f16
    fcvtl v0.4s, v0.4h         // → f32
    ldr  q1, [x1], #16         // gamma[i:i+4] f32
    fmul v0.4s, v0.4s, v17.4s  // * inv_rms
    fmul v0.4s, v0.4s, v1.4s   // * gamma
    fcvtn v2.4h, v0.4s         // → f16
    str  d2, [x0], #8
    sub  w5, w5, #4
    b    .L_rmsf16_scale4
.L_rmsf16_scaletail:
    cbz  w5, .L_rmsf16_next
    ldr  h0, [x0]
    fcvt s0, h0
    ldr  s1, [x1], #4
    fmul s0, s0, s17
    fmul s0, s0, s1
    fcvt h0, s0
    str  h0, [x0], #2
    sub  w5, w5, #1
    b    .L_rmsf16_scaletail

.L_rmsf16_next:
    // Avanzar x (f16: dim * 2 bytes)
    lsl  x24, x22, #1
    add  x19, x19, x24
    sub  w21, w21, #1
    b    .L_rmsf16_row

.L_rmsf16_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_softmax_f16
// void ark_softmax_f16(__fp16 *x, int n_seq, int dim)
// ABI: x0=x  w1=n_seq  w2=dim
//
// Softmax numéricamente estable sobre logits FP16.
// Aritmética interna en FP32; resultado re-cuantizado a FP16.
//
// CORREGIDO v0.61 (BUG #1 — Underflow por intermedio FP16):
//   La versión anterior guardaba exp() como f16 temporal en el buffer
//   durante la pasada 2 (fcvtn + str d1). Los valores pequeños de exp()
//   llegaban a 0.0 exacto en FP16 mucho antes que en FP32, haciendo que
//   tokens con probabilidad pequeña-pero-no-nula quedaran con prob=0.
//   Esto causaba pérdida de entropía y spikes de pérdida durante el
//   entrenamiento.
//
//   Fix: 3 pasadas sin intermediario f16:
//     Pasada 1 — max en FP32 (guardado en d8, callee-saved robusto)
//     Pasada 2 — solo acumular suma en FP32 (sin escribir al buffer)
//     Pasada 3 — recalcular exp() desde el original f16, dividir y escribir
//
//   El recálculo de exp() en la pasada 3 es idéntico en valor al de la
//   pasada 2 porque leemos el mismo f16 original. El coste es una segunda
//   pasada de lectura + exp(), pero se elimina el round-trip f16 intermedio
//   que destruía la precisión.
//
//   d8 = max (callee-saved: sobrevive los bl _ark_exp4_inline/_ark_expf_scalar)
//   d9 = 1/suma (calculado al inicio de la pasada 3)
// ============================================================
.globl _ark_softmax_f16
_ark_softmax_f16:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  d8,  d9,  [sp, #48]    // d8=max, d9=1/sum (callee-saved robustos)

    mov  x19, x0
    mov  w20, w1
    mov  w21, w2

.L_sfxf16_row:
    cbz  w20, .L_sfxf16_done

    // ── PASADA 1: Encontrar MAX en FP32 ──────────────────────
    // Iniciamos con el primer elemento y hacemos fmax 4-wide.
    ldr  h8, [x19]
    fcvt s8, h8
    dup  v22.4s, v8.s[0]
    mov  x0, x19
    mov  w5, w21
.L_sfxf16_max4:
    cmp  w5, #4
    b.lt .L_sfxf16_maxtail
    ldr  d0, [x0], #8
    fcvtl v0.4s, v0.4h
    fmax v22.4s, v22.4s, v0.4s
    sub  w5, w5, #4
    b    .L_sfxf16_max4
.L_sfxf16_maxtail:
    fmaxv s22, v22.4s
    dup   v22.4s, v22.s[0]
    cbz   w5, .L_sfxf16_max_end
    ldr   h0, [x0], #2
    fcvt  s0, h0
    fmax  s22, s22, s0
    dup   v22.4s, v22.s[0]
    sub   w5, w5, #1
    b     .L_sfxf16_maxtail
.L_sfxf16_max_end:
    // Guardar max en d8 (callee-saved: sobrevive los bl que siguen)
    fmov  s8, s22

    // ── PASADA 2: Suma de exp(x - max) en FP32, SIN escribir al buffer ─
    // Solo acumulamos la suma; el buffer f16 queda intacto para la pasada 3.
    movi v23.4s, #0
    fmov s9, #0.0
    mov  x0, x19
    mov  w5, w21
.L_sfxf16_exp4:
    cmp  w5, #4
    b.lt .L_sfxf16_exptail
    ldr  d0, [x0], #8           // leer 4 × f16, avanzar puntero
    fcvtl v0.4s, v0.4h
    dup   v22.4s, v8.s[0]       // recargar max desde d8 (callee-saved)
    fsub  v0.4s, v0.4s, v22.4s
    bl   _ark_exp4_inline        // v0 = exp(x - max) en FP32
    fadd v23.4s, v23.4s, v0.4s  // acumular suma
    // NO se escribe al buffer — aquí estaba el bug
    sub  w5, w5, #4
    b    .L_sfxf16_exp4
.L_sfxf16_exptail:
    faddp v23.4s, v23.4s, v23.4s
    faddp s23, v23.2s
    fadd  s9, s9, s23
    cbz   w5, .L_sfxf16_exp_end
    ldr   h0, [x0], #2
    fcvt  s0, h0
    fsub  s0, s0, s8            // usar s8 (max, callee-saved)
    stp   x0,  x19, [sp, #-32]!
    stp   w5,  w20, [sp, #16]
    bl    _ark_expf_scalar
    ldp   w5,  w20, [sp, #16]
    ldp   x0,  x19, [sp], #32
    fadd  s9, s9, s0
    sub   w5, w5, #1
    b     .L_sfxf16_exptail
.L_sfxf16_exp_end:

    // ── PASADA 3: Recalcular exp y dividir por suma ───────────
    // Leemos el f16 original (intacto), recalculamos exp() en FP32,
    // dividimos por sum y escribimos el resultado final en f16.
    // El recálculo da el mismo valor que pasada 2 porque el input es igual.
    fmov  s10, #1.0
    fdiv  s9, s10, s9           // s9 = 1.0 / suma (guardado en d9)
    mov   x0, x19
    mov   w5, w21
.L_sfxf16_div4:
    cmp   w5, #4
    b.lt  .L_sfxf16_divtail
    ldr   d0, [x0]              // leer 4 × f16 original (sin avanzar aún)
    fcvtl v0.4s, v0.4h
    dup   v22.4s, v8.s[0]       // max desde d8
    fsub  v0.4s, v0.4s, v22.4s
    bl    _ark_exp4_inline
    dup   v24.4s, v9.s[0]       // 1/suma desde d9
    fmul  v0.4s, v0.4s, v24.4s
    fcvtn v1.4h, v0.4s          // → f16 final (aquí sí es correcto: el valor ya está normalizado)
    str   d1, [x0], #8
    sub   w5, w5, #4
    b     .L_sfxf16_div4
.L_sfxf16_divtail:
    cbz   w5, .L_sfxf16_next
    ldr   h0, [x0]
    fcvt  s0, h0
    fsub  s0, s0, s8
    stp   x0,  x19, [sp, #-32]!
    stp   w5,  w20, [sp, #16]
    bl    _ark_expf_scalar
    ldp   w5,  w20, [sp, #16]
    ldp   x0,  x19, [sp], #32
    fmul  s0, s0, s9
    fcvt  h0, s0
    str   h0, [x0], #2
    sub   w5, w5, #1
    b     .L_sfxf16_divtail

.L_sfxf16_next:
    lsl   x0, x21, #1          // dim * 2 bytes
    add   x19, x19, x0
    sub   w20, w20, #1
    b     .L_sfxf16_row

.L_sfxf16_done:
    ldp  d8,  d9,  [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_add_rmsnorm_f16
// void ark_add_rmsnorm_f16(__fp16 *x, const __fp16 *residual,
//                           const float *gamma, int n_seq, int dim)
// ABI: x0=x  x1=residual  x2=gamma  w3=n_seq  w4=dim
//
// Fusiona residual add + RMSNorm sobre buffers FP16.
// Gamma en FP32 (parámetros aprendibles siempre en FP32).
// ============================================================
.globl _ark_add_rmsnorm_f16
_ark_add_rmsnorm_f16:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0
    mov  x20, x1
    mov  x21, x2
    mov  w22, w3
    mov  x23, x4

.L_arnf16_row:
    cbz  w22, .L_arnf16_done

    // Fase 1: x[i] += residual[i]  (FP16 → FP32 → suma → FP16)
    mov  x0, x19
    mov  x1, x20
    mov  w5, w23
.L_arnf16_add4:
    cmp  w5, #4
    b.lt .L_arnf16_addtail
    ldr  d0, [x0]               // 4 × f16 de x
    ldr  d1, [x1], #8           // 4 × f16 de residual
    fcvtl v0.4s, v0.4h
    fcvtl v1.4s, v1.4h
    fadd v0.4s, v0.4s, v1.4s
    fcvtn v2.4h, v0.4s
    str  d2, [x0], #8
    sub  w5, w5, #4
    b    .L_arnf16_add4
.L_arnf16_addtail:
    cbz  w5, .L_arnf16_add_end
    ldr  h0, [x0]
    ldr  h1, [x1], #2
    fcvt s0, h0
    fcvt s1, h1
    fadd s0, s0, s1
    fcvt h0, s0
    str  h0, [x0], #2
    sub  w5, w5, #1
    b    .L_arnf16_addtail
.L_arnf16_add_end:
    lsl  x24, x23, #1           // dim * 2 bytes
    add  x20, x20, x24

    // Fase 2: Σx² en FP32
    movi v16.4s, #0
    mov  x0, x19
    mov  w5, w23
.L_arnf16_sq4:
    cmp  w5, #4
    b.lt .L_arnf16_sqtail
    ldr  d0, [x0], #8
    fcvtl v0.4s, v0.4h
    fmla v16.4s, v0.4s, v0.4s
    sub  w5, w5, #4
    b    .L_arnf16_sq4
.L_arnf16_sqtail:
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    cbz  w5, .L_arnf16_sq_end
    ldr  h0, [x0], #2
    fcvt s0, h0
    fmadd s16, s0, s0, s16
    sub  w5, w5, #1
    b    .L_arnf16_sqtail
.L_arnf16_sq_end:

    ucvtf s0, w23
    fdiv  s16, s16, s0
    MOV32 w9, 0x3727, 0xC5AC
    fmov  s1, w9
    fadd  s16, s16, s1
    fsqrt s16, s16
    frecpe  s17, s16
    fmul    s0,  s17, s16
    frecps  s0,  s0,  s17
    fmul    s17, s17, s0
    dup     v17.4s, v17.s[0]

    // Fase 3: x[i] = f16(x[i] * inv_rms * gamma[i])
    mov  x0, x19
    mov  x1, x21
    mov  w5, w23
.L_arnf16_scale4:
    cmp  w5, #4
    b.lt .L_arnf16_scaletail
    ldr  d0, [x0]
    fcvtl v0.4s, v0.4h
    ldr  q1, [x1], #16
    fmul v0.4s, v0.4s, v17.4s
    fmul v0.4s, v0.4s, v1.4s
    fcvtn v2.4h, v0.4s
    str  d2, [x0], #8
    sub  w5, w5, #4
    b    .L_arnf16_scale4
.L_arnf16_scaletail:
    cbz  w5, .L_arnf16_next
    ldr  h0, [x0]
    fcvt s0, h0
    ldr  s1, [x1], #4
    fmul s0, s0, s17
    fmul s0, s0, s1
    fcvt h0, s0
    str  h0, [x0], #2
    sub  w5, w5, #1
    b    .L_arnf16_scaletail

.L_arnf16_next:
    add  x19, x19, x24
    sub  w22, w22, #1
    b    .L_arnf16_row

.L_arnf16_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_swiglu_f16
// void ark_swiglu_f16(const __fp16 *gate, const __fp16 *up,
//                      __fp16 *out, uint64_t n)
// ABI: x0=gate  x1=up  x2=out  x3=n
//
// SwiGLU: out[i] = silu(gate[i]) * up[i]
// gate y up en FP16; out en FP16.
// Aritmética interna en FP32.
// ============================================================
.globl _ark_swiglu_f16
_ark_swiglu_f16:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0
    mov  x20, x1
    mov  x21, x2
    mov  x22, x3

    fmov v30.4s, #1.0

.L_swf16_loop4:
    cmp  x22, #4
    b.lt .L_swf16_tail

    ldr  d0, [x19], #8          // 4 × f16 gate
    ldr  d3, [x20], #8          // 4 × f16 up
    fcvtl v0.4s, v0.4h         // gate → f32
    fcvtl v3.4s, v3.4h         // up   → f32
    mov  v4.16b, v0.16b         // preservar gate original

    fneg v0.4s, v0.4s           // Invierte signo para estabilizar
    bl   _ark_exp4_inline       // v0 = exp(-gate)

    fadd v5.4s, v0.4s, v30.4s
    fdiv v5.4s, v30.4s, v5.4s  // sigmoid = 1 / (1 + exp(-gate))
    fmul v5.4s, v4.4s, v5.4s   // silu = gate * sigmoid
    fmul v5.4s, v5.4s, v3.4s   // out  = silu * up

    fcvtn v6.4h, v5.4s
    str  d6, [x21], #8
    sub  x22, x22, #4
    b    .L_swf16_loop4

.L_swf16_tail:
    cbz  x22, .L_swf16_done

    ldr  h0, [x19], #2
    ldr  h3, [x20], #2
    fcvt s0, h0
    fcvt s3, h3
    fmov s4, s0
    fneg s0, s0                 // Invertir signo

    stp  x19, x20, [sp, #-32]!
    stp  x21, x22, [sp, #16]
    str  s4,[sp, #-16]!
    str  s3, [sp, #4]
    bl   _ark_expf_scalar
    ldr  s3, [sp, #4]
    ldr  s4, [sp], #16
    ldp  x21, x22,[sp, #16]
    ldp  x19, x20, [sp], #32

    fmov s5, #1.0
    fadd s5, s0, s5
    fmov s6, #1.0
    fdiv s5, s6, s5             // sigmoid = 1 / (1 + exp(-gate))
    fmul s5, s4, s5
    fmul s5, s5, s3
    fcvt h5, s5
    str  h5, [x21], #2
    sub  x22, x22, #1
    b    .L_swf16_tail

.L_swf16_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// ============================================================
//  EMBEDDING + BACKWARD
// ============================================================
// ============================================================

// ============================================================
// _ark_embed_gather
// void ark_embed_gather(
//     const float    *table,   x0 — [vocab × dim]  FP32
//     const uint32_t *indices, x1 — [n_seq]
//     float          *out,     x2 — [n_seq × dim]  FP32
//     uint64_t        n_seq,   x3
//     uint64_t        dim      x4
// )
//
// Para cada token t: out[t] = table[indices[t]]
//
// Estrategia: fila completa de `dim` floats copiada con NEON 4-wide.
// Sin scatter (solo gather puro): acceso secuencial a out, aleatorio a table.
// Alineación: table debe estar alineada a 16 bytes (los embeddings usuales
// allocados por Rust vec! o ndarray lo están).
// ============================================================
.globl _ark_embed_gather
_ark_embed_gather:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0            // ptr table
    mov  x20, x1            // ptr indices
    mov  x21, x2            // ptr out
    mov  x22, x3            // n_seq
    mov  x23, x4            // dim

    // stride en bytes: dim * 4
    lsl  x24, x23, #2       // x24 = dim * sizeof(f32)

.L_emb_loop:
    cbz  x22, .L_emb_done

    // Cargar índice de token
    ldr  w9, [x20], #4      // w9 = token_id
    uxtw x9, w9
    mul  x9, x9, x24        // byte offset = token_id * dim * 4
    add  x0, x19, x9        // ptr a fila table[token_id]
    mov  x1, x21            // ptr a fila de salida

    // Copiar dim floats (4-wide)
    mov  w5, w23
.L_emb_copy4:
    cmp  w5, #4
    b.lt .L_emb_copytail
    ldr  q0, [x0], #16
    str  q0, [x1], #16
    sub  w5, w5, #4
    b    .L_emb_copy4
.L_emb_copytail:
    cbz  w5, .L_emb_next
    ldr  s0, [x0], #4
    str  s0, [x1], #4
    sub  w5, w5, #1
    b    .L_emb_copytail

.L_emb_next:
    add  x21, x21, x24      // avanzar out a la siguiente fila
    sub  x22, x22, #1
    b    .L_emb_loop

.L_emb_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_embed_gather_f16
// void ark_embed_gather_f16(
//     const __fp16   *table,   x0 — [vocab × dim] FP16
//     const uint32_t *indices, x1 — [n_seq]
//     float          *out,     x2 — [n_seq × dim] FP32  (dequant al vuelo)
//     uint64_t        n_seq,   x3
//     uint64_t        dim      x4
// )
//
// Igual que embed_gather pero tabla en FP16 → salida en FP32.
// Útil cuando la tabla de embeddings se almacena en FP16 (2x memoria)
// y el forward necesita FP32 para la primera capa.
// ============================================================
.globl _ark_embed_gather_f16
_ark_embed_gather_f16:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]

    mov  x19, x0
    mov  x20, x1
    mov  x21, x2
    mov  x22, x3
    mov  x23, x4

    // stride tabla en bytes: dim * 2 (f16)
    lsl  x24, x23, #1       // x24 = dim * sizeof(f16)

.L_embf16_loop:
    cbz  x22, .L_embf16_done

    ldr  w9, [x20], #4
    uxtw x9, w9
    mul  x9, x9, x24
    add  x0, x19, x9        // ptr fila f16
    mov  x1, x21            // ptr salida f32

    mov  w5, w23
.L_embf16_copy8:
    cmp  w5, #8
    b.lt .L_embf16_copy4
    ldr  q0, [x0], #16      // 8 × f16
    fcvtl  v1.4s, v0.4h
    fcvtl2 v2.4s, v0.8h
    stp  q1, q2, [x1], #32
    sub  w5, w5, #8
    b    .L_embf16_copy8
.L_embf16_copy4:
    cmp  w5, #4
    b.lt .L_embf16_copytail
    ldr  d0, [x0], #8
    fcvtl v1.4s, v0.4h
    str  q1, [x1], #16
    sub  w5, w5, #4
    b    .L_embf16_copy4
.L_embf16_copytail:
    cbz  w5, .L_embf16_next
    ldr  h0, [x0], #2
    fcvt s0, h0
    str  s0, [x1], #4
    sub  w5, w5, #1
    b    .L_embf16_copytail

.L_embf16_next:
    // out avanza dim * 4 (FP32)
    lsl  x9, x23, #2
    add  x21, x21, x9
    sub  x22, x22, #1
    b    .L_embf16_loop

.L_embf16_done:
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #64
    ret

// ============================================================
// _ark_rmsnorm_backward  —  v0.62 FINAL (todos los bugs corregidos)
//
// void ark_rmsnorm_backward(
//     const float *dy,        x0 — gradiente upstream  [n_seq × dim]
//     const float *x_norm,    x1 — x normalizado (post-norm) [n_seq × dim]
//     const float *gamma,     x2 — escala aprendida   [dim]
//     float       *dx,        x3 — grad respecto a x_norm [n_seq × dim]
//     float       *dgamma,    x4 — grad respecto a gamma [dim] (acumulado)
//     float        inv_rms,   s0 — 1/RMS precalculado (escalar por fila)
//     int          n_seq,     w5
//     int          dim        w6
// )
//
// Gradiente de RMSNorm respecto a x_norm (FÓRMULA CORRECTA):
//
//   Para cada fila i:
//     Paso 1: dot_eff = Σ_j (dy[i,j] * gamma[j]) * x_norm[i,j] / dim
//     Paso 2: dx[i,j] = gamma[j] * inv_rms * (dy[i,j] - x_norm[i,j] * dot_eff)
//     Paso 3: dgamma[j] += dy[i,j] * x_norm[i,j]
//
// CORRECCIONES:
//
//   v0.61 BUG #2 (fórmula):
//     El dot del Paso 1 calculaba dot(dy, x_norm) sin gamma.
//     Fix: se carga gamma en el loop y se calcula dy*gamma antes
//     de multiplicar por x_norm.
//
//   v0.62 BUG #3 (punteros gamma/dgamma):
//     Al final de .L_rmnbwd_next, las instrucciones mov x21,x21 y
//     mov x23,x23 no reiniciaban los punteros a la base.
//     Fix: se guardan gamma_base en x25 y dgamma_base en x26 durante
//     el prólogo y se recargan al inicio de cada fila.
//
//   v0.62 BUG #4 (punteros dy/x_norm en Paso 2):
//     Los punteros x0/x1 se avanzaban durante el Paso 1 (dot product)
//     y el Paso 2 los usaba desde esa posición avanzada.
//     Fix: se reinician x0/x1 desde x19/x20 al inicio del Paso 2,
//     y x19/x20 solo se avanzan al final de la fila completa.
//
// inv_rms debe calcularse fuera (Rust/C) usando el RMS del forward.
//
// ABI: x0=dy  x1=x_norm  x2=gamma  x3=dx  x4=dgamma
//      s0=inv_rms  w5=n_seq  w6=dim
//
// Registros callee-saved:
//   x19 = dy (base, avanza entre filas)
//   x20 = x_norm (base, avanza entre filas)
//   x21 = gamma (se reinicia cada fila desde x25)
//   x22 = dx (base, avanza entre filas)
//   x23 = dgamma (se reinicia cada fila desde x26)
//   x24 = n_seq (contador)
//   x25 = dim
//   x26 = gamma_base (guardado en prólogo)
//   x27 = dgamma_base (guardado en prólogo) — no se usa, sobra
//   d8  = inv_rms (callee-saved)
// ============================================================
// _ark_rmsnorm_backward
// ============================================================
.globl _ark_rmsnorm_backward
_ark_rmsnorm_backward:
    // CORRECCIÓN 1: Reservar 112 bytes para que d8 y d9 quepan dentro del frame
    stp  x29, x30, [sp, #-112]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24,[sp, #48]
    stp  x25, x26, [sp, #64]
    stp  x27, x28, [sp, #80]
    stp  d8,  d9,  [sp, #96]

    mov  x19, x0            // dy base
    mov  x20, x1            // x_norm base
    mov  x21, x2            // gamma (se usará como puntero móvil)
    mov  x22, x3            // dx base
    mov  x23, x4            // dgamma (se usará como puntero móvil)
    fmov s8, s0             // d8 = inv_rms (callee-saved)
    mov  w24, w5            // n_seq
    mov  w28, w6            // dim  ← usar x28 en lugar de x25

    // ── Guardar bases de gamma y dgamma ─────────
    mov  x25, x21           // x25 = gamma_base
    mov  x26, x23           // x26 = dgamma_base

.L_rmnbwd_row:
    cbz  w24, .L_rmnbwd_done

    // ── Reiniciar gamma y dgamma al inicio de cada fila ──────
    mov  x2, x25            // gamma = gamma_base
    mov  x4, x26            // dgamma = dgamma_base

    // ── Paso 1: dot(dy * gamma, x_norm) / dim ────────────────
    movi v16.4s, #0
    mov  x0, x19            // dy ptr para esta fila
    mov  x1, x20            // x_norm ptr para esta fila
    mov  x7, x2             // gamma ptr (local, se reinicia cada fila)
    mov  w5, w28            // dim  ← leer desde x28
.L_rmnbwd_dot4:
    cmp  w5, #4
    b.lt .L_rmnbwd_dottail
    ldr  q0, [x0], #16      // dy[i:i+4]
    ldr  q1, [x1], #16      // x_norm[i:i+4]
    ldr  q2,[x7], #16      // gamma[i:i+4]
    fmul v2.4s, v0.4s, v2.4s    // dy * gamma
    fmla v16.4s, v2.4s, v1.4s   // acumular (dy*gamma) * x_norm
    sub  w5, w5, #4
    b    .L_rmnbwd_dot4
.L_rmnbwd_dottail:
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s       // s16 = dot(dy*gamma, x_norm)
    cbz  w5, .L_rmnbwd_dot_end
    ldr  s0, [x0], #4
    ldr  s1, [x1], #4
    ldr  s2, [x7], #4
    fmul s2, s0, s2         // dy * gamma
    fmadd s16, s2, s1, s16
    sub  w5, w5, #1
    b    .L_rmnbwd_dottail
.L_rmnbwd_dot_end:

    // CORRECCIÓN 2: Usar w28 para convertir la dimensión a float
    ucvtf s9, w28            // s9 = dim (float)
    fdiv  s16, s16, s9       // s16 = dot/dim
    dup   v16.4s, v16.s[0]  // broadcast

    // inv_rms broadcast
    dup   v17.4s, v8.s[0]

    // ── Paso 2: dx[i] = gamma[i] * inv_rms * (dy[i] - x_norm[i] * dot/dim) ─
    //           dgamma[i] += dy[i] * x_norm[i]
    //
    mov  x0, x19            // dy  vuelve al inicio de la fila
    mov  x1, x20            // x_norm vuelve al inicio de la fila
    mov  x7, x2             // gamma vuelve al inicio
    mov  x3, x22            // dx ptr para esta fila
    mov  x6, x4             // dgamma ptr (local)
    mov  w5, w28            // dim  ← leer desde x28

.L_rmnbwd_update4:
    cmp  w5, #4
    b.lt .L_rmnbwd_updatetail

    ldr  q0, [x0], #16      // dy[i:i+4]
    ldr  q1, [x1], #16      // x_norm[i:i+4]
    ldr  q2, [x7], #16      // gamma[i:i+4]

    // dy[i] - x_norm[i] * (dot/dim)
    fmls v0.4s, v1.4s, v16.4s   // v0 = dy - x_norm*(dot/dim)

    // dx[i] = gamma * inv_rms * (dy - ...)
    fmul v3.4s, v2.4s, v17.4s   // gamma * inv_rms
    fmul v3.4s, v3.4s, v0.4s    // * (dy - x_norm*(dot/dim))
    str  q3, [x3], #16

    // dgamma[i] += dy[i] * x_norm[i]
    ldr  q4, [x6]
    ldr  q0,[x0, #-16]         // re-leer dy (x0 ya avanzó)
    ldr  q1, [x1, #-16]         // re-leer x_norm
    fmla v4.4s, v0.4s, v1.4s
    str  q4,[x6], #16

    sub  w5, w5, #4
    b    .L_rmnbwd_update4

.L_rmnbwd_updatetail:
    cbz  w5, .L_rmnbwd_next

    ldr  s0, [x0], #4       // dy[i]
    ldr  s1, [x1], #4       // x_norm[i]
    ldr  s2, [x7], #4       // gamma[i]

    // dy[i] - x_norm[i] * (dot/dim)
    fmul s3, s1, s16        // x_norm * dot/dim  (s16 = escalar)
    fsub s0, s0, s3

    // dx[i]
    fmul s3, s2, s8         // gamma * inv_rms
    fmul s3, s3, s0
    str  s3, [x3], #4

    // dgamma[i]
    ldr  s3, [x6]
    ldr  s5, [x0, #-4]
    ldr  s6, [x1, #-4]
    fmadd s3, s5, s6, s3
    str  s3, [x6], #4

    sub  w5, w5, #1
    b    .L_rmnbwd_updatetail

.L_rmnbwd_next:
    // ── Avanzar punteros de fila al siguiente token ──────────
    lsl  x0, x28, #2        // dim * 4 bytes
    add  x19, x19, x0       // dy   += stride
    add  x20, x20, x0       // x_norm += stride
    add  x22, x22, x0       // dx   += stride
    // gamma y dgamma NO avanzan aquí — se reinician arriba desde x25/x26

    sub  w24, w24, #1
    b    .L_rmnbwd_row

.L_rmnbwd_done:
    ldp  d8,  d9,  [sp, #96]
    ldp  x27, x28, [sp, #80]
    ldp  x25, x26, [sp, #64]
    ldp  x23, x24,[sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    // CORRECCIÓN 1b: Restaurar los mismos 112 bytes que se reservaron en el prólogo
    ldp  x29, x30, [sp], #112
    ret