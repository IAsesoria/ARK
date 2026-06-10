.macro MOV32 reg, hi, lo
    movz \reg, #\hi, lsl #16
    movk \reg, #\lo
.endm

.section __TEXT,__text,regular,pure_instructions
.align 4

// ============================================================
// _ark_asm_adam_step  (FP32 puro)
//
// AdamW con corrección de bias completa.
//   m[i] = β1·m[i] + (1−β1)·g[i]
//   v[i] = β2·v[i] + (1−β2)·g[i]²
//   bc1  = 1 − β1^t
//   bc2  = 1 − β2^t
//   lr_eff = lr · √bc2 / bc1
//   w[i] = w[i]·(1−wd·lr) − lr_eff · m[i] / (√v[i] + ε)
// ============================================================
.globl _ark_asm_adam_step
_ark_asm_adam_step:
    stp  x29, x30, [sp, #-128]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]
    stp  d8,  d9,  [sp, #64]
    stp  d10, d11, [sp, #80]
    stp  d12, d13, [sp, #96]
    stp  d14, d15, [sp, #112]

    mov  x19, x0          // w
    mov  x20, x1          // g
    mov  x21, x2          // m
    mov  x22, x3          // v
    mov  x23, x4          // n (u64)

    // ── FIX v0.51: garantizar t ≥ 1 para evitar bc1=bc2=0 → NaN ──
    // Si el caller usa índice base-cero (t=0 en el primer paso),
    // bc1 = 1 − β1^0 = 0 y la división fdiv s15,s15,s13 inyecta NaN.
    // cinc incrementa w5 solo si la condición eq (w5==0) se cumple.
    cmp  w5, #0
    cinc w5, w5, eq

    fmov s8,  s0          // d8  = lr
    fmov s9,  s1          // d9  = beta1
    fmov s10, s2          // d10 = beta2
    fmov s11, s3          // d11 = eps

    // 1 − wd·lr → d12
    fmul s12, s8, s4
    fmov s0, #1.0
    fsub s12, s0, s12

    // bc1 = 1 − β1^t → d13 (Exponenciación Binaria O(log t))
    fmov  s13, #1.0       // res = 1.0
    fmov  s0, s9          // base = beta1 (s9)
    mov   w9, w5          // t
.L_adam_bc1_bin:
    cbz   w9, .L_adam_bc1_done
    tbz   w9, #0, .L_adam_bc1_skip
    fmul  s13, s13, s0
.L_adam_bc1_skip:
    fmul  s0, s0, s0
    lsr   w9, w9, #1
    b     .L_adam_bc1_bin
.L_adam_bc1_done:
    fmov  s15, #1.0       // s15 temporal para la resta
    fsub  s13, s15, s13   // d13 = bc1 = 1.0 - beta1^t

    // bc2 = 1 − β2^t → d14 (Exponenciación Binaria O(log t))
    fmov  s14, #1.0       // res = 1.0
    fmov  s0, s10         // base = beta2 (s10)
    mov   w9, w5          // t
.L_adam_bc2_bin:
    cbz   w9, .L_adam_bc2_done
    tbz   w9, #0, .L_adam_bc2_skip
    fmul  s14, s14, s0
.L_adam_bc2_skip:
    fmul  s0, s0, s0
    lsr   w9, w9, #1
    b     .L_adam_bc2_bin
.L_adam_bc2_done:
    fmov  s15, #1.0
    fsub  s14, s15, s14   // d14 = bc2 = 1.0 - beta2^t

    // lr_eff = lr · √bc2 / bc1 → d15
    fsqrt s15, s14
    fdiv  s15, s15, s13
    fmul  s15, s15, s8    // d15 = lr_eff

    // Vectores constantes (caller-saved v16–v22)
    dup  v16.4s, v9.s[0]          // beta1
    dup  v17.4s, v10.s[0]         // beta2
    fmov s0, #1.0
    fsub s0, s0, s9               // 1 − beta1
    dup  v18.4s, v0.s[0]
    fmov s0, #1.0
    fsub s0, s0, s10              // 1 − beta2
    dup  v19.4s, v0.s[0]
    dup  v20.4s, v11.s[0]         // eps
    dup  v21.4s, v15.s[0]         // lr_eff
    dup  v22.4s, v12.s[0]         // 1 − wd·lr

.L_adam_loop4:
    cmp  x23, #4
    b.lt .L_adam_tail

    ldr  q0, [x20], #16           // g[i:i+4]
    ldr  q1, [x21]                // m
    ldr  q2, [x22]                // v
    ldr  q3, [x19]                // w

    fmul v1.4s, v1.4s, v16.4s
    fmla v1.4s, v18.4s, v0.4s
    str  q1, [x21], #16

    fmul v2.4s, v2.4s, v17.4s
    fmul v4.4s, v0.4s, v0.4s
    fmla v2.4s, v19.4s, v4.4s
    str  q2, [x22], #16

    fsqrt v5.4s, v2.4s
    fadd  v5.4s, v5.4s, v20.4s
    fdiv  v6.4s, v1.4s, v5.4s
    fmul  v6.4s, v6.4s, v21.4s

    fmul v3.4s, v3.4s, v22.4s
    fsub v3.4s, v3.4s, v6.4s
    str  q3, [x19], #16

    sub  x23, x23, #4
    b    .L_adam_loop4

.L_adam_tail:
    cbz  x23, .L_adam_done

    ldr  s0, [x20], #4
    ldr  s1, [x21]
    ldr  s2, [x22]
    ldr  s3, [x19]

    fmul  s1, s1, s9
    fmov  s4, #1.0
    fsub  s4, s4, s9
    fmadd s1, s4, s0, s1
    str   s1, [x21], #4

    fmul  s2, s2, s10
    fmov  s4, #1.0
    fsub  s4, s4, s10
    fmul  s5, s0, s0
    fmadd s2, s4, s5, s2
    str   s2, [x22], #4

    fsqrt s4, s2
    fadd  s4, s4, s11
    fdiv  s4, s1, s4
    fmul  s4, s4, s15

    fmul  s3, s3, s12
    fsub  s3, s3, s4
    str   s3, [x19], #4

    sub  x23, x23, #1
    b    .L_adam_tail

.L_adam_done:
    ldp  d14, d15, [sp, #112]
    ldp  d12, d13, [sp, #96]
    ldp  d10, d11, [sp, #80]
    ldp  d8,  d9,  [sp, #64]
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #128
    ret

// ============================================================
// _ark_asm_adam_step_f16  — AdamW FP16 MIXTO
//
// Misma matemática que adam_step, pero:
//   - w almacenado en __fp16 → dequantizar antes, requantizar al final
//   - g en FP32 (acumulados por el backward)
//   - m, v en FP32 siempre
//
// CORREGIDO v0.51: mismo fix t=0 que la versión FP32.
//
// ABI: x0=w(__fp16)  x1=g(f32)  x2=m(f32)  x3=v(f32)  x4=n(u64)
//      s0=lr  s1=b1  s2=b2  s3=eps  s4=wd  w5=t(u32)
// ============================================================
.globl _ark_asm_adam_step_f16
_ark_asm_adam_step_f16:
    stp  x29, x30, [sp, #-128]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  x23, x24, [sp, #48]
    stp  d8,  d9,  [sp, #64]
    stp  d10, d11, [sp, #80]
    stp  d12, d13, [sp, #96]
    stp  d14, d15, [sp, #112]

    mov  x19, x0          // w (f16)
    mov  x20, x1          // g (f32)
    mov  x21, x2          // m (f32)
    mov  x22, x3          // v (f32)
    mov  x23, x4          // n (u64)

    // ── FIX v0.51: t=0 → NaN ──
    cmp  w5, #0
    cinc w5, w5, eq

    fmov s8,  s0          // lr
    fmov s9,  s1          // beta1
    fmov s10, s2          // beta2
    fmov s11, s3          // eps

    fmul s12, s8, s4
    fmov s0, #1.0
    fsub s12, s0, s12     // 1 − wd·lr

    // bc1 = 1 − β1^t → d13 (Exponenciación Binaria O(log t))
    fmov  s13, #1.0       // res = 1.0
    fmov  s0, s9          // base = beta1 (s9)
    mov   w9, w5          // t
.L_adamf16_bc1_bin:
    cbz   w9, .L_adamf16_bc1_done
    tbz   w9, #0, .L_adamf16_bc1_skip
    fmul  s13, s13, s0
.L_adamf16_bc1_skip:
    fmul  s0, s0, s0
    lsr   w9, w9, #1
    b     .L_adamf16_bc1_bin
.L_adamf16_bc1_done:
    fmov  s15, #1.0
    fsub  s13, s15, s13   // d13 = bc1

    // bc2 = 1 − β2^t → d14 (Exponenciación Binaria O(log t))
    fmov  s14, #1.0       // res = 1.0
    fmov  s0, s10         // base = beta2 (s10)
    mov   w9, w5          // t
.L_adamf16_bc2_bin:
    cbz   w9, .L_adamf16_bc2_done
    tbz   w9, #0, .L_adamf16_bc2_skip
    fmul  s14, s14, s0
.L_adamf16_bc2_skip:
    fmul  s0, s0, s0
    lsr   w9, w9, #1
    b     .L_adamf16_bc2_bin
.L_adamf16_bc2_done:
    fmov  s15, #1.0
    fsub  s14, s15, s14   // d14 = bc2

    // lr_eff
    fsqrt s15, s14
    fdiv  s15, s15, s13
    fmul  s15, s15, s8

    // Vectores constantes
    dup  v16.4s, v9.s[0]
    dup  v17.4s, v10.s[0]
    fmov s0, #1.0
    fsub s0, s0, s9
    dup  v18.4s, v0.s[0]
    fmov s0, #1.0
    fsub s0, s0, s10
    dup  v19.4s, v0.s[0]
    dup  v20.4s, v11.s[0]
    dup  v21.4s, v15.s[0]
    dup  v22.4s, v12.s[0]

    // ── Bucle 4-wide: dequant w f16→f32, update, requant f32→f16 ──
.L_adamf16_loop4:
    cmp  x23, #4
    b.lt .L_adamf16_tail

    ldr  d7, [x19]                // 4 × f16  (w)
    fcvtl v3.4s, v7.4h            // w → f32

    ldr  q0, [x20], #16           // g (f32)
    ldr  q1, [x21]                // m
    ldr  q2, [x22]                // v

    // m = β1·m + (1−β1)·g
    fmul v1.4s, v1.4s, v16.4s
    fmla v1.4s, v18.4s, v0.4s
    str  q1, [x21], #16

    // v = β2·v + (1−β2)·g²
    fmul v2.4s, v2.4s, v17.4s
    fmul v4.4s, v0.4s, v0.4s
    fmla v2.4s, v19.4s, v4.4s
    str  q2, [x22], #16

    // update = lr_eff · m / (√v + ε)
    fsqrt v5.4s, v2.4s
    fadd  v5.4s, v5.4s, v20.4s
    fdiv  v6.4s, v1.4s, v5.4s
    fmul  v6.4s, v6.4s, v21.4s

    // w = w·(1−wd·lr) − update
    fmul v3.4s, v3.4s, v22.4s
    fsub v3.4s, v3.4s, v6.4s

    // requantizar w f32 → f16
    fcvtn v7.4h, v3.4s
    str  d7, [x19], #8            // escribir 4 × f16

    sub  x23, x23, #4
    b    .L_adamf16_loop4

    // ── Tail escalar ──────────────────────────────────────────
.L_adamf16_tail:
    cbz  x23, .L_adamf16_done

    ldr  h0, [x19]                // w[i] como f16
    fcvt s3, h0                   // w → f32

    ldr  s0, [x20], #4            // g[i]
    ldr  s1, [x21]                // m[i]
    ldr  s2, [x22]                // v[i]

    fmul  s1, s1, s9
    fmov  s4, #1.0
    fsub  s4, s4, s9
    fmadd s1, s4, s0, s1
    str   s1, [x21], #4

    fmul  s2, s2, s10
    fmov  s4, #1.0
    fsub  s4, s4, s10
    fmul  s5, s0, s0
    fmadd s2, s4, s5, s2
    str   s2, [x22], #4

    fsqrt s4, s2
    fadd  s4, s4, s11
    fdiv  s4, s1, s4
    fmul  s4, s4, s15

    fmul  s3, s3, s12
    fsub  s3, s3, s4

    fcvt  h3, s3                  // w actualizado → f16
    str   h3, [x19], #2

    sub  x23, x23, #1
    b    .L_adamf16_tail

.L_adamf16_done:
    ldp  d14, d15, [sp, #112]
    ldp  d12, d13, [sp, #96]
    ldp  d10, d11, [sp, #80]
    ldp  d8,  d9,  [sp, #64]
    ldp  x23, x24, [sp, #48]
    ldp  x21, x22, [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #128
    ret

// ============================================================
// _ark_asm_grad_clip
// float ark_asm_grad_clip(float *grads, uint64_t n, float threshold) → f32
// ABI: x0=grads  x1=n(u64)  s0=threshold
// Retorna: s0 = norma L2 pre-clip (siempre)
// Sin cambios desde v0.41 — era correcto.
// ============================================================
.globl _ark_asm_grad_clip
_ark_asm_grad_clip:
    stp  x29, x30, [sp, #-48]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  d8,  d9,  [sp, #32]

    mov  x19, x0
    mov  x20, x1
    fmov s8, s0               // threshold

    movi v16.4s, #0
    mov  x0, x19
    mov  x1, x20
.L_clip_norm4:
    cmp  x1, #4
    b.lt .L_clip_norm_reduce
    ldr  q0, [x0], #16
    fmla v16.4s, v0.4s, v0.4s
    sub  x1, x1, #4
    b    .L_clip_norm4
.L_clip_norm_reduce:
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
.L_clip_normtail:
    cbz   x1, .L_clip_norm_end
    ldr   s0, [x0], #4
    fmadd s16, s0, s0, s16
    subs  x1, x1, #1
    b.ne  .L_clip_normtail
.L_clip_norm_end:
    fsqrt s9, s16             // d9 = norm

    fcmp  s9, s8
    b.le  .L_clip_skip

    fdiv  s17, s8, s9
    dup   v17.4s, v17.s[0]

    mov  x0, x19
    mov  x1, x20
.L_clip_scale4:
    cmp  x1, #4
    b.lt .L_clip_scaletail
    ldr  q0, [x0]
    fmul v0.4s, v0.4s, v17.4s
    str  q0, [x0], #16
    sub  x1, x1, #4
    b    .L_clip_scale4
.L_clip_scaletail:
    cbz  x1, .L_clip_skip
    ldr  s0, [x0]
    fmul s0, s0, s17
    str  s0, [x0], #4
    subs x1, x1, #1
    b.ne .L_clip_scaletail

.L_clip_skip:
    fmov s0, s9

    ldp  d8,  d9,  [sp, #32]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #48
    ret

