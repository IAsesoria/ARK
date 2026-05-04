// ============================================================
// shaders/attention.metal — ARK v1.1  DEFINITIVO (Corregido)
// Metal Shading Language — Apple Silicon (M1/M2/M3/M4)
//
// Kernels exportados:
//
//   attention_scores_f16
//     Computa scores = Q × Kᵀ / √d_head para cada cabeza.
//     Q, K en FP16 → scores en FP32.
//     Soporta: multi-head attention, batch ≥ 1.
//
//   attention_softmax_f32
//     Softmax estable (max-shift) sobre scores FP32 in-place.
//     Una threadgroup por fila de scores.
//     CORREGIDO v1.1: reducción tipo árbol robusta para T no-potencia-de-2.
//
//   attention_weighted_sum_f16
//     out = softmax_scores × V
//     scores en FP32, V en FP16 → out en FP16.
//
//   attention_full_f16
//     Kernel fusionado: Q×Kᵀ + softmax + ×V en un solo dispatch.
//     CORREGIDO v1.1: reducción robusta en dot product y softmax.
//
//   attention_full_f16_causal
//     Igual que attention_full_f16 con máscara causal (GPT-style).
//     CORREGIDO v1.1: reducción robusta en dot product y softmax.
//
// Correcciones v1.1:
//   BUG #1 — attention_softmax_f32: la reducción tipo árbol asumía T
//            potencia de 2 exacta. Si T no era potencia de 2, elementos
//            de la cola quedaban fuera del max y la suma, produciendo
//            una distribución incorrecta. Fix: stride redondeado hacia
//            arriba con condición de bounds en cada paso.
//   BUG #2 — attention_full_f16 / causal: misma reducción defectuosa
//            aplicada al dot product parcial (reduce_buf). Fix idéntico.
//
// Notación de dimensiones:
//   B  = batch size
//   T  = seq_len (número de tokens)
//   H  = n_heads
//   D  = d_model (dim total)
//   Dh = d_head = D / H
//
// Layout de tensores (todos row-major / C-contiguous):
//   Q, K, V : [B, H, T, Dh]
//   scores  : [B, H, T, T]
//   out     : [B, H, T, Dh]
//
// Precisión:
//   Pesos Q/K/V: FP16
//   Scores y acumuladores: FP32
//   Salida: FP16
//
// Threadgroup sizes recomendados:
//   attention_scores_f16    : [32, 1, 1]
//   attention_softmax_f32   : [T, 1, 1]  (T ≤ 1024)
//   attention_weighted_sum  : [32, 1, 1]
//   attention_full_f16      : [Dh, 1, 1] (Dh ≤ 512)
// ============================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Constantes y utilidades
// ============================================================

// Epsilon para softmax (evita log(0))
constant float kSoftmaxEps = 1e-9f;

// rsqrt escalar: 1/sqrt(x)
inline float ark_rsqrt(float x) {
    return rsqrt(x);
}

// ============================================================
// attention_scores_f16
//
// Computa: scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:]) * scale
//   donde scale = 1/√Dh
//
// Cada thread calcula un elemento de la matriz de scores.
//
// Parámetros via buffer:
//   0: Q       [B*H*T*Dh]  f16
//   1: K       [B*H*T*Dh]  f16
//   2: scores  [B*H*T*T]   f32 (salida)
//   3: params  {B, H, T, Dh}  uint4
//
// Dispatch: (T, T, B*H)
// ============================================================
kernel void attention_scores_f16(
    device const half*    Q       [[ buffer(0) ]],
    device const half*    K       [[ buffer(1) ]],
    device       float*   scores  [[ buffer(2) ]],
    constant     uint4&   params  [[ buffer(3) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint T  = params.z;
    const uint Dh = params.w;
    const uint H  = params.y;

    const uint j   = gid.x;    // columna de K (token j)
    const uint i   = gid.y;    // fila de Q (token i)
    const uint bh  = gid.z;    // índice batch*heads combinado

    if (i >= T || j >= T) return;

    const uint qk_base = bh * T * Dh;

    device const half* q_row = Q + qk_base + i * Dh;
    device const half* k_row = K + qk_base + j * Dh;

    float acc = 0.0f;
    uint d = 0;

    // 8-wide unroll con half8
    for (; d + 8 <= Dh; d += 8) {
        half8 qv = *((device const half8*)(q_row + d));
        half8 kv = *((device const half8*)(k_row + d));
        float8 qf = float8(qv);
        float8 kf = float8(kv);
        acc += dot(qf.lo, kf.lo) + dot(qf.hi, kf.hi);
    }
    // Tail 4-wide
    for (; d + 4 <= Dh; d += 4) {
        half4 qv = *((device const half4*)(q_row + d));
        half4 kv = *((device const half4*)(k_row + d));
        acc += dot(float4(qv), float4(kv));
    }
    // Tail escalar
    for (; d < Dh; d++) {
        acc += float(q_row[d]) * float(k_row[d]);
    }

    acc *= ark_rsqrt(float(Dh));
    scores[bh * T * T + i * T + j] = acc;
}

// ============================================================
// attention_softmax_f32
//
// Softmax in-place sobre scores[bh, i, :] (fila de longitud T).
// Una threadgroup por fila. threads_per_threadgroup = T (≤ 1024).
//
// CORREGIDO v1.1: reducción tipo árbol robusta para T no-potencia-de-2.
//   La versión anterior usaba stride >>= 1, que perdía elementos
//   de la cola cuando T no era potencia de 2 exacta.
//   Fix: stride = (stride+1)/2 con guard tg_pos+stride < tg_size.
//
// Parámetros:
//   0: scores  [B*H*T*T]  f32 (in-place)
//   1: params  {B, H, T, Dh}  uint4
//
// Dispatch: (1, T, B*H)
// ============================================================
kernel void attention_softmax_f32(
    device       float*   scores  [[ buffer(0) ]],
    constant     uint4&   params  [[ buffer(1) ]],
    uint tg_pos  [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint2 gid    [[ threadgroup_position_in_grid ]],
    threadgroup float* shared_buf [[ threadgroup(0) ]]
) {
    const uint T   = params.z;
    const uint i   = gid.y;    // fila
    const uint bh  = gid.x;    // batch*heads

    const uint row_off = bh * T * T + i * T;

    const uint j = tg_pos;
    float val = (j < T) ? scores[row_off + j] : -INFINITY;

    // Cargar en shared memory
    shared_buf[tg_pos] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Reducción de MAX robusta (soporta T no-potencia-de-2) ──
    //
    // stride empieza en ceil(tg_size/2) en lugar de tg_size/2.
    // La condición tg_pos+stride < tg_size evita leer fuera de rango.
    // Cuando stride llega a 1, se hace la última comparación y se rompe
    // explícitamente (evita el bucle infinito que ocurriría si stride
    // se volviera 0 por (1+1)/2 = 1 de nuevo).
    uint current_size = tg_size;
    while (current_size > 1) {
        uint stride = (current_size + 1) / 2;
        if (tg_pos < current_size / 2) {
            shared_buf[tg_pos] = max(shared_buf[tg_pos],
                                     shared_buf[tg_pos + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        current_size = stride;
    }
    float max_val = shared_buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // exp(x - max)
    float exp_val = (j < T) ? exp(val - max_val) : 0.0f;

    // ── Reducción de SUMA robusta (misma lógica) ──
    shared_buf[tg_pos] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint current_size_sum = tg_size;
    while (current_size_sum > 1) {
        uint stride = (current_size_sum + 1) / 2;
        if (tg_pos < current_size_sum / 2) {
            shared_buf[tg_pos] += shared_buf[tg_pos + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        current_size_sum = stride;
    }
    float sum_val = shared_buf[0] + kSoftmaxEps;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Escribir probabilidades
    if (j < T) {
        scores[row_off + j] = exp_val / sum_val;
    }
}

// ============================================================
// attention_weighted_sum_f16
//
// out[b,h,i,d] = Σ_j scores[b,h,i,j] * V[b,h,j,d]
//
// scores en FP32, V en FP16, out en FP16.
// Sin cambios respecto a v1.0 — no usa reducción compartida.
//
// Parámetros:
//   0: scores  [B*H*T*T]   f32
//   1: V       [B*H*T*Dh]  f16
//   2: out     [B*H*T*Dh]  f16 (salida)
//   3: params  {B, H, T, Dh}  uint4
//
// Dispatch: (Dh, T, B*H)
// ============================================================
kernel void attention_weighted_sum_f16(
    device const float*   scores  [[ buffer(0) ]],
    device const half*    V       [[ buffer(1) ]],
    device       half*    out     [[ buffer(2) ]],
    constant     uint4&   params  [[ buffer(3) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint T  = params.z;
    const uint Dh = params.w;

    const uint d  = gid.x;
    const uint i  = gid.y;
    const uint bh = gid.z;

    if (d >= Dh || i >= T) return;

    const uint scores_base = bh * T * T + i * T;
    const uint v_base      = bh * T * Dh;

    float acc = 0.0f;
    for (uint j = 0; j < T; j++) {
        float s  = scores[scores_base + j];
        float vv = float(V[v_base + j * Dh + d]);
        acc += s * vv;
    }

    out[bh * T * Dh + i * Dh + d] = half(acc);
}

// ============================================================
// attention_full_f16
//
// Kernel FUSIONADO: Q×Kᵀ + softmax + ×V en un solo dispatch.
// Elimina dos round-trips a DRAM para la matriz de scores.
//
// CORREGIDO v1.1: reducción robusta tanto en el dot product
//   (reduce_buf) como en el softmax (scores_row). Ambas usaban
//   stride >>= 1 que fallaba para tsize/T no-potencia-de-2.
//
// Recomendado cuando T ≤ 512 y Dh ≤ 512.
//
// Parámetros:
//   0: Q       [B*H*T*Dh]  f16
//   1: K       [B*H*T*Dh]  f16
//   2: V       [B*H*T*Dh]  f16
//   3: out     [B*H*T*Dh]  f16
//   4: params  {B, H, T, Dh}  uint4
//
// Dispatch: (1, T, B*H)
// threadgroup(0): scores_row  [T]     f32
// threadgroup(1): reduce_buf  [tsize] f32
// ============================================================
kernel void attention_full_f16(
    device const half*    Q       [[ buffer(0) ]],
    device const half*    K       [[ buffer(1) ]],
    device const half*    V       [[ buffer(2) ]],
    device       half*    out     [[ buffer(3) ]],
    constant     uint4&   params  [[ buffer(4) ]],
    uint tpos    [[ thread_position_in_threadgroup ]],
    uint tsize   [[ threads_per_threadgroup ]],
    uint2 tg_id  [[ threadgroup_position_in_grid ]],
    threadgroup float* scores_row [[ threadgroup(0) ]],
    threadgroup float* reduce_buf [[ threadgroup(1) ]]
) {
    const uint T  = params.z;
    const uint Dh = params.w;

    const uint i  = tg_id.y;
    const uint bh = tg_id.x;

    const uint base = bh * T * Dh;
    device const half* q_row = Q + base + i * Dh;
    const float scale = ark_rsqrt(float(Dh));

    // ── Fase 1: scores_row[j] = dot(q[i], k[j]) * scale ──────
    for (uint j = 0; j < T; j++) {
        device const half* k_row = K + base + j * Dh;

        float partial = 0.0f;
        for (uint d = tpos; d < Dh; d += tsize) {
            partial += float(q_row[d]) * float(k_row[d]);
        }

        // Reducción de suma robusta sobre reduce_buf
        reduce_buf[tpos] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint current_size_sum = tsize;
        while (current_size_sum > 1) {
            uint stride = (current_size_sum + 1) / 2;
            if (tpos < current_size_sum / 2) {
                reduce_buf[tpos] += reduce_buf[tpos + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            current_size_sum = stride;
        }

        if (tpos == 0) {
            scores_row[j] = reduce_buf[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Fase 2: Softmax sobre scores_row ──────────────────────
    // Thread 0 calcula max y suma secuencialmente (T ≤ 512).
    // Para T mayor usar attention_softmax_f32 separado.
    if (tpos == 0) {
        float max_val = scores_row[0];
        for (uint j = 1; j < T; j++) {
            max_val = max(max_val, scores_row[j]);
        }
        float sum_val = 0.0f;
        for (uint j = 0; j < T; j++) {
            scores_row[j] = exp(scores_row[j] - max_val);
            sum_val += scores_row[j];
        }
        sum_val += kSoftmaxEps;
        for (uint j = 0; j < T; j++) {
            scores_row[j] /= sum_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Fase 3: out[i, d] = Σ_j scores_row[j] * V[bh, j, d] ─
    for (uint d = tpos; d < Dh; d += tsize) {
        float acc = 0.0f;
        for (uint j = 0; j < T; j++) {
            acc += scores_row[j] * float(V[base + j * Dh + d]);
        }
        out[base + i * Dh + d] = half(acc);
    }
}

// ============================================================
// attention_full_f16_causal
//
// Igual que attention_full_f16 pero con máscara causal:
//   scores[i,j] = -∞  si j > i
//
// CORREGIDO v1.1: reducción robusta en el dot product.
//   La reducción del softmax (thread 0 secuencial) no tenía el bug
//   ya que itera con for, pero se mantiene igual por claridad.
// ============================================================
kernel void attention_full_f16_causal(
    device const half*    Q       [[ buffer(0) ]],
    device const half*    K       [[ buffer(1) ]],
    device const half*    V       [[ buffer(2) ]],
    device       half*    out     [[ buffer(3) ]],
    constant     uint4&   params  [[ buffer(4) ]],
    uint tpos    [[ thread_position_in_threadgroup ]],
    uint tsize   [[ threads_per_threadgroup ]],
    uint2 tg_id  [[ threadgroup_position_in_grid ]],
    threadgroup float* scores_row [[ threadgroup(0) ]],
    threadgroup float* reduce_buf [[ threadgroup(1) ]]
) {
    const uint T  = params.z;
    const uint Dh = params.w;

    const uint i  = tg_id.y;
    const uint bh = tg_id.x;

    const uint base = bh * T * Dh;
    device const half* q_row = Q + base + i * Dh;

    const float scale   = ark_rsqrt(float(Dh));
    const float neg_inf = -1e9f;

    // ── Fase 1: scores con máscara causal ─────────────────────
    for (uint j = 0; j < T; j++) {
        if (j > i) {
            // Posición futura: enmascarar sin dot product
            if (tpos == 0) scores_row[j] = neg_inf;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        device const half* k_row = K + base + j * Dh;
        float partial = 0.0f;
        for (uint d = tpos; d < Dh; d += tsize) {
            partial += float(q_row[d]) * float(k_row[d]);
        }

        // Reducción robusta
        reduce_buf[tpos] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint current_size_sum = tsize;
        while (current_size_sum > 1) {
            uint stride = (current_size_sum + 1) / 2;
            if (tpos < current_size_sum / 2) {
                reduce_buf[tpos] += reduce_buf[tpos + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            current_size_sum = stride;
        }
        if (tpos == 0) scores_row[j] = reduce_buf[0] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Fase 2: Softmax (-∞ contribuye 0 a la suma) ───────────
    if (tpos == 0) {
        float max_val = scores_row[0];
        for (uint j = 1; j < T; j++) max_val = max(max_val, scores_row[j]);
        float sum_val = 0.0f;
        for (uint j = 0; j < T; j++) {
            scores_row[j] = exp(scores_row[j] - max_val);
            sum_val += scores_row[j];
        }
        sum_val += kSoftmaxEps;
        for (uint j = 0; j < T; j++) scores_row[j] /= sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Fase 3: weighted sum (solo j ≤ i por causalidad) ──────
    for (uint d = tpos; d < Dh; d += tsize) {
        float acc = 0.0f;
        for (uint j = 0; j <= i; j++) {
            acc += scores_row[j] * float(V[base + j * Dh + d]);
        }
        out[base + i * Dh + d] = half(acc);
    }
}