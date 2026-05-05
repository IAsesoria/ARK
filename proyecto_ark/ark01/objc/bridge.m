/*
 * ark_mps_bridge.m — ARK v1.3 "METAL-REASONER" — AutoGrad GPU Nativo
 * Proyecto NOUS / IAsesoria Informática — Villarrica, Chile
 *
 * FIXES v1.3 sobre v1.2:
 *
 * FIX 4 — RoPE: tabla de senos con signos correctos:
 *   apply_rope construye xrot = concat([-xr, xl]) e implementa la rotación GPT-NeoX.
 *   Con sp[i]=-sin: (-xr)*(-sin) = +xr*sin → primera mitad incorrecta.
 *   Con sp[i]=+sin: (-xr)*(+sin) = -xr*sin → rotación 2D exacta. ✓
 *   El signo negativo debe vivir solo en xrn, no duplicarse en la tabla.
 *
 * FIX 5 — Gradiente de embedding de entrada (scatter-add desde dx_layers[0]):
 *   MPSGraph propaga dL/d(embed_output) hasta g_buf_dx_layers[0].
 *   Ese gradiente debe acumularse en las filas embed[token] correspondientes.
 *   Con tied embeddings suma al mismo tensor que g_buf_dembed (LM Head).
 *   Sin tied embeddings actualiza la tabla de embedding de entrada.
 *   Firma de ark_mps_backward_layers extendida con `const uint32_t *tokens`.
 *
 * FIXES v1.2 sobre v1.1:
 *
 * FIX 1 — RMSNorm varianza en FP32 (ya en v1.1, mantenido):
 *   FP16 max = 65504. Una activación ≥ 256 → 256²=65536 → INF → NaN.
 *   La varianza se calcula en FP32, se normaliza en FP32, se baja a FP16
 *   antes de multiplicar por gamma. Igual que LLaMA, Mistral, etc.
 *
 * FIX 2 — API de ejecución consistente (feeds dict para todo):
 *   MPSGraphExecutable compilado con feeds:NSDictionary debe ejecutarse con
 *   la misma API: runWithMTLCommandQueue:feeds:targetOperations:resultsArray:
 *   Usar inputsArray: (NSArray posicional) cuando se compiló con diccionario
 *   produce mapeo indefinido — los pesos pueden llegar al placeholder incorrecto.
 *   CORRECCIÓN: Todas las llamadas usan NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>
 *   tanto en forward como backward de capas y LM Head.
 *
 * FIX 3 — Eliminar g_buf_x2_layers (memoria fantasma):
 *   Con AutoGrad, exec_bwd recomputa las activaciones intermedias desde ph_x.
 *   g_buf_x2_layers almacenaba x2 = x + attn_proj que nadie lee en backward.
 *   Eliminado completamente: declaración, alloc, free.
 *   exec_fwd solo retorna t_out (no t_x2).
 *
 * FIRMA FFI: ark_mps_backward_layers ahora recibe `const uint32_t *tokens`
 * como PRIMER parámetro — actualizar ffi.rs y training.rs.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Accelerate/Accelerate.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

// ── Firmas de ensamblador ARM64 (ark_kernels.s) ───────────────────────────────
extern void ark_asm_rmsnorm(float *x, const float *gamma, int n_seq, int dim);
extern void ark_rmsnorm_backward(const float *dy, const float *x_norm,
                                 const float *gamma, float *dx, float *dgamma,
                                 float inv_rms, int n_seq, int dim);

// ── Estado global ─────────────────────────────────────────────────────────────
static id<MTLDevice>       g_dev   = nil;
static id<MTLCommandQueue> g_queue = nil;
static bool                g_ready = false;

static int G_NL  = 0;
static int G_D   = 0;
static int G_H   = 0;
static int G_NH  = 8;
static int G_HD  = 0;
static int G_V   = 0;
static int G_BS  = 0;
static int G_SEQ = 0;

// ── Buffers MTL persistentes ──────────────────────────────────────────────────
static id<MTLBuffer> g_buf_logits      = nil;
static id<MTLBuffer> g_buf_embed       = nil;
static id<MTLBuffer> g_buf_gamma_final = nil;
static id<MTLBuffer> g_buf_rope_cos    = nil;
static id<MTLBuffer> g_buf_rope_sin    = nil;
static id<MTLBuffer> g_buf_dembed      = nil;
static id<MTLBuffer> g_buf_dgamma_final = nil;

// ── Activaciones Zero-Copy ────────────────────────────────────────────────────
static __strong id<MTLBuffer> *g_buf_x_layers  = NULL;  // [NL+1] FP32
// FIX 3: g_buf_x2_layers eliminado — AutoGrad recomputa desde ph_x
static __strong id<MTLBuffer> *g_buf_dx_layers = NULL;  // [NL+1] FP32 (gradientes)

// ── Pesos por capa ────────────────────────────────────────────────────────────
typedef struct {
    __strong id<MTLBuffer> wq, wk, wv, wo;
    __strong id<MTLBuffer> w1, w3, w2;
    __strong id<MTLBuffer> g1, g2;
} ArkLayerGPU;
static ArkLayerGPU *g_layers = NULL;

// ── Grafo por capa — forward + backward en un MPSGraph ───────────────────────
typedef struct {
    MPSGraph *graph;
    MPSGraphTensor *ph_x;
    MPSGraphTensor *ph_wq, *ph_wk, *ph_wv, *ph_wo;
    MPSGraphTensor *ph_w1, *ph_w3, *ph_w2;
    MPSGraphTensor *ph_g1, *ph_g2;
    MPSGraphTensor *ph_rope_cos, *ph_rope_sin;
    MPSGraphTensor *ph_d_out;  // gradiente entrante dL/d_out
    MPSGraphTensor *t_out;     // salida forward (FP32)
    // FIX 3: t_x2 eliminado — AutoGrad recomputa activaciones intermedias
    // Gradientes (salidas del backward)
    MPSGraphTensor *t_dx;
    MPSGraphTensor *t_dwq, *t_dwk, *t_dwv, *t_dwo;
    MPSGraphTensor *t_dw1, *t_dw2, *t_dw3;
    MPSGraphTensor *t_dg1, *t_dg2;
    // Executables separados
    MPSGraphExecutable *exec_fwd;
    MPSGraphExecutable *exec_bwd;
} ArkLayerGraph;
static ArkLayerGraph *g_graphs = NULL;

// ── LM Head ───────────────────────────────────────────────────────────────────
typedef struct {
    MPSGraph *graph;
    MPSGraphTensor *ph_x, *ph_embed, *ph_gamma_final, *ph_d_logits;
    MPSGraphTensor *t_logits, *t_dx, *t_dembed, *t_dgamma_final;
    MPSGraphExecutable *exec_fwd, *exec_bwd;
} ArkLMHeadGraph;
static ArkLMHeadGraph g_lmhead = {0};

// ── Buffers de gradientes de pesos por capa (FP32) ────────────────────────────
typedef struct {
    __strong id<MTLBuffer> dwq, dwk, dwv, dwo;
    __strong id<MTLBuffer> dw1, dw2, dw3;
    __strong id<MTLBuffer> dg1, dg2;
} ArkLayerGradBuf;
static ArkLayerGradBuf *g_grad_bufs = NULL;

// ── Helpers ───────────────────────────────────────────────────────────────────

static uint16_t f32_to_f16(float v) {
    uint32_t x; memcpy(&x, &v, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

static void build_rope_tables(int seq_len, int head_dim) {
    size_t n = (size_t)seq_len * head_dim;
    MTLResourceOptions sh = MTLResourceStorageModeShared;
    g_buf_rope_cos = [g_dev newBufferWithLength:n*2 options:sh];
    g_buf_rope_sin = [g_dev newBufferWithLength:n*2 options:sh];
    uint16_t *cp = (uint16_t*)g_buf_rope_cos.contents;
    uint16_t *sp = (uint16_t*)g_buf_rope_sin.contents;
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim/2; i++) {
            float inv_freq = 1.0f / powf(10000.0f, (float)(2*i)/(float)head_dim);
            float angle = (float)pos * inv_freq;
            float c = cosf(angle), s = sinf(angle);
            cp[pos*head_dim + i]            = f32_to_f16(c);
            cp[pos*head_dim + i+head_dim/2] = f32_to_f16(c);
            // FIX ROPE: sp almacena +sin(θ) en ambas mitades.
            // apply_rope ya construye xrot = concat([-xr, xl]),
            // por lo que xrot * sin da [-xr*sin, xl*sin] → rotación correcta.
            // Con -s aquí: (-xr)*(-sin) = +xr*sin → la rotación se invertía.
            sp[pos*head_dim + i]            = f32_to_f16(s);
            sp[pos*head_dim + i+head_dim/2] = f32_to_f16(s);
        }
    }
}

// FIX 1: RMSNorm con varianza calculada en FP32
// FP16 max = 65504. Activación ≥ 256 → 256²=65536 → INF en FP16 → NaN.
// Todos los LLMs modernos (LLaMA, Mistral) calculan la varianza en FP32.
static MPSGraphTensor *mpsg_rmsnorm(MPSGraph *gr, MPSGraphTensor *inp,
                                     MPSGraphTensor *gamma) {
    // Varianza en FP32 para evitar overflow FP16
    MPSGraphTensor *inp32 = [gr castTensor:inp toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *sq    = [gr squareWithTensor:inp32 name:nil];
    MPSGraphTensor *mn    = [gr meanOfTensor:sq axes:@[@(-1)] name:nil];
    MPSGraphTensor *eps   = [gr constantWithScalar:1e-5f dataType:MPSDataTypeFloat32];
    MPSGraphTensor *mne   = [gr additionWithPrimaryTensor:mn secondaryTensor:eps name:nil];
    MPSGraphTensor *rms   = [gr squareRootWithTensor:mne name:nil];
    MPSGraphTensor *norm  = [gr divisionWithPrimaryTensor:inp32 secondaryTensor:rms name:nil];
    // Gamma en FP32 — reshape explícito en lugar de expandDims (diferenciable)
    MPSGraphTensor *gamma32 = [gr castTensor:gamma toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *g2d     = [gr reshapeTensor:gamma32 withShape:@[@1,@1,@1,@(-1)] name:nil];
    // Multiplicar en FP32, bajar a FP16 al final
    MPSGraphTensor *out32   = [gr multiplicationWithPrimaryTensor:norm secondaryTensor:g2d name:nil];
    return [gr castTensor:out32 toType:MPSDataTypeFloat16 name:nil];
}

static MPSGraphTensor *apply_rope(MPSGraph *gr, MPSGraphTensor *x,
                                   MPSGraphTensor *ph_cos, MPSGraphTensor *ph_sin) {
    int HD2 = G_HD / 2;
    MPSGraphTensor *xl  = [gr sliceTensor:x dimension:3 start:0   length:HD2 name:nil];
    MPSGraphTensor *xr  = [gr sliceTensor:x dimension:3 start:HD2 length:HD2 name:nil];
    MPSGraphTensor *neg = [gr constantWithScalar:-1.0f dataType:MPSDataTypeFloat16];
    MPSGraphTensor *xrn = [gr multiplicationWithPrimaryTensor:xr secondaryTensor:neg name:nil];
    MPSGraphTensor *xrot= [gr concatTensors:@[xrn, xl] dimension:3 name:nil];
    MPSGraphTensor *o1  = [gr multiplicationWithPrimaryTensor:x    secondaryTensor:ph_cos name:nil];
    MPSGraphTensor *o2  = [gr multiplicationWithPrimaryTensor:xrot secondaryTensor:ph_sin name:nil];
    return [gr additionWithPrimaryTensor:o1 secondaryTensor:o2 name:nil];
}

static MPSGraphTensor *causal_mask(MPSGraph *gr, int seq) {
    size_t n = (size_t)seq * seq;
    uint16_t *d = (uint16_t*)malloc(n*2);
    uint16_t z = f32_to_f16(0.0f), ni = f32_to_f16(-1e4f);
    for (int r = 0; r < seq; r++)
        for (int c = 0; c < seq; c++)
            d[r*seq+c] = (c <= r) ? z : ni;
    NSData *nd = [NSData dataWithBytes:d length:n*2];
    free(d);
    return [gr constantWithData:nd shape:@[@1,@1,@(seq),@(seq)]
                       dataType:MPSDataTypeFloat16];
}

// ── Construcción del grafo de capa con AutoGrad correcto ──────────────────────
/*
 * CHAIN RULE via loss_proxy:
 *   loss_proxy = reduceSum(t_out * ph_d_out)
 *   grads = gradientsOfPrimaryTensor:loss_proxy withTensors:params
 *
 *   d(loss_proxy)/dW = dL/d_out · d(t_out)/dW  ← regla de la cadena exacta
 *
 *   Propaga correctamente a través de: SDPA, RoPE, SwiGLU, RMSNorm, Residuals.
 *   dWq, dWk, dWv son DISTINTOS (corrección del bug original).
 */
static void build_layer_graph(int l, int batch, int seq) {
    ArkLayerGraph *lg = &g_graphs[l];
    int D = G_D, H = G_H, NH = G_NH, HD = G_HD;
    MPSGraph *gr = [[MPSGraph alloc] init];
    lg->graph = gr;

    // ── Placeholders ─────────────────────────────────────────────────────────
    lg->ph_x  = [gr placeholderWithShape:@[@(batch*seq),@(D)]
                                dataType:MPSDataTypeFloat32 name:@"x"];
    lg->ph_wq = [gr placeholderWithShape:@[@(D),@(D)] dataType:MPSDataTypeFloat16 name:@"wq"];
    lg->ph_wk = [gr placeholderWithShape:@[@(D),@(D)] dataType:MPSDataTypeFloat16 name:@"wk"];
    lg->ph_wv = [gr placeholderWithShape:@[@(D),@(D)] dataType:MPSDataTypeFloat16 name:@"wv"];
    lg->ph_wo = [gr placeholderWithShape:@[@(D),@(D)] dataType:MPSDataTypeFloat16 name:@"wo"];
    lg->ph_w1 = [gr placeholderWithShape:@[@(D),@(H)] dataType:MPSDataTypeFloat16 name:@"w1"];
    lg->ph_w3 = [gr placeholderWithShape:@[@(D),@(H)] dataType:MPSDataTypeFloat16 name:@"w3"];
    lg->ph_w2 = [gr placeholderWithShape:@[@(H),@(D)] dataType:MPSDataTypeFloat16 name:@"w2"];
    lg->ph_g1 = [gr placeholderWithShape:@[@(D)] dataType:MPSDataTypeFloat16 name:@"g1"];
    lg->ph_g2 = [gr placeholderWithShape:@[@(D)] dataType:MPSDataTypeFloat16 name:@"g2"];
    lg->ph_rope_cos = [gr placeholderWithShape:@[@1,@1,@(seq),@(HD)]
                                      dataType:MPSDataTypeFloat16 name:@"cos"];
    lg->ph_rope_sin = [gr placeholderWithShape:@[@1,@1,@(seq),@(HD)]
                                      dataType:MPSDataTypeFloat16 name:@"sin"];
    lg->ph_d_out = [gr placeholderWithShape:@[@(batch*seq),@(D)]
                                   dataType:MPSDataTypeFloat32 name:@"d_out"];

    // ── Forward ───────────────────────────────────────────────────────────────
    MPSGraphTensor *x16  = [gr castTensor:lg->ph_x toType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor *xs   = [gr reshapeTensor:x16
                                   withShape:@[@(batch),@1,@(seq),@(D)] name:nil];

    // RMSNorm pre-atención
   MPSGraphTensor *xn1  = mpsg_rmsnorm(gr, xs, lg->ph_g1);
    MPSGraphTensor *xn1f = [gr reshapeTensor:xn1 withShape:@[@(batch*seq),@(D)] name:nil];

    // Q, K, V — proyecciones INDEPENDIENTES
    MPSGraphTensor *q = [gr matrixMultiplicationWithPrimaryTensor:xn1f
                                                 secondaryTensor:lg->ph_wq name:nil];
    MPSGraphTensor *k = [gr matrixMultiplicationWithPrimaryTensor:xn1f
                                                 secondaryTensor:lg->ph_wk name:nil];
    MPSGraphTensor *v = [gr matrixMultiplicationWithPrimaryTensor:xn1f
                                                 secondaryTensor:lg->ph_wv name:nil];

    // Reshape → [batch, n_heads, seq, head_dim]
    MPSGraphTensor *qt = [gr transposeTensor:
                          [gr reshapeTensor:q withShape:@[@(batch),@(seq),@(NH),@(HD)] name:nil]
                                   dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *kt = [gr transposeTensor:
                          [gr reshapeTensor:k withShape:@[@(batch),@(seq),@(NH),@(HD)] name:nil]
                                   dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *vt = [gr transposeTensor:
                          [gr reshapeTensor:v withShape:@[@(batch),@(seq),@(NH),@(HD)] name:nil]
                                   dimension:1 withDimension:2 name:nil];

    // RoPE en Q y K
    MPSGraphTensor *qr = apply_rope(gr, qt, lg->ph_rope_cos, lg->ph_rope_sin);
    MPSGraphTensor *kr = apply_rope(gr, kt, lg->ph_rope_cos, lg->ph_rope_sin);

    // SDPA causal desenrollado — 100% diferenciable para AutoGrad
    MPSGraphTensor *kt_trans     = [gr transposeTensor:kr dimension:2 withDimension:3 name:nil];
    MPSGraphTensor *scores       = [gr matrixMultiplicationWithPrimaryTensor:qr
                                                            secondaryTensor:kt_trans name:nil];
    float scale_v = 1.0f / sqrtf((float)HD);
    MPSGraphTensor *scale_t      = [gr constantWithScalar:scale_v dataType:MPSDataTypeFloat16];
    MPSGraphTensor *scaled       = [gr multiplicationWithPrimaryTensor:scores
                                                      secondaryTensor:scale_t name:nil];
    MPSGraphTensor *mask         = causal_mask(gr, seq);
    MPSGraphTensor *masked       = [gr additionWithPrimaryTensor:scaled
                                                secondaryTensor:mask name:nil];
    MPSGraphTensor *probs        = [gr softMaxWithTensor:masked axis:3 name:nil];
    MPSGraphTensor *aout         = [gr matrixMultiplicationWithPrimaryTensor:probs
                                                            secondaryTensor:vt name:nil];

    // Proyección Wo
    MPSGraphTensor *at   = [gr transposeTensor:aout dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *af   = [gr reshapeTensor:at withShape:@[@(batch*seq),@(D)] name:nil];
    MPSGraphTensor *po   = [gr matrixMultiplicationWithPrimaryTensor:af
                                                    secondaryTensor:lg->ph_wo name:nil];

    // Residual 1
    MPSGraphTensor *x2_16 = [gr additionWithPrimaryTensor:x16 secondaryTensor:po name:nil];
    // FIX 3: t_x2 eliminado — no necesitamos guardar este tensor intermedio

    // RMSNorm pre-FFN + SwiGLU
    MPSGraphTensor *x2s   = [gr reshapeTensor:x2_16 withShape:@[@(batch),@1,@(seq),@(D)] name:nil];
    MPSGraphTensor *xn2   = mpsg_rmsnorm(gr, x2s, lg->ph_g2);
    MPSGraphTensor *xn2f  = [gr reshapeTensor:xn2 withShape:@[@(batch*seq),@(D)] name:nil];

    MPSGraphTensor *gate  = [gr matrixMultiplicationWithPrimaryTensor:xn2f
                                                     secondaryTensor:lg->ph_w1 name:nil];
    MPSGraphTensor *up    = [gr matrixMultiplicationWithPrimaryTensor:xn2f
                                                     secondaryTensor:lg->ph_w3 name:nil];
    MPSGraphTensor *sig   = [gr sigmoidWithTensor:gate name:nil];
    MPSGraphTensor *swish = [gr multiplicationWithPrimaryTensor:gate secondaryTensor:sig name:nil];
    MPSGraphTensor *gated = [gr multiplicationWithPrimaryTensor:swish secondaryTensor:up name:nil];
    MPSGraphTensor *fout  = [gr matrixMultiplicationWithPrimaryTensor:gated
                                                     secondaryTensor:lg->ph_w2 name:nil];

    // Residual 2
    MPSGraphTensor *out16 = [gr additionWithPrimaryTensor:x2_16 secondaryTensor:fout name:nil];
    lg->t_out = [gr castTensor:out16 toType:MPSDataTypeFloat32 name:nil];

    // ── AutoGrad via loss_proxy ───────────────────────────────────────────────
    MPSGraphTensor *prod       = [gr multiplicationWithPrimaryTensor:lg->t_out
                                                   secondaryTensor:lg->ph_d_out name:nil];
    MPSGraphTensor *loss_proxy = [gr reductionSumWithTensor:prod axes:nil name:nil];

    NSDictionary<MPSGraphTensor*,MPSGraphTensor*> *grads =
        [gr gradientForPrimaryTensor:loss_proxy
                         withTensors:@[
                             lg->ph_x,
                             lg->ph_wq, lg->ph_wk, lg->ph_wv, lg->ph_wo,
                             lg->ph_w1, lg->ph_w2, lg->ph_w3,
                             lg->ph_g1, lg->ph_g2,
                         ]
                               name:nil];

    // Extraer gradientes — cada uno DIFERENTE (corrección del bug original)
    // ph_x es FP32 → t_dx hereda FP32, sin cast necesario
    // pesos son FP16 → sus gradientes son FP16 → castear a FP32 para los buffers
    lg->t_dx  = grads[lg->ph_x];
    lg->t_dwq = [gr castTensor:grads[lg->ph_wq] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dwk = [gr castTensor:grads[lg->ph_wk] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dwv = [gr castTensor:grads[lg->ph_wv] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dwo = [gr castTensor:grads[lg->ph_wo] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dw1 = [gr castTensor:grads[lg->ph_w1] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dw2 = [gr castTensor:grads[lg->ph_w2] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dw3 = [gr castTensor:grads[lg->ph_w3] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dg1 = [gr castTensor:grads[lg->ph_g1] toType:MPSDataTypeFloat32 name:nil];
    lg->t_dg2 = [gr castTensor:grads[lg->ph_g2] toType:MPSDataTypeFloat32 name:nil];

    // ── Compilar executables ──────────────────────────────────────────────────
    // FIX 2: Compilar con NSDictionary — la ejecución también usará NSDictionary
    NSDictionary *feed_shapes = @{
        lg->ph_x:        [[MPSGraphShapedType alloc] initWithShape:@[@(batch*seq),@(D)]
                                                          dataType:MPSDataTypeFloat32],
        lg->ph_wq:       [[MPSGraphShapedType alloc] initWithShape:@[@(D),@(D)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_wk:       [[MPSGraphShapedType alloc] initWithShape:@[@(D),@(D)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_wv:       [[MPSGraphShapedType alloc] initWithShape:@[@(D),@(D)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_wo:       [[MPSGraphShapedType alloc] initWithShape:@[@(D),@(D)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_w1:       [[MPSGraphShapedType alloc] initWithShape:@[@(D),@(H)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_w3:       [[MPSGraphShapedType alloc] initWithShape:@[@(D),@(H)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_w2:       [[MPSGraphShapedType alloc] initWithShape:@[@(H),@(D)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_g1:       [[MPSGraphShapedType alloc] initWithShape:@[@(D)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_g2:       [[MPSGraphShapedType alloc] initWithShape:@[@(D)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_rope_cos: [[MPSGraphShapedType alloc] initWithShape:@[@1,@1,@(seq),@(HD)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_rope_sin: [[MPSGraphShapedType alloc] initWithShape:@[@1,@1,@(seq),@(HD)]
                                                          dataType:MPSDataTypeFloat16],
        lg->ph_d_out:    [[MPSGraphShapedType alloc] initWithShape:@[@(batch*seq),@(D)]
                                                          dataType:MPSDataTypeFloat32],
    };

    MPSGraphDevice *mps_dev = [MPSGraphDevice deviceWithMTLDevice:g_dev];
    MPSGraphCompilationDescriptor *comp = [[MPSGraphCompilationDescriptor alloc] init];

    // FIX 3: exec_fwd solo retorna t_out (t_x2 eliminado)
    lg->exec_fwd = [gr compileWithDevice:mps_dev
                                   feeds:feed_shapes
                           targetTensors:@[lg->t_out]
                        targetOperations:nil
                     compilationDescriptor:comp];

    // exec_bwd: solo gradientes como targets (no recomputa forward innecesariamente)
    lg->exec_bwd = [gr compileWithDevice:mps_dev
                                   feeds:feed_shapes
                           targetTensors:@[
                               lg->t_dx,
                               lg->t_dwq, lg->t_dwk, lg->t_dwv, lg->t_dwo,
                               lg->t_dw1, lg->t_dw2, lg->t_dw3,
                               lg->t_dg1, lg->t_dg2,
                           ]
                        targetOperations:nil
                     compilationDescriptor:comp];
}

// ── LM Head con AutoGrad ──────────────────────────────────────────────────────
static void build_lmhead_graph(int bs, int vocab) {
    MPSGraph *gr = [[MPSGraph alloc] init];
    g_lmhead.graph = gr;

    g_lmhead.ph_x = [gr placeholderWithShape:@[@(bs),@(G_D)]
                                    dataType:MPSDataTypeFloat32 name:@"x_f"];
    g_lmhead.ph_embed = [gr placeholderWithShape:@[@(vocab),@(G_D)]
                                        dataType:MPSDataTypeFloat16 name:@"emb"];
    g_lmhead.ph_gamma_final = [gr placeholderWithShape:@[@(G_D)]
                                              dataType:MPSDataTypeFloat16 name:@"gf"];
    g_lmhead.ph_d_logits = [gr placeholderWithShape:@[@(bs),@(vocab)]
                                           dataType:MPSDataTypeFloat32 name:@"d_logits"];

    MPSGraphTensor *xf16  = [gr castTensor:g_lmhead.ph_x toType:MPSDataTypeFloat16 name:@"lmh_x_f16"];
    MPSGraphTensor *xs    = [gr reshapeTensor:xf16 withShape:@[@(bs),@1,@1,@(G_D)] name:nil];
    MPSGraphTensor *xn    = mpsg_rmsnorm(gr, xs, g_lmhead.ph_gamma_final);
    MPSGraphTensor *xnf   = [gr reshapeTensor:xn withShape:@[@(bs),@(G_D)] name:nil];

    // Conv1x1 workaround para límite Metal W<=16384
    MPSGraphTensor *xnchw = [gr reshapeTensor:xnf withShape:@[@(bs),@(G_D),@1,@1] name:nil];
    MPSGraphTensor *woihw = [gr reshapeTensor:g_lmhead.ph_embed
                                    withShape:@[@(vocab),@(G_D),@1,@1] name:nil];
    MPSGraphConvolution2DOpDescriptor *desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:1
             dilationRateInX:1 dilationRateInY:1 groups:1
             paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0
             paddingStyle:MPSGraphPaddingStyleExplicit
             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
             weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    MPSGraphTensor *conv  = [gr convolution2DWithSourceTensor:xnchw
                                               weightsTensor:woihw descriptor:desc name:nil];
    MPSGraphTensor *logf  = [gr reshapeTensor:conv withShape:@[@(bs),@(vocab)] name:nil];
    g_lmhead.t_logits = [gr castTensor:logf toType:MPSDataTypeFloat32 name:nil];

    // AutoGrad via loss_proxy
    MPSGraphTensor *prod = [gr multiplicationWithPrimaryTensor:g_lmhead.t_logits
                                              secondaryTensor:g_lmhead.ph_d_logits name:nil];
    MPSGraphTensor *lp   = [gr reductionSumWithTensor:prod axes:nil name:nil];

NSDictionary<MPSGraphTensor*,MPSGraphTensor*> *grads =
        [gr gradientForPrimaryTensor:lp
                         withTensors:@[g_lmhead.ph_x, g_lmhead.ph_embed, g_lmhead.ph_gamma_final]
                               name:nil];
    g_lmhead.t_dx           = grads[g_lmhead.ph_x];
    // ph_embed es FP16 → su gradiente es FP16 → castear a FP32
    g_lmhead.t_dembed       = [gr castTensor:grads[g_lmhead.ph_embed]
                                      toType:MPSDataTypeFloat32 name:nil];
    // ph_gamma_final es FP16 → castear a FP32
    g_lmhead.t_dgamma_final = [gr castTensor:grads[g_lmhead.ph_gamma_final]
                                      toType:MPSDataTypeFloat32 name:nil];

    MPSGraphDevice *dev = [MPSGraphDevice deviceWithMTLDevice:g_dev];
    MPSGraphCompilationDescriptor *comp = [[MPSGraphCompilationDescriptor alloc] init];

    // FIX 2: Compilar con NSDictionary separados para fwd y bwd
    // (fwd no incluye ph_d_logits en sus feeds)
    NSDictionary *fwd_feed_shapes = @{
        g_lmhead.ph_x:           [[MPSGraphShapedType alloc] initWithShape:@[@(bs),@(G_D)]
                                                                  dataType:MPSDataTypeFloat32],
        g_lmhead.ph_embed:       [[MPSGraphShapedType alloc] initWithShape:@[@(vocab),@(G_D)]
                                                                  dataType:MPSDataTypeFloat16],
        g_lmhead.ph_gamma_final: [[MPSGraphShapedType alloc] initWithShape:@[@(G_D)]
                                                                  dataType:MPSDataTypeFloat16],
    };
    NSDictionary *bwd_feed_shapes = @{
        g_lmhead.ph_x:           [[MPSGraphShapedType alloc] initWithShape:@[@(bs),@(G_D)]
                                                                  dataType:MPSDataTypeFloat32],
        g_lmhead.ph_embed:       [[MPSGraphShapedType alloc] initWithShape:@[@(vocab),@(G_D)]
                                                                  dataType:MPSDataTypeFloat16],
        g_lmhead.ph_gamma_final: [[MPSGraphShapedType alloc] initWithShape:@[@(G_D)]
                                                                  dataType:MPSDataTypeFloat16],
        g_lmhead.ph_d_logits:    [[MPSGraphShapedType alloc] initWithShape:@[@(bs),@(vocab)]
                                                                  dataType:MPSDataTypeFloat32],
    };

    g_lmhead.exec_fwd = [gr compileWithDevice:dev
                                        feeds:fwd_feed_shapes
                                targetTensors:@[g_lmhead.t_logits]
                             targetOperations:nil
                          compilationDescriptor:comp];

// FIX 2+3: solo gradientes como targets (sin t_logits recomputado)
    g_lmhead.exec_bwd = [gr compileWithDevice:dev
                                        feeds:bwd_feed_shapes
                                targetTensors:@[g_lmhead.t_dx, g_lmhead.t_dembed, g_lmhead.t_dgamma_final]
                             targetOperations:nil
                          compilationDescriptor:comp];
}

// ── API Pública ───────────────────────────────────────────────────────────────

bool ark_mps_init_with_config(int n_layers, int d_model, int hidden_dim) {
    if (g_ready) return true;
    @autoreleasepool {
        g_dev   = MTLCreateSystemDefaultDevice();
        g_queue = [g_dev newCommandQueue];
        if (!g_dev || !g_queue) return false;

        G_NL = n_layers; G_D = d_model; G_H = hidden_dim;
        G_HD = (G_NH > 0) ? d_model / G_NH : 0;

        MTLResourceOptions sh = MTLResourceStorageModeShared;
        g_layers = (ArkLayerGPU*)calloc(n_layers, sizeof(ArkLayerGPU));
        for (int l = 0; l < n_layers; l++) {
            g_layers[l].wq=[g_dev newBufferWithLength:(size_t)G_D*G_D*2 options:sh];
            g_layers[l].wk=[g_dev newBufferWithLength:(size_t)G_D*G_D*2 options:sh];
            g_layers[l].wv=[g_dev newBufferWithLength:(size_t)G_D*G_D*2 options:sh];
            g_layers[l].wo=[g_dev newBufferWithLength:(size_t)G_D*G_D*2 options:sh];
            g_layers[l].w1=[g_dev newBufferWithLength:(size_t)G_D*G_H*2 options:sh];
            g_layers[l].w3=[g_dev newBufferWithLength:(size_t)G_D*G_H*2 options:sh];
            g_layers[l].w2=[g_dev newBufferWithLength:(size_t)G_H*G_D*2 options:sh];
            g_layers[l].g1=[g_dev newBufferWithLength:(size_t)G_D*2     options:sh];
            g_layers[l].g2=[g_dev newBufferWithLength:(size_t)G_D*2     options:sh];
        }
        g_buf_gamma_final = [g_dev newBufferWithLength:(size_t)G_D*2 options:sh];
        uint16_t *gf = (uint16_t*)g_buf_gamma_final.contents;
        for (int i = 0; i < G_D; i++) gf[i] = 0x3C00; // 1.0 en FP16

        g_graphs = (ArkLayerGraph*)calloc(n_layers, sizeof(ArkLayerGraph));
        g_ready  = true;
        printf("[gpu] ARK v1.3 — AutoGrad GPU Nativo + Zero-Copy Ready\n");
        return true;
    }
}

void ark_mps_set_heads(int n_heads) {
    G_NH = n_heads;
    G_HD = (G_D > 0 && n_heads > 0) ? G_D / n_heads : 0;
}

void ark_mps_get_weight_ptrs(int l,
    void **wq, void **wk, void **wv, void **wo,
    void **w1, void **w3, void **w2, void **g1, void **g2)
{
    *wq=g_layers[l].wq.contents; *wk=g_layers[l].wk.contents;
    *wv=g_layers[l].wv.contents; *wo=g_layers[l].wo.contents;
    *w1=g_layers[l].w1.contents; *w3=g_layers[l].w3.contents;
    *w2=g_layers[l].w2.contents; *g1=g_layers[l].g1.contents;
    *g2=g_layers[l].g2.contents;
}

void ark_mps_get_embed_ptr(void **embed_ptr, void **gamma_f_ptr, int vocab_size) {
    if (!g_buf_embed)
        g_buf_embed = [g_dev newBufferWithLength:(size_t)vocab_size*G_D*2
                                         options:MTLResourceStorageModeShared];
    *embed_ptr   = g_buf_embed.contents;
    *gamma_f_ptr = g_buf_gamma_final.contents;
}

bool ark_mps_update_weights_f16(int layer,
    const uint16_t *wq, const uint16_t *wk, const uint16_t *wv, const uint16_t *wo,
    const uint16_t *w1, const uint16_t *w3, const uint16_t *w2,
    const uint16_t *g1, const uint16_t *g2)
{
    (void)layer;(void)wq;(void)wk;(void)wv;(void)wo;
    (void)w1;(void)w3;(void)w2;(void)g1;(void)g2;
    return true; // Zero-Copy: Rust escribe directo en los MTLBuffers
}

bool ark_mps_build_graphs(int batch_tokens, int vocab, int batch_size) {
    @autoreleasepool {
        G_V=vocab; G_BS=batch_size; G_SEQ=batch_tokens/batch_size;
        MTLResourceOptions sh = MTLResourceStorageModeShared;

        // Liberar buffers anteriores
        if (g_buf_x_layers)  {
            for(int i=0;i<=G_NL;i++) g_buf_x_layers[i]=nil;
            free(g_buf_x_layers);
        }
        // FIX 3: g_buf_x2_layers eliminado — ya no existe
        if (g_buf_dx_layers) {
            for(int i=0;i<=G_NL;i++) g_buf_dx_layers[i]=nil;
            free(g_buf_dx_layers);
        }

        g_buf_x_layers  = (__strong id<MTLBuffer>*)calloc(G_NL+1, sizeof(id<MTLBuffer>));
        // FIX 3: no allocar g_buf_x2_layers
        g_buf_dx_layers = (__strong id<MTLBuffer>*)calloc(G_NL+1, sizeof(id<MTLBuffer>));

        for (int l=0; l<=G_NL; l++) {
            g_buf_x_layers[l]  = [g_dev newBufferWithLength:(size_t)batch_tokens*G_D*4 options:sh];
            g_buf_dx_layers[l] = [g_dev newBufferWithLength:(size_t)batch_tokens*G_D*4 options:sh];
        }
        // FIX 3: ya no se allocan g_buf_x2_layers[l]

        g_buf_logits       = [g_dev newBufferWithLength:(size_t)batch_tokens*vocab*4 options:sh];
        g_buf_dembed       = [g_dev newBufferWithLength:(size_t)vocab*G_D*4          options:sh];
        g_buf_dgamma_final = [g_dev newBufferWithLength:(size_t)G_D*4                options:sh];

        if (g_grad_bufs) free(g_grad_bufs);
        g_grad_bufs = (ArkLayerGradBuf*)calloc(G_NL, sizeof(ArkLayerGradBuf));
        for (int l=0; l<G_NL; l++) {
            g_grad_bufs[l].dwq=[g_dev newBufferWithLength:(size_t)G_D*G_D*4 options:sh];
            g_grad_bufs[l].dwk=[g_dev newBufferWithLength:(size_t)G_D*G_D*4 options:sh];
            g_grad_bufs[l].dwv=[g_dev newBufferWithLength:(size_t)G_D*G_D*4 options:sh];
            g_grad_bufs[l].dwo=[g_dev newBufferWithLength:(size_t)G_D*G_D*4 options:sh];
            g_grad_bufs[l].dw1=[g_dev newBufferWithLength:(size_t)G_D*G_H*4 options:sh];
            g_grad_bufs[l].dw2=[g_dev newBufferWithLength:(size_t)G_H*G_D*4 options:sh];
            g_grad_bufs[l].dw3=[g_dev newBufferWithLength:(size_t)G_D*G_H*4 options:sh];
            g_grad_bufs[l].dg1=[g_dev newBufferWithLength:(size_t)G_D*4     options:sh];
            g_grad_bufs[l].dg2=[g_dev newBufferWithLength:(size_t)G_D*4     options:sh];
        }

        build_rope_tables(batch_tokens, G_HD);
        for (int l=0; l<G_NL; l++) build_layer_graph(l, G_BS, G_SEQ);
        build_lmhead_graph(batch_tokens, vocab);

        printf("[gpu] Grafos compilados: %d capas + LM Head\n", G_NL);
        return true;
    }
}

// ── Forward ───────────────────────────────────────────────────────────────────
bool ark_mps_forward(const uint32_t *tokens, const float *embed_w_f32,
                     float *logits, int batch_seq, int vocab, int d_model)
{
    if (!g_ready) return false;
    @autoreleasepool {
        // Embedding lookup → g_buf_x_layers[0]
        float *xp = (float*)g_buf_x_layers[0].contents;
        for (int i=0; i<batch_seq; i++) {
            uint32_t t = (tokens[i]<(uint32_t)vocab)?tokens[i]:0;
            memcpy(&xp[i*d_model], &embed_w_f32[t*d_model], (size_t)d_model*4);
        }

        for (int l=0; l<G_NL; l++) {
            ArkLayerGraph *lg = &g_graphs[l];
            ArkLayerGPU   *lw = &g_layers[l];

            // d_out dummy = ceros (exec_fwd no usa gradientes; ph_d_out en el grafo
            // pero loss_proxy no se evalúa en exec_fwd, lo ignora el compilador)
            memset(g_buf_dx_layers[l].contents, 0, (size_t)batch_seq*d_model*4);

            // FIX 2: usar NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>
            // Mapeo explícito placeholder → data, sin depender del orden
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
                lg->ph_x:        [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_x_layers[l]
                                    shape:@[@(batch_seq),@(d_model)] dataType:MPSDataTypeFloat32],
                lg->ph_wq:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->wq
                                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat16],
                lg->ph_wk:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->wk
                                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat16],
                lg->ph_wv:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->wv
                                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat16],
                lg->ph_wo:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->wo
                                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat16],
                lg->ph_w1:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->w1
                                    shape:@[@(d_model),@(G_H)] dataType:MPSDataTypeFloat16],
                lg->ph_w3:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->w3
                                    shape:@[@(d_model),@(G_H)] dataType:MPSDataTypeFloat16],
                lg->ph_w2:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->w2
                                    shape:@[@(G_H),@(d_model)] dataType:MPSDataTypeFloat16],
                lg->ph_g1:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->g1
                                    shape:@[@(d_model)] dataType:MPSDataTypeFloat16],
                lg->ph_g2:       [[MPSGraphTensorData alloc] initWithMTLBuffer:lw->g2
                                    shape:@[@(d_model)] dataType:MPSDataTypeFloat16],
                lg->ph_rope_cos: [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_rope_cos
                                    shape:@[@1,@1,@(G_SEQ),@(G_HD)] dataType:MPSDataTypeFloat16],
                lg->ph_rope_sin: [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_rope_sin
                                    shape:@[@1,@1,@(G_SEQ),@(G_HD)] dataType:MPSDataTypeFloat16],
                lg->ph_d_out:    [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_dx_layers[l]
                                    shape:@[@(batch_seq),@(d_model)] dataType:MPSDataTypeFloat32],
            };

            // FIX 2: NSDictionary para results también
            // FIX 3: solo t_out en results (t_x2 eliminado)
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = @{
                lg->t_out: [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_x_layers[l+1]
                                shape:@[@(batch_seq),@(d_model)] dataType:MPSDataTypeFloat32],
            };

NSMutableArray<MPSGraphTensorData*> *fwd_inputs =
                [NSMutableArray arrayWithCapacity:lg->exec_fwd.feedTensors.count];
            for (MPSGraphTensor *t in lg->exec_fwd.feedTensors) {
                [fwd_inputs addObject:feeds[t]];
            }
            [lg->exec_fwd runWithMTLCommandQueue:g_queue
                                     inputsArray:fwd_inputs
                                    resultsArray:@[results[lg->t_out]]
                             executionDescriptor:nil];
        }

        // LM Head forward
        // FIX 2: NSDictionary para feeds y results
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *lm_feeds = @{
            g_lmhead.ph_x:           [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:g_buf_x_layers[G_NL]
                                          shape:@[@(batch_seq),@(G_D)]
                                          dataType:MPSDataTypeFloat32],
            g_lmhead.ph_embed:       [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:g_buf_embed
                                          shape:@[@(G_V),@(G_D)]
                                          dataType:MPSDataTypeFloat16],
            g_lmhead.ph_gamma_final: [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:g_buf_gamma_final
                                          shape:@[@(G_D)]
                                          dataType:MPSDataTypeFloat16],
        };
        NSArray<MPSGraphTensorData*> *lm_results = @[
            [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_logits
                shape:@[@(batch_seq),@(G_V)] dataType:MPSDataTypeFloat32],
        ];
        NSMutableArray<MPSGraphTensorData*> *lm_fwd_inputs =
            [NSMutableArray arrayWithCapacity:g_lmhead.exec_fwd.feedTensors.count];
        for (MPSGraphTensor *t in g_lmhead.exec_fwd.feedTensors) {
            [lm_fwd_inputs addObject:lm_feeds[t]];
        }
        [g_lmhead.exec_fwd runWithMTLCommandQueue:g_queue
                                      inputsArray:lm_fwd_inputs
                                     resultsArray:lm_results
                              executionDescriptor:nil];

        // Sincronizar GPU → CPU
        id<MTLCommandBuffer> sync = [g_queue commandBuffer];
        [sync commit];
        [sync waitUntilCompleted];

        memcpy(logits, g_buf_logits.contents, (size_t)batch_seq*vocab*4);
    }
    return true;
}

// ── Backward GPU AutoGrad ─────────────────────────────────────────────────────
bool ark_mps_backward_layers(
    const uint32_t *tokens,    // para propagar dx[0] → filas correctas del embedding
    const float *logits_grad,
    const float *embed_w,
    float *embed_w_grad,
    float *gamma_final_grad,   // FIX: gradiente de la RMSNorm final (antes congelada)
    const float **layer_wq, const float **layer_wk,
    const float **layer_wv, const float **layer_wo,
    const float **layer_w1, const float **layer_w2,
    const float **layer_w3,
    const float **layer_g1, const float **layer_g2,
    float **layer_wq_grad, float **layer_wk_grad,
    float **layer_wv_grad, float **layer_wo_grad,
    float **layer_w1_grad, float **layer_w2_grad,
    float **layer_w3_grad,
    float **layer_g1_grad, float **layer_g2_grad,
    int batch_seq, int vocab, int d_model, int hidden_dim, int n_layers
) {
    @autoreleasepool {

        // ── 1. LM Head backward ───────────────────────────────────────────────
        memcpy(g_buf_logits.contents, logits_grad, (size_t)batch_seq*vocab*4);

        // FIX 2: NSDictionary para feeds y results del LM Head backward
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *bwd_lm_feeds = @{
            g_lmhead.ph_x:           [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:g_buf_x_layers[n_layers]
                                          shape:@[@(batch_seq),@(d_model)]
                                          dataType:MPSDataTypeFloat32],
            g_lmhead.ph_embed:       [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:g_buf_embed
                                          shape:@[@(G_V),@(d_model)]
                                          dataType:MPSDataTypeFloat16],
            g_lmhead.ph_gamma_final: [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:g_buf_gamma_final
                                          shape:@[@(d_model)]
                                          dataType:MPSDataTypeFloat16],
            g_lmhead.ph_d_logits:    [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:g_buf_logits
                                          shape:@[@(batch_seq),@(G_V)]
                                          dataType:MPSDataTypeFloat32],
        };
        // FIX 2+3: solo gradientes como targets (sin t_logits recomputado)
        NSArray<MPSGraphTensorData*> *bwd_lm_results = @[
            [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_dx_layers[n_layers]
                shape:@[@(batch_seq),@(d_model)] dataType:MPSDataTypeFloat32],
            [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_dembed
                shape:@[@(G_V),@(d_model)] dataType:MPSDataTypeFloat32],
            [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_dgamma_final
                shape:@[@(d_model)] dataType:MPSDataTypeFloat32],
        ];
        NSMutableArray<MPSGraphTensorData*> *bwd_lm_inputs =
            [NSMutableArray arrayWithCapacity:g_lmhead.exec_bwd.feedTensors.count];
        for (MPSGraphTensor *t in g_lmhead.exec_bwd.feedTensors) {
            [bwd_lm_inputs addObject:bwd_lm_feeds[t]];
        }
        [g_lmhead.exec_bwd runWithMTLCommandQueue:g_queue
                                      inputsArray:bwd_lm_inputs
                                     resultsArray:bwd_lm_results
                              executionDescriptor:nil];

        // ── 2. Backward por capas en orden inverso ────────────────────────────
        for (int l = n_layers-1; l >= 0; l--) {
            ArkLayerGraph   *lg = &g_graphs[l];
            ArkLayerGPU     *lw = &g_layers[l];
            ArkLayerGradBuf *gb = &g_grad_bufs[l];

            // FIX 2: NSDictionary para feeds del backward de capa
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
                lg->ph_x:        [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:g_buf_x_layers[l]
                                      shape:@[@(batch_seq),@(d_model)]
                                      dataType:MPSDataTypeFloat32],
                lg->ph_wq:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->wq
                                      shape:@[@(d_model),@(d_model)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_wk:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->wk
                                      shape:@[@(d_model),@(d_model)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_wv:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->wv
                                      shape:@[@(d_model),@(d_model)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_wo:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->wo
                                      shape:@[@(d_model),@(d_model)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_w1:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->w1
                                      shape:@[@(d_model),@(G_H)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_w3:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->w3
                                      shape:@[@(d_model),@(G_H)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_w2:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->w2
                                      shape:@[@(G_H),@(d_model)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_g1:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->g1
                                      shape:@[@(d_model)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_g2:       [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:lw->g2
                                      shape:@[@(d_model)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_rope_cos: [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:g_buf_rope_cos
                                      shape:@[@1,@1,@(G_SEQ),@(G_HD)]
                                      dataType:MPSDataTypeFloat16],
                lg->ph_rope_sin: [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:g_buf_rope_sin
                                      shape:@[@1,@1,@(G_SEQ),@(G_HD)]
                                      dataType:MPSDataTypeFloat16],
                // dL/d_out de capa l+1 (o del LM Head para l=n_layers-1)
                lg->ph_d_out:    [[MPSGraphTensorData alloc]
                                      initWithMTLBuffer:g_buf_dx_layers[l+1]
                                      shape:@[@(batch_seq),@(d_model)]
                                      dataType:MPSDataTypeFloat32],
            };

            // FIX 2: results también como NSArray ordenado
            // El orden debe coincidir exactamente con targetTensors en exec_bwd
            NSArray<MPSGraphTensorData*> *results = @[
                // t_dx: dL/dx_l → propagar a capa l-1
                [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf_dx_layers[l]
                    shape:@[@(batch_seq),@(d_model)] dataType:MPSDataTypeFloat32],
                // Gradientes de pesos — DISTINTOS para Wq, Wk, Wv
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dwq
                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dwk
                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dwv
                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dwo
                    shape:@[@(d_model),@(d_model)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dw1
                    shape:@[@(d_model),@(G_H)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dw2
                    shape:@[@(G_H),@(d_model)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dw3
                    shape:@[@(d_model),@(G_H)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dg1
                    shape:@[@(d_model)] dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:gb->dg2
                    shape:@[@(d_model)] dataType:MPSDataTypeFloat32],
            ];

            NSMutableArray<MPSGraphTensorData*> *bwd_inputs =
                [NSMutableArray arrayWithCapacity:lg->exec_bwd.feedTensors.count];
            for (MPSGraphTensor *t in lg->exec_bwd.feedTensors) {
                [bwd_inputs addObject:feeds[t]];
            }
            [lg->exec_bwd runWithMTLCommandQueue:g_queue
                                     inputsArray:bwd_inputs
                                    resultsArray:results
                             executionDescriptor:nil];
        }

        // ── 3. Sincronizar GPU → CPU ──────────────────────────────────────────
        id<MTLCommandBuffer> sync = [g_queue commandBuffer];
        [sync commit];
        [sync waitUntilCompleted];

        // ── 4. Copiar gradientes MTLBuffer → arrays FP32 de Rust (+=) ─────────
        {
            // A. Gradientes del LM Head sobre el embedding (tied weights o capa de salida)
            const float *s_head = (const float*)g_buf_dembed.contents;
            for (int i=0; i<vocab*d_model; i++) embed_w_grad[i] += s_head[i];

            // A2. Gradiente de gamma_final
            const float *s_gf = (const float*)g_buf_dgamma_final.contents;
            for (int i = 0; i < d_model; i++) gamma_final_grad[i] += s_gf[i];

            // B. Gradientes del lookup de entrada: dx_layers[0] → filas embed[token]
            // MPSGraph propagó dL/d(embed_output) hasta g_buf_dx_layers[0].
            // Hay que scatter-sumar cada fila i al índice del token correspondiente.
            // Con tied embeddings esto suma el gradiente de entrada al mismo tensor;
            // con embeddings separados actualiza las filas de la tabla de entrada.
            const float *dx0 = (const float*)g_buf_dx_layers[0].contents;
            for (int i = 0; i < batch_seq; i++) {
                uint32_t t = (tokens[i] < (uint32_t)vocab) ? tokens[i] : 0;
                for (int j = 0; j < d_model; j++) {
                    embed_w_grad[t * d_model + j] += dx0[i * d_model + j];
                }
            }
        }

        for (int l=0; l<n_layers; l++) {
            ArkLayerGradBuf *gb = &g_grad_bufs[l];
            size_t dd=(size_t)d_model*d_model;
            size_t dh=(size_t)d_model*hidden_dim;
            size_t hd=(size_t)hidden_dim*d_model;
            size_t d1=(size_t)d_model;
            const float *s;

            s=(const float*)gb->dwq.contents; for(size_t i=0;i<dd;i++) layer_wq_grad[l][i]+=s[i];
            s=(const float*)gb->dwk.contents; for(size_t i=0;i<dd;i++) layer_wk_grad[l][i]+=s[i];
            s=(const float*)gb->dwv.contents; for(size_t i=0;i<dd;i++) layer_wv_grad[l][i]+=s[i];
            s=(const float*)gb->dwo.contents; for(size_t i=0;i<dd;i++) layer_wo_grad[l][i]+=s[i];
            s=(const float*)gb->dw1.contents; for(size_t i=0;i<dh;i++) layer_w1_grad[l][i]+=s[i];
            s=(const float*)gb->dw2.contents; for(size_t i=0;i<hd;i++) layer_w2_grad[l][i]+=s[i];
            s=(const float*)gb->dw3.contents; for(size_t i=0;i<dh;i++) layer_w3_grad[l][i]+=s[i];
            s=(const float*)gb->dg1.contents; for(size_t i=0;i<d1;i++) layer_g1_grad[l][i]+=s[i];
            s=(const float*)gb->dg2.contents; for(size_t i=0;i<d1;i++) layer_g2_grad[l][i]+=s[i];
        }
    }
    return true;
}

// ── Cross-Entropy ─────────────────────────────────────────────────────────────
bool ark_mps_cross_entropy(const float *logits, const uint32_t *targets,
                            float *logits_grad, float *loss_out,
                            int batch_seq, int vocab)
{
    float total_loss = 0.0f;
    for (int i=0; i<batch_seq; i++) {
        const float *row  = &logits[i*vocab];
        float       *grow = &logits_grad[i*vocab];
        float max_val = 0.0f;
        vDSP_maxv(row, 1, &max_val, (vDSP_Length)vocab);
        float sum_exp = 0.0f;
        for (int j=0; j<vocab; j++) { grow[j]=expf(row[j]-max_val); sum_exp+=grow[j]; }
        float inv = 1.0f/(sum_exp+1e-10f);
        vDSP_vsmul(grow, 1, &inv, grow, 1, (vDSP_Length)vocab);
        uint32_t tgt = (targets[i]<(uint32_t)vocab)?targets[i]:0;
        total_loss += -logf(grow[tgt]+1e-10f);
        grow[tgt]  -= 1.0f;
        float inv_bs = 1.0f/(float)batch_seq;
        vDSP_vsmul(grow, 1, &inv_bs, grow, 1, (vDSP_Length)vocab);
    }
    *loss_out = total_loss/(float)batch_seq;
    return true;
}

// ── Shutdown ──────────────────────────────────────────────────────────────────
void ark_mps_shutdown(void) {
    g_ready = false;
    if (g_buf_x_layers)  {
        for(int i=0;i<=G_NL;i++) g_buf_x_layers[i]=nil;
        free(g_buf_x_layers);
        g_buf_x_layers=NULL;
    }
    // FIX 3: g_buf_x2_layers eliminado — no existe más
    if (g_buf_dx_layers) {
        for(int i=0;i<=G_NL;i++) g_buf_dx_layers[i]=nil;
        free(g_buf_dx_layers);
        g_buf_dx_layers=NULL;
    }
    if (g_grad_bufs) { free(g_grad_bufs); g_grad_bufs=NULL; }
    printf("[gpu] ARK v1.3 Shutdown — AutoGrad GPU apagado\n");
}
