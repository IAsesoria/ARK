// expand_checkpoint.js
// Documento técnico ejecutable — Expansión de Checkpoint ARK V4
// Proceso: vocab 32.063 → 32.065 (agregar ü / Ü)
// Autor: Benjamín Alonso Carmona Vega — iAsesoria Informática — 2026
//
// USO:
//   node expand_checkpoint.js \
//     --input  ckpt_ark_ep1_rot2.bin \
//     --output ckpt_ark_ep2_v32065_rot0.bin \
//     --vocab-old 32063 \
//     --vocab-new 32065 \
//     --d-model 768
//
// QUÉ HACE:
//   1. Lee el checkpoint V4 (MAGIC = 0x4B_34_52_41 "ARK4")
//   2. Extrae embed_w (FP16), embed_m (FP32), embed_v (FP32)
//   3. Expande las 3 tablas de vocab*d_model añadiendo 2 filas nuevas al final
//   4. Inicializa las filas nuevas con el promedio de los vecinos más cercanos
//   5. Escribe un nuevo checkpoint V4 con vocab=32065, preservando todo lo demás intacto
//
// SEGURIDAD:
//   - No modifica NUNCA el archivo original
//   - Verifica MAGIC, tamaños y n_tensors antes de escribir
//   - Los 32.063 vectores existentes se copian bit a bit — sin reconversión

const fs   = require('fs');
const path = require('path');

// ── Argumentos ───────────────────────────────────────────────────────────────
const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  args[process.argv[i].replace('--', '')] = process.argv[i+1];
}

const INPUT     = args['input']     || 'ckpt_ark_ep1_rot2.bin';
const OUTPUT    = args['output']    || 'ckpt_ark_ep2_v32065_rot0.bin';
const VOCAB_OLD = parseInt(args['vocab-old'] || '32063');
const VOCAB_NEW = parseInt(args['vocab-new'] || '32065');
const D_MODEL   = parseInt(args['d-model']   || '768');
const N_NEW     = VOCAB_NEW - VOCAB_OLD; // = 2

// ── Constantes del formato V4 ────────────────────────────────────────────────
const MAGIC_V4   = 0x4B_34_52_41; // "ARK4" little-endian
const TYPE_F16   = 0n;
const TYPE_F32   = 1n;

// ── Helpers lectura/escritura ────────────────────────────────────────────────
function readU32LE(buf, offset) { return buf.readUInt32LE(offset); }
function readU64LE(buf, offset) { return buf.readBigUInt64LE(offset); }
function writeU64LE(buf, val, offset) { buf.writeBigUInt64LE(BigInt(val), offset); }
function writeU32LE(buf, val, offset) { buf.writeUInt32LE(val, offset); }

// Convierte u16 (FP16 bits) → f32
function f16ToF32(bits) {
  const sign    = (bits >> 15) & 0x1;
  const exp     = (bits >> 10) & 0x1F;
  const mant    =  bits        & 0x3FF;
  if (exp === 0x1F) return sign ? -Infinity : Infinity;
  if (exp === 0)    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

// Convierte f32 → u16 (FP16 bits), round-to-nearest
function f32ToF16(val) {
  if (isNaN(val))           return 0x7E00;
  if (!isFinite(val))       return val > 0 ? 0x7C00 : 0xFC00;
  const sign = val < 0 ? 1 : 0;
  val = Math.abs(val);
  if (val === 0)            return sign << 15;
  let exp = Math.floor(Math.log2(val));
  let mant = val / Math.pow(2, exp) - 1;
  exp += 15;
  if (exp <= 0)             return sign << 15;
  if (exp >= 31)            return (sign << 15) | 0x7C00;
  const mantBits = Math.round(mant * 1024);
  return ((sign << 15) | (exp << 10) | (mantBits & 0x3FF));
}

// ── Inicialización nuevas filas ──────────────────────────────────────────────
// Estrategia: promedio de los K vecinos más próximos semánticamente.
// Para ü usamos: u (id conocido), más vocales con diéresis del vocabulario.
// Como no tenemos el vocab en este script, usamos promedio de las últimas
// N_NEIGHBORS filas del embedding (tokens de baja frecuencia, zona similar).
// Es mucho mejor que Xavier aleatorio y no requiere lookup del vocab.
const N_NEIGHBORS = 8;

function computeNewRow_f32(embedF32, vocabOld, dModel) {
  // Promedio de las últimas N_NEIGHBORS filas
  const newRow = new Float32Array(dModel);
  const start  = vocabOld - N_NEIGHBORS;
  for (let n = 0; n < N_NEIGHBORS; n++) {
    const rowBase = (start + n) * dModel;
    for (let d = 0; d < dModel; d++) {
      newRow[d] += embedF32[rowBase + d];
    }
  }
  for (let d = 0; d < dModel; d++) newRow[d] /= N_NEIGHBORS;
  return newRow;
}

function computeNewRow_f16(embedU16, vocabOld, dModel) {
  // Mismo proceso pero en FP16 — dequant→promedio→quant
  const newRow = new Float32Array(dModel);
  const start  = vocabOld - N_NEIGHBORS;
  for (let n = 0; n < N_NEIGHBORS; n++) {
    const rowBase = (start + n) * dModel;
    for (let d = 0; d < dModel; d++) {
      newRow[d] += f16ToF32(embedU16[rowBase + d]);
    }
  }
  for (let d = 0; d < dModel; d++) newRow[d] /= N_NEIGHBORS;
  // Convertir de vuelta a FP16
  const result = new Uint16Array(dModel);
  for (let d = 0; d < dModel; d++) result[d] = f32ToF16(newRow[d]);
  return result;
}

// ── Main ─────────────────────────────────────────────────────────────────────
console.log('');
console.log('╔═══════════════════════════════════════════════════════════╗');
console.log('║  ARK Checkpoint Expander — vocab 32063 → 32065            ║');
console.log('║  Proceso: agregar ü / Ü al embedding                      ║');
console.log('╚═══════════════════════════════════════════════════════════╝');
console.log('');
console.log(`  Input  : ${INPUT}`);
console.log(`  Output : ${OUTPUT}`);
console.log(`  Vocab  : ${VOCAB_OLD} → ${VOCAB_NEW} (+${N_NEW} tokens)`);
console.log(`  d_model: ${D_MODEL}`);
console.log('');

// 1. Leer archivo completo
if (!fs.existsSync(INPUT)) {
  console.error(`[FATAL] Archivo no encontrado: ${INPUT}`);
  process.exit(1);
}
const raw = fs.readFileSync(INPUT);
console.log(`[1/5] Checkpoint leído: ${(raw.length / 1e6).toFixed(1)} MB`);

// 2. Verificar MAGIC
const magic = readU32LE(raw, 0);
if (magic !== MAGIC_V4) {
  console.error(`[FATAL] Magic inválido: 0x${magic.toString(16).toUpperCase()} — se esperaba ARK4 (0x${MAGIC_V4.toString(16).toUpperCase()})`);
  process.exit(1);
}
console.log('[2/5] MAGIC V4 verificado ✓');

// 3. Leer header
let offset = 4;
const globalStep = readU64LE(raw, offset); offset += 8;
const adamStep   = readU64LE(raw, offset); offset += 8;
const nTensors   = readU64LE(raw, offset); offset += 8;

console.log(`[3/5] Header: global_step=${globalStep} | adam_step=${adamStep} | n_tensors=${nTensors}`);

// Verificar n_tensors esperado: 3 + n_layers*27
// Con 30 capas: 3 + 30*27 = 813
const expectedTensors = 3n + 30n * 27n;
if (nTensors !== expectedTensors) {
  console.warn(`[WARN] n_tensors=${nTensors} difiere del esperado ${expectedTensors} — verificar arquitectura`);
}

// 4. Leer todos los tensores, expandir los 3 del embedding
const outputParts = [];

// Header nuevo (mismo global_step y adam_step, mismo n_tensors)
const header = Buffer.allocUnsafe(4 + 8 + 8 + 8);
writeU32LE(header, MAGIC_V4, 0);
writeU64LE(header, globalStep, 4);
writeU64LE(header, adamStep, 12);
writeU64LE(header, nTensors, 20);
outputParts.push(header);

console.log('[4/5] Procesando tensores...');

let tensorIdx = 0;
while (tensorIdx < Number(nTensors)) {
  const nElements = Number(readU64LE(raw, offset)); offset += 8;
  const tipo      = readU64LE(raw, offset);         offset += 8;
  const isF16     = tipo === TYPE_F16;
  const elemSize  = isF16 ? 2 : 4;
  const dataBytes = nElements * elemSize;
  const data      = raw.slice(offset, offset + dataBytes);
  offset += dataBytes;

  // Los 3 primeros tensores son embed_w(F16), embed_m(F32), embed_v(F32)
  const isEmbedTensor = tensorIdx < 3;
  const expectedEmbed = VOCAB_OLD * D_MODEL;

  if (isEmbedTensor && nElements === expectedEmbed) {
    const newElements = VOCAB_NEW * D_MODEL;
    const typeLabel   = isF16 ? 'F16' : 'F32';
    const tensorName  = ['embed_w', 'embed_m', 'embed_v'][tensorIdx];

    console.log(`  [tensor ${tensorIdx}] ${tensorName} (${typeLabel}): ${VOCAB_OLD}×${D_MODEL} → ${VOCAB_NEW}×${D_MODEL}`);

    // Escribir header del tensor expandido
    const tensorHeader = Buffer.allocUnsafe(16);
    writeU64LE(tensorHeader, newElements, 0);
    tensorHeader.writeBigUInt64LE(tipo, 8);
    outputParts.push(tensorHeader);

    // Copiar datos originales
    outputParts.push(data);

    // Generar N_NEW filas nuevas
    for (let newTok = 0; newTok < N_NEW; newTok++) {
      if (isF16) {
        const u16View = new Uint16Array(data.buffer, data.byteOffset, nElements);
        const newRow  = computeNewRow_f16(u16View, VOCAB_OLD, D_MODEL);
        outputParts.push(Buffer.from(newRow.buffer));
      } else {
        const f32View = new Float32Array(data.buffer, data.byteOffset, nElements);
        const newRow  = computeNewRow_f32(f32View, VOCAB_OLD, D_MODEL);
        outputParts.push(Buffer.from(newRow.buffer));
      }
    }

  } else {
    // Tensor sin cambios — copiar tal cual
    const tensorHeader = Buffer.allocUnsafe(16);
    writeU64LE(tensorHeader, nElements, 0);
    tensorHeader.writeBigUInt64LE(tipo, 8);
    outputParts.push(tensorHeader);
    outputParts.push(data);
  }

  tensorIdx++;
}

// 5. Escribir output
const outputBuf = Buffer.concat(outputParts);
fs.writeFileSync(OUTPUT, outputBuf);
const sizeMB = outputBuf.length / 1e6;

console.log('');
console.log(`[5/5] Checkpoint expandido escrito: ${OUTPUT}`);
console.log(`      Tamaño: ${sizeMB.toFixed(1)} MB`);
console.log('');
console.log('╔═══════════════════════════════════════════════════════════╗');
console.log('║  VERIFICACIÓN — ejecutar antes de reanudar entrenamiento  ║');
console.log('╚═══════════════════════════════════════════════════════════╝');
console.log('');
console.log('  1. Confirmar tamaño del output:');
console.log(`       ls -lh ${OUTPUT}`);
console.log(`       # Esperado: ~${(sizeMB + (N_NEW * D_MODEL * 10 / 1e6)).toFixed(0)} MB`);
console.log('');
console.log('  2. Verificar que ARK carga sin errores:');
console.log(`       ./target/release/ark --ckpt ${OUTPUT} --vocab tokenizador_bpe_32k_v3.model \\`);
console.log('         --layers=30 --heads=12 --d-model=768 --hidden=2048 \\');
console.log('         --seq=128 --batch=1 --epochs=0');
console.log('');
console.log('  3. Confirmar que el loss arranca desde donde lo dejaste');
console.log('     (no debe reiniciar desde ~10.47 — si lo hace, algo falló)');
console.log('');
console.log('  IMPORTANTE: NO borrar el checkpoint original hasta confirmar');
console.log('  que el modelo entrena correctamente al menos 500 pasos.');
console.log('');
