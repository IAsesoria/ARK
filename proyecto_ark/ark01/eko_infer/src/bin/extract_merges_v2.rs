// src/bin/extract_merges.rs
// Exporta vocab_sp.json (pieza->id) y vocab_scores.json (pieza->score)
// desde tokenizador_bpe_32k.model (SentencePiece protobuf).
// Los scores se usan para tokenización Viterbi (sin necesidad de merges).

use std::fs;
use std::io::Write;

fn read_varint(data: &[u8], pos: &mut usize) -> u64 {
    let mut result = 0u64;
    let mut shift = 0;
    loop {
        let b = data[*pos]; *pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 { break; }
        shift += 7;
    }
    result
}

fn skip_field(data: &[u8], pos: &mut usize, wire: u64) {
    match wire {
        0 => { read_varint(data, pos); }
        1 => { *pos += 8; }
        2 => { let l = read_varint(data, pos) as usize; *pos += l; }
        5 => { *pos += 4; }
        _ => { eprintln!("[WARN] wire desconocido: {}", wire); }
    }
}

fn parse_piece(data: &[u8], start: usize, end: usize) -> (String, f32, u32) {
    let mut pos = start;
    let mut piece = String::new();
    let mut score: f32 = 0.0;
    let mut ptype: u32 = 1;
    while pos < end {
        let tag   = read_varint(data, &mut pos);
        let field = tag >> 3;
        let wire  = tag & 0x7;
        match field {
            1 => {
                let l = read_varint(data, &mut pos) as usize;
                piece = String::from_utf8_lossy(&data[pos..pos+l]).to_string();
                pos += l;
            }
            2 => {
                let bytes = [data[pos], data[pos+1], data[pos+2], data[pos+3]];
                score = f32::from_le_bytes(bytes);
                pos += 4;
            }
            3 => { ptype = read_varint(data, &mut pos) as u32; }
            _ => { skip_field(data, &mut pos, wire); }
        }
    }
    (piece, score, ptype)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: extract_merges <tokenizador.model> [directorio_salida]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let out_dir    = if args.len() > 2 { args[2].clone() } else { ".".to_string() };

    let data = fs::read(model_path).expect("No se puede leer el .model");
    eprintln!("[sp] {} bytes", data.len());

    let mut pieces: Vec<(String, f32, u32)> = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        let tag   = read_varint(&data, &mut pos);
        let field = tag >> 3;
        let wire  = tag & 0x7;
        if wire == 2 {
            let msg_len = read_varint(&data, &mut pos) as usize;
            let end = pos + msg_len;
            if field == 1 {
                pieces.push(parse_piece(&data, pos, end));
            }
            pos = end;
        } else {
            skip_field(&data, &mut pos, wire);
        }
    }

    eprintln!("[sp] Piezas: {}", pieces.len());
    eprintln!("[sp] Primeras 8:");
    for (i, (p, s, t)) in pieces.iter().take(8).enumerate() {
        eprintln!("  [{}] {:?}  score={:.6}  type={}", i, p, s, t);
    }

    // ── vocab_sp.json: pieza -> id ────────────────────────────────────────────
    let vocab_path = format!("{}/vocab_sp.json", out_dir);
    {
        let mut f = fs::File::create(&vocab_path).expect("No se puede crear vocab_sp.json");
        write!(f, "{{").unwrap();
        for (i, (p, _, _)) in pieces.iter().enumerate() {
            let e = p.replace('\\', "\\\\").replace('"', "\\\"");
            if i > 0 { write!(f, ",").unwrap(); }
            write!(f, "\n  \"{}\": {}", e, i).unwrap();
        }
        writeln!(f, "\n}}").unwrap();
    }
    eprintln!("[sp] vocab_sp.json -> {}", vocab_path);

    // ── vocab_scores.json: pieza -> score (para Viterbi) ─────────────────────
    // Solo piezas NORMAL (type=1) y tokens especiales con score no nulo
    // Score en SentencePiece es log-probabilidad (negativo, más cercano a 0 = más frecuente)
    let scores_path = format!("{}/vocab_scores.json", out_dir);
    {
        let mut f = fs::File::create(&scores_path).expect("No se puede crear vocab_scores.json");
        write!(f, "{{").unwrap();
        let mut first = true;
        for (p, s, t) in &pieces {
            // Incluir todas las piezas tipo NORMAL
            if *t != 1 { continue; }
            let e = p.replace('\\', "\\\\").replace('"', "\\\"");
            if !first { write!(f, ",").unwrap(); }
            first = false;
            // Guardamos el score como float con suficiente precisión
            write!(f, "\n  \"{}\": {:.8}", e, s).unwrap();
        }
        writeln!(f, "\n}}").unwrap();
    }
    eprintln!("[sp] vocab_scores.json -> {}", scores_path);

    // ── Estadísticas de scores ────────────────────────────────────────────────
    let normal: Vec<f32> = pieces.iter()
        .filter(|(_,_,t)| *t == 1)
        .map(|(_, s, _)| *s)
        .collect();
    if !normal.is_empty() {
        let min = normal.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = normal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let avg = normal.iter().sum::<f32>() / normal.len() as f32;
        eprintln!("[sp] scores NORMAL: min={:.4} max={:.4} avg={:.4} count={}",
                  min, max, avg, normal.len());
    }

    // Verificación: mostrar scores de palabras comunes
    let check = vec!["▁hola", "▁casa", "▁el", "▁la", "▁de", "▁sistema", "▁nervioso"];
    eprintln!("[sp] Verificación de scores:");
    let vocab_map: std::collections::HashMap<&str, (usize, f32)> = pieces.iter()
        .enumerate()
        .map(|(i, (p, s, _))| (p.as_str(), (i, *s)))
        .collect();
    for w in &check {
        match vocab_map.get(w) {
            Some((id, s)) => eprintln!("  {:?} -> id={} score={:.6}", w, id, s),
            None          => eprintln!("  {:?} -> NO ENCONTRADO", w),
        }
    }
}
