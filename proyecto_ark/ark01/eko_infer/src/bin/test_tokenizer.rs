// src/bin/test_tokenizer.rs
// Prueba sistemática del tokenizador BPE.
// Verifica que palabras comunes en español se tokenicen como un solo token.

use std::collections::HashMap;

const METASPACE: char = '\u{2581}';

struct BpeTokenizer {
    vocab:    HashMap<String, u32>,
    id2piece: Vec<String>,
    merges:   Vec<(String, String)>,
    unk_id:   u32,
}

impl BpeTokenizer {
    fn load(vocab_path: &str, merges_path: &str) -> Self {
        let raw = std::fs::read_to_string(vocab_path).expect("vocab");
        let json: serde_json::Value = serde_json::from_str(&raw).expect("json");
        let vocab_obj = json.as_object().expect("objeto");
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (k, v) in vocab_obj {
            vocab.insert(k.clone(), v.as_u64().unwrap_or(0) as u32);
        }
        let mut id2piece: Vec<String> = vec![String::new(); vocab.len()];
        for (k, &id) in &vocab {
            if (id as usize) < id2piece.len() {
                id2piece[id as usize] = k.clone();
            }
        }
        let merges_raw = std::fs::read_to_string(merges_path).expect("merges");
        let mut merges: Vec<(String, String)> = Vec::new();
        for line in merges_raw.lines() {
            let mut parts = line.splitn(2, ' ');
            let a = parts.next().unwrap_or("").to_string();
            let b = parts.next().unwrap_or("").to_string();
            if !a.is_empty() && !b.is_empty() { merges.push((a, b)); }
        }
        let unk_id = vocab.get("<OOV>").or_else(|| vocab.get("<unk>")).copied().unwrap_or(0);
        eprintln!("[tok] vocab={} merges={} unk={}", vocab.len(), merges.len(), unk_id);
        Self { vocab, id2piece, merges, unk_id }
    }

    fn bpe_word(&self, word: &str) -> Vec<String> {
        // HashMap con ownership completo — sin referencias a self.merges
        let merge_rank: HashMap<(String, String), usize> = self.merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            if pieces.len() < 2 { break; }
            let mut best_rank = usize::MAX;
            let mut best_pos  = usize::MAX;
            for i in 0..pieces.len() - 1 {
                let key = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = merge_rank.get(&key) {
                    if rank < best_rank { best_rank = rank; best_pos = i; }
                }
            }
            if best_pos == usize::MAX { break; }
            let merged = format!("{}{}", pieces[best_pos], pieces[best_pos + 1]);
            pieces.remove(best_pos + 1);
            pieces[best_pos] = merged;
        }
        pieces
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        let prefixed = format!("{}{}", METASPACE, word);
        let pieces = self.bpe_word(&prefixed);
        pieces.iter().map(|p| self.vocab.get(p.as_str()).copied().unwrap_or(self.unk_id)).collect()
    }

    fn id_of(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Uso: test_tokenizer <vocab_sp.json> <merges.txt>");
        std::process::exit(1);
    }
    let tok = BpeTokenizer::load(&args[1], &args[2]);

    // Palabras de prueba: (palabra, token_esperado_con_metaspace)
    // Si el tokenizador funciona bien, cada palabra debería ser 1 token.
    let palabras = vec![
        "hola", "casa", "mundo", "el", "la", "de", "que", "en", "un",
        "una", "con", "por", "para", "como", "pero", "más", "año",
        "años", "también", "sobre", "entre", "hasta", "desde", "cuando",
        "donde", "porque", "aunque", "mientras", "durante", "después",
        "antes", "parte", "vida", "tiempo", "forma", "lugar", "país",
        "ciudad", "gobierno", "mundo", "sistema", "nervioso", "humano",
        "España", "México", "Argentina", "Chile", "agua", "fuego",
    ];

    let mut ok = 0;
    let mut fail = 0;
    let mut unk_count = 0;

    println!("{:<20} {:>8}  {:<30} {}", "PALABRA", "IDS", "TOKENS", "ESPERADO");
    println!("{}", "-".repeat(80));

    for palabra in &palabras {
        let ids = tok.encode_word(palabra);
        let expected_key = format!("{}{}", METASPACE, palabra);
        let expected_id  = tok.id_of(&expected_key);

        let tokens: Vec<String> = ids.iter().map(|&id| {
            if (id as usize) < tok.id2piece.len() {
                format!("{:?}", tok.id2piece[id as usize])
            } else {
                format!("?{}", id)
            }
        }).collect();

        let is_unk = ids.iter().any(|&i| i == tok.unk_id);
        let is_single = ids.len() == 1 && expected_id.map_or(false, |e| ids[0] == e);

        let status = if is_single {
            ok += 1; "✓"
        } else if is_unk {
            unk_count += 1; "UNK"
        } else {
            fail += 1; "✗"
        };

        let exp_str = expected_id.map_or("(no en vocab)".to_string(), |i| format!("id={}", i));
        println!("{:<20} {:>8}  {:<30} {}  {}",
            palabra,
            ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
            tokens.join(" "),
            exp_str,
            status
        );
    }

    println!("{}", "-".repeat(80));
    println!("✓ OK: {}  ✗ Fragmentados: {}  UNK: {}  Total: {}",
             ok, fail, unk_count, palabras.len());

    // Test extra: caracteres especiales del español
    println!("\n--- Caracteres base ---");
    let chars_es = vec!['á','é','í','ó','ú','ñ','ü','¿','¡','à','è'];
    for ch in chars_es {
        let s = ch.to_string();
        let id = tok.id_of(&s);
        println!("  {:?} -> {:?}", ch, id);
    }

    // Test: ver si ▁ existe como pieza sola
    let meta_id = tok.id_of(&METASPACE.to_string());
    println!("\n  METASPACE '▁' -> {:?}", meta_id);

    // Test: primeras 10 piezas de una letra para ver si los merges aplican
    println!("\n--- Debug BPE para 'hola' ---");
    let word = format!("{}hola", METASPACE);
    let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    println!("  chars: {:?}", chars);
    let merge_rank: HashMap<(String, String), usize> = tok.merges
        .iter().enumerate()
        .map(|(i,(a,b))| ((a.clone(),b.clone()),i))
        .collect();
    println!("  total merge_rank entries: {}", merge_rank.len());

    // Buscar merges relevantes para 'hola'
    let relevant = vec![
        (METASPACE.to_string(), "h".to_string()),
        ("▁h".to_string(), "o".to_string()),
        ("▁ho".to_string(), "l".to_string()),
        ("▁hol".to_string(), "a".to_string()),
        ("h".to_string(), "o".to_string()),
        ("ho".to_string(), "l".to_string()),
        ("hol".to_string(), "a".to_string()),
        ("o".to_string(), "l".to_string()),
        ("ol".to_string(), "a".to_string()),
    ];
    for (a, b) in &relevant {
        let rank = merge_rank.get(&(a.clone(), b.clone()));
        println!("  merge {:?}+{:?} -> rank={:?}", a, b, rank);
    }
}
