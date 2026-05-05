// src/main.rs — ARK v1.0 "METAL-REASONER"
//
// Entry point del motor ARK.
// Modos:
//   - Entrenamiento: ark [opciones]
//   - Build corpus:  ark --build-corpus=F1,F2 --vocab=V --corpus-out=salida.bin
//
// Mejoras v1.0:
//   - Soporte nativo para argumentos intercambiables (guión y guión bajo).
//   - Banners y ayuda actualizados para reflejar Zero-Copy, RoPE y SDPA.
//   - Ejemplos orientados a modelos razonadores (Chinchilla-optimum) en 8GB RAM.

mod config;
mod ffi;
mod io;
mod memory;
mod optimizer;
mod training;

use config::Config;
use training::Trainer;
use io::CorpusBuilder;

fn print_banner() {
    println!("╔═══");
    println!("║  ARK v1.0 — ZERO-COPY METAL-REASONER ENGINE");
    println!("║  NOUS / IAsesoria Informática");
    println!("║  Villarrica, Chile");
    println!("╚═══");
    println!("  Modo: GPU (MPSGraph SDPA+RoPE) + CPU (NEON/AMX)");
    println!("  Memoria: Zero-Copy 100% nativo (FP16)");
    println!("  Tokenizador: BPE 32k (SentencePiece)");
    println!();
}

fn print_help() {
    println!("USO: ark [opciones]");
    println!();
    println!("RUTAS:");
    println!("  --corpus=R1,R2,...               Corpus JSONL/.txt (uno o varios)");
    println!("  --vocab=RUTA                     Modelo BPE (.model SentencePiece)");
    println!("  --ckpt=RUTA                      Checkpoint de salida/entrada");
    println!();
    println!("ARQUITECTURA (Optimizada para razonamiento en 8GB):");
    println!("  --layers=N    / --n-layers=N     Número de capas          [default: 32]");
    println!("  --heads=N     / --n-heads=N      Número de cabezas        [default: 8]");
    println!("  --d-model=N   / --d_model=N      Dimensión embedding      [default: 512]");
    println!("  --hidden-dim=N / --hidden=N      Dimensión FFN (SwiGLU)   [default: 2048]");
    println!("  --seq=N                          Long. secuencia (RoPE)   [default: 2048]");
    println!("  --vocab-size=N                   Tamaño vocabulario       [default: 32000]");
    println!();
    println!("ENTRENAMIENTO:");
    println!("  --epochs=N                       Épocas totales           [default: 3]");
    println!("  --batch=N                        Tamaño del batch[default: 2]");
    println!("  --lr=F                           Learning rate inicial[default: 3e-4]");
    println!("  --clip=F                         Gradient clipping        [default: 1.0]");
    println!("  --warmup=N                       Pasos de warmup del LR[default: 100]");
    println!();
    println!("ADAMW (Zero-Copy):");
    println!("  --beta1=F                        Beta1                    [default: 0.9]");
    println!("  --beta2=F                        Beta2                    [default: 0.999]");
    println!("  --eps=F                          Épsilon                  [default: 1e-8]");
    println!("  --wd=F                           Weight decay[default: 0.01]");
    println!();
    println!("CONSTRUCCIÓN DE CORPUS (.bin opcional):");
    println!("  ark --build-corpus=F1,F2 --vocab=V --corpus-out=salida.bin");
    println!();
    println!("EJEMPLO ENTRENAMIENTO RAZONADOR M1 8GB (Límite Físico):");
    println!("  ark --corpus=datos.jsonl --vocab=tokenizer.model \\");
    println!("      --ckpt=modelo.bin --layers=32 --d-model=512 \\");
    println!("      --hidden=2048 --heads=8 --seq=2048 --batch=2 \\");
    println!("      --epochs=3 --lr=3e-4 --warmup=5000 --clip=1.0");
    println!();
}

fn normalizar_key(key: &str) -> String {
    // Convierte guiones a guión bajo y pasa a minúscula para comparación uniforme.
    // Permite usar indiferentemente --d-model o --d_model
    key.replace('-', "_").to_lowercase()
}

fn parse_args(cfg: &mut Config) {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Modo ayuda
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    // Modo posicional (legacy): ark corpus vocab ckpt epochs lr
    let positional_mode = !args.is_empty() && !args[0].starts_with("--");

    if positional_mode {
        if !args.is_empty() {
            cfg.corpus_paths = args[0]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if args.len() > 1 {
            cfg.vocab_path = args[1].clone();
        }
        if args.len() > 2 {
            cfg.ckpt_path = args[2].clone();
        }
        if args.len() > 3 {
            cfg.n_epochs = args[3].parse().expect("[args] epochs debe ser entero");
        }
        if args.len() > 4 {
            cfg.lr = args[4].parse().expect("[args] lr debe ser float");
        }
        return;
    }

    // Modo flags (--clave=valor)
    for arg in &args {
        if !arg.starts_with("--") {
            continue;
        }
        let arg = &arg[2..];

        // Manejo de flags booleanos sin valor explícito (--fp16, --fp32, --help)
        let (key_raw, val) = match arg.split_once('=') {
            Some(kv) => kv,
            None => {
                let key = normalizar_key(arg);
                match key.as_str() {
                    "fp16" | "use_fp16" => cfg.use_fp16 = true,
                    "fp32" | "use_fp32" => cfg.use_fp16 = false,
                    "help" | "h" => {
                        print_help();
                        std::process::exit(0);
                    }
                    _ => eprintln!("[args] flag sin valor ignorado: --{}", arg),
                }
                continue;
            }
        };

        let key = normalizar_key(key_raw);

        // Closures seguras para parsing
        let parse_int = |s: &str, name: &str| -> usize {
            s.parse().unwrap_or_else(|_| {
                eprintln!("[args] {} debe ser entero, recibido: {}", name, s);
                std::process::exit(1);
            })
        };
        let parse_f32 = |s: &str, name: &str| -> f32 {
            s.parse().unwrap_or_else(|_| {
                eprintln!("[args] {} debe ser float, recibido: {}", name, s);
                std::process::exit(1);
            })
        };

        match key.as_str() {
            // ── Rutas ──────
            "corpus" => {
                cfg.corpus_paths = val
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "vocab" | "vocab_path" => cfg.vocab_path = val.into(),
            "ckpt"  | "checkpoint" => cfg.ckpt_path  = val.into(),

            // ── Arquitectura ──────
            "d_model"    | "dmodel"  => cfg.d_model    = parse_int(val, "d-model"),
            "n_heads"    | "heads"   => cfg.n_heads    = parse_int(val, "heads"),
            "n_layers"   | "layers"  => cfg.n_layers   = parse_int(val, "layers"),
            "hidden_dim" | "hidden"  => cfg.hidden_dim = parse_int(val, "hidden-dim"),
            "seq"        | "seq_len" => cfg.seq_len    = parse_int(val, "seq"),
            "vocab_size"             => cfg.vocab_size = parse_int(val, "vocab-size"),

            // ── Entrenamiento ─────
            "epochs" | "n_epochs"      => cfg.n_epochs     = parse_int(val, "epochs"),
            "batch"  | "batch_size"    => cfg.batch_size   = parse_int(val, "batch"),
            "lr"     | "learning_rate" => cfg.lr           = parse_f32(val, "lr"),
            "loss-scale-max" | "scale_max" => cfg.loss_scale_max = parse_f32(val, "loss-scale-max"),
            "clip"   | "grad_clip"     => cfg.grad_clip    = parse_f32(val, "clip"),
            "warmup" | "warmup_steps"  => cfg.warmup_steps = parse_int(val, "warmup"),

            // ── Precisión ─────
            "fp16" | "use_fp16" => cfg.use_fp16 = true,
            "fp32" | "use_fp32" => cfg.use_fp16 = false,

            // ── Adam ─────
            "beta1"                 => cfg.beta1        = parse_f32(val, "beta1"),
            "beta2"                 => cfg.beta2        = parse_f32(val, "beta2"),
            "eps" | "adam_eps"      => cfg.adam_eps     = parse_f32(val, "eps"),
            "wd"  | "weight_decay"  => cfg.weight_decay = parse_f32(val, "wd"),

            // ── Build corpus ──────
            "build_corpus" | "corpus_out" => { /* Ignorados durante train mode */ }

            _ => eprintln!("[args] opción desconocida ignorada: --{}", key_raw),
        }
    }
}

fn main() -> anyhow::Result<()> {
    print_banner();

    let args_raw: Vec<String> = std::env::args().skip(1).collect();

    // ── Modo ayuda inmediato ───────
    if args_raw.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }

    // ── Modo --build-corpus ─────
    let build_flag = args_raw
        .iter()
        .find(|a| a.starts_with("--build-corpus="));

    if let Some(flag) = build_flag {
        let fuentes_str = flag.trim_start_matches("--build-corpus=");
        let fuentes: Vec<&str> = fuentes_str
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        anyhow::ensure!(
            !fuentes.is_empty(),
            "[build-corpus] no se proporcionaron archivos fuente"
        );

        let vocab = args_raw
            .iter()
            .find(|a| a.starts_with("--vocab="))
            .map(|a| a.trim_start_matches("--vocab="))
            .unwrap_or("entren/tokenizador_bpe_32k.model");

        let salida = args_raw
            .iter()
            .find(|a| a.starts_with("--corpus-out="))
            .map(|a| a.trim_start_matches("--corpus-out="))
            .unwrap_or("entren/corpus.bin");

        CorpusBuilder::build(&fuentes, vocab, salida)?;
        return Ok(());
    }

    // ── Modo Entrenamiento (Zero-Copy) ────
    let mut cfg = Config::default_ark();
    parse_args(&mut cfg);

    // Recalcular campos derivados (head_dim = d_model / n_heads)
    cfg.fix_derived();

    // Validar configuración para prevenir fallos en Metal
    cfg.validate()?;

    // Mostrar configuración en consola
    cfg.print();

    // Iniciar entrenamiento con ARK v1.0
    let mut trainer = Trainer::new(cfg)?;
    trainer.run()?;

    Ok(())
}
