// build.rs — ARK v1.0
// Compila Objective-C (bridge Metal/MPSGraph) y Ensamblador ARM64 (kernels AMX).

use std::process::Command;
use std::path::PathBuf;

fn main() {
    // ── Recompilar solo si estos archivos cambian ────────────────────────────
    println!("cargo:rerun-if-changed=objc/bridge.m");
    println!("cargo:rerun-if-changed=asm/kern.s");
    println!("cargo:rerun-if-changed=asm/opti.s");

    // ── Directorio de salida de Cargo ────────────────────────────────────────
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // ── 1. Compilar Objective-C (bridge Metal/MPSGraph) ─────────────────────
    // -DACCELERATE_NEW_LAPACK elimina deprecaciones de cblas_sgemm en macOS 13.3+
    cc::Build::new()
        .compiler("clang")
        .file("objc/bridge.m")
        .flag("-fobjc-arc")
        .flag("-O3")
        .flag("-mmacosx-version-min=15.0")
        .flag("-Wno-unused-parameter")
        .flag("-DACCELERATE_NEW_LAPACK")
        .compile("bridge");

    // ── 2. Ensamblar kern.s directamente con clang (no con `as`) ─────────────
    // `as` de Apple no acepta flags de cc-rs (-O0, -ffunction-sections, etc.)
    // Clang como driver de ensamblado los ignora limpiamente.
    // -arch arm64 es explícito para evitar ambigüedad en macs con Rosetta.
    let kern_obj = PathBuf::from(&out_dir).join("kern.o");
    let status_kern = Command::new("clang")
        .args(&[
            "-arch", "arm64",
            "-mmacosx-version-min=15.0",
            "-x", "assembler",
            "-c", "asm/kern.s",
            "-o", kern_obj.to_str().unwrap(),
        ])
        .status()
        .expect("fallo al invocar clang para kern.s");
    assert!(status_kern.success(), "Error ensamblando kern.s");

    // ── 3. Ensamblar opti.s ──────────────────────────────────────────────────
    let opti_obj = PathBuf::from(&out_dir).join("opti.o");
    let status_opti = Command::new("clang")
        .args(&[
            "-arch", "arm64",
            "-mmacosx-version-min=15.0",
            "-x", "assembler",
            "-c", "asm/opti.s",
            "-o", opti_obj.to_str().unwrap(),
        ])
        .status()
        .expect("fallo al invocar clang para opti.s");
    assert!(status_opti.success(), "Error ensamblando opti.s");

    // ── 4. Empaquetar objetos ASM en libark_asm.a ────────────────────────────
    let lib_path = PathBuf::from(&out_dir).join("libark_asm.a");
    let status_ar = Command::new("ar")
        .args(&[
            "rcs",
            lib_path.to_str().unwrap(),
            kern_obj.to_str().unwrap(),
            opti_obj.to_str().unwrap(),
        ])
        .status()
        .expect("fallo al invocar ar");
    assert!(status_ar.success(), "Error creando libark_asm.a");

    // Decirle a Cargo dónde buscar y qué enlazar
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=ark_asm");

    // ── 5. Enlazar frameworks nativos de macOS ───────────────────────────────
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    println!("cargo:rustc-link-lib=framework=Accelerate");
}