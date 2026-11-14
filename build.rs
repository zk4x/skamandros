//! Builds the device firmware and checks its instruction encoding.
//!
//! Runs as part of any build that depends on this crate, so a fresh clone needs
//! no separate step. Moved here from `tt-tests/build.rs` so that the role images a
//! backend runs are owned by a shippable crate rather than by the test harness.
//!
//! The firmware is a separate workspace targeting `riscv32im-unknown-none-elf`.
//! It is built into `OUT_DIR` with its own `--target-dir`, and the host's
//! `CARGO_*` environment is stripped from the child so the nested invocation does
//! not inherit the parent's target directory or profile.

#[cfg(feature = "tenstorrent")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "tenstorrent")]
use std::process::Command;

/// The C spelling of every entry point: `(name, return type, parameters)`.
///
/// `write_c_header` renders the header from `src/abi.rs` in source order and
/// looks each name up here, so a new `extern "C" fn tt*` that this table does
/// not spell fails the build instead of quietly missing from the header.
const DECLARATIONS: &[(&str, &str, &str)] = &[
    ("ttAbiVersion", "uint32_t", "void"),
    (
        "ttCompile",
        "ttStatus",
        "const char *text, size_t len, ttModule **out, char **err",
    ),
    ("ttModuleUnload", "ttStatus", "ttModule *module, char **err"),
    ("ttInit", "ttStatus", "char **err"),
    ("ttRelease", "ttStatus", "char **err"),
    (
        "ttMemAlloc",
        "ttStatus",
        "size_t bytes, uint64_t *addr, char **err",
    ),
    (
        "ttMemcpyH2D",
        "ttStatus",
        "uint64_t dst, const uint8_t *src, size_t len, char **err",
    ),
    (
        "ttMemcpyD2H",
        "ttStatus",
        "uint8_t *dst, uint64_t src, size_t len, char **err",
    ),
    (
        "ttLaunch",
        "ttStatus",
        "ttModule *module, const uint64_t *params, size_t nparams, char **err",
    ),
    ("ttFree", "void", "void *ptr"),
];

/// Binaries in `firmware/src/bin` to build and expose.
#[cfg(feature = "tenstorrent")]
const BINARIES: &[&str] = &[
    "heartbeat",
    "sfpu_mul",
    "prng_seed",
    "corpus",
    "corpus_t0",
    "role_t0",
    "role_t1",
    "role_t2",
    "exec_t0",
    "exec_t1",
    "exec_t2",
    "eth_e1",
    "dm_b",
    "nc_probe",
    "dm_nc",
    "icache_b",
    "icache_nc",
    "icache_t0",
    "icache_t1",
    "icache_t2",
    "tag_search_b",
    "noc_probe_b",
    "read_loop_b",
    "exec_b",
    "exec_nc",
];

#[cfg(feature = "tenstorrent")]
const TARGET: &str = "riscv32im-unknown-none-elf";

fn main() {
    write_c_header();
    // The firmware belongs to the Tenstorrent backend, so a build without that
    // feature compiles none of it (and does not need the RISC-V target).
    #[cfg(feature = "tenstorrent")]
    {
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let firmware_dir = manifest.join("firmware");
        println!("cargo:rerun-if-changed=build.rs");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        // Track individual files rather than directories. A directory-level
        // `rerun-if-changed` does not reliably notice an edit to a file nested inside
        // it, which showed up here as firmware changes that never rebuilt.
        track_sources(&firmware_dir);

        // The firmware sources ship inside this crate but are not a package of
        // their own in the tree (`Cargo.toml.in`, not `Cargo.toml`): cargo refuses
        // to package a nested package. Copy them into `OUT_DIR` and give them the
        // manifest there, so the build works both from the tree and from the
        // published crate.
        let build_dir = out_dir.join("firmware-src");
        copy_tree(&firmware_dir, &build_dir);
        std::fs::copy(
            firmware_dir.join("Cargo.toml.in"),
            build_dir.join("Cargo.toml"),
        )
        .expect("could not write the firmware manifest");
        println!(
            "cargo:rerun-if-changed={}",
            firmware_dir.join("Cargo.toml.in").display()
        );

        let target_dir = out_dir.join("firmware-target");
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());

        let mut cmd = Command::new(&cargo);
        cmd.current_dir(&build_dir)
            .args(["build", "--release", "--target", TARGET])
            .arg("--target-dir")
            .arg(&target_dir);
        let mut features = Vec::new();
        // The mover's read mode is chosen here, at build time (`tt-firmware`'s features).
        if std::env::var_os("CARGO_FEATURE_FAST_READS").is_some() {
            features.push("fast-reads");
        }
        if std::env::var_os("CARGO_FEATURE_ENCODED_GUARD").is_some() {
            features.push("encoded-guard");
        }
        if !features.is_empty() {
            cmd.args(["--features", &features.join(",")]);
        }
        // A nested cargo inherits variables that redirect it back at the parent
        // workspace; strip them so the firmware build is genuinely independent.
        // `RUSTC_WORKSPACE_WRAPPER` is `cargo clippy`'s `clippy-driver`: left in, the
        // images of a clippy run came out about 900 bytes larger than a build's
        // (`dm_b` 24,540 bytes against 23,644, of the 24,576 it may have), so a
        // mover change that fit a build overflowed clippy. Firmware is linted on
        // its own (`AGENTS.md`).
        for (key, _) in std::env::vars() {
            if key.starts_with("CARGO_")
                || key == "RUSTFLAGS"
                || key == "RUSTC_WRAPPER"
                || key == "RUSTC_WORKSPACE_WRAPPER"
            {
                cmd.env_remove(key);
            }
        }
        let status = cmd
            .status()
            .expect("could not run cargo for the firmware build");
        assert!(status.success(), "firmware build failed");

        let objcopy = tool("llvm-objcopy");
        let objdump = tool("llvm-objdump");

        let binaries = BINARIES.iter().copied();
        for name in binaries {
            let elf = target_dir.join(TARGET).join("release").join(name);
            assert!(
                elf.exists(),
                "firmware binary {name} was not produced at {}",
                elf.display()
            );

            let bin = out_dir.join(format!("{name}.bin"));
            let status = Command::new(&objcopy)
                .args(["-O", "binary"])
                .arg(&elf)
                .arg(&bin)
                .status()
                .expect("could not run llvm-objcopy");
            assert!(status.success(), "objcopy failed for {name}");

            check_instruction_set(&objdump, &elf, name);

            // The entry point, so the crate can check each image against the core
            // it is loaded for instead of trusting a literal next to it.
            println!(
                "cargo:rustc-env=FIRMWARE_{}_ENTRY={}",
                name.to_uppercase(),
                elf_entry(&elf)
            );

            println!(
                "cargo:rustc-env=FIRMWARE_{}={}",
                name.to_uppercase(),
                bin.display()
            );
        }
    }
}

/// Emit `rerun-if-changed` for every source file under `dir`, skipping build
/// output.
#[cfg(feature = "tenstorrent")]
fn track_sources(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name == "target" || name == ".git" {
            continue;
        }
        if path.is_dir() {
            track_sources(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Copy the firmware source tree (no `target/`, no `.git/`) into `dst`.
#[cfg(feature = "tenstorrent")]
fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("could not create the firmware build directory");
    for entry in std::fs::read_dir(src)
        .expect("could not read the firmware sources")
        .flatten()
    {
        let name = entry.file_name();
        if name == "target" || name == ".git" {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to).expect("could not copy a firmware source file");
        }
    }
}

/// Locate one of the `llvm-tools` binaries shipped with the active toolchain.
#[cfg(feature = "tenstorrent")]
fn tool(name: &str) -> PathBuf {
    let sysroot = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .arg("--print")
        .arg("sysroot")
        .output()
        .expect("could not ask rustc for its sysroot");
    let sysroot = String::from_utf8(sysroot.stdout).unwrap();
    let host = std::env::var("HOST").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".into());
    let path = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin")
        .join(name);
    assert!(
        path.exists(),
        "{name} not found at {}. Install it with `rustup component add llvm-tools`.",
        path.display()
    );
    path
}

/// Assert the image contains nothing Blackhole cannot execute.
///
/// This is a correctness gate, not hygiene. Each of these is a documented gap in
/// the baby RISC-V implementation, and executing an unimplemented instruction is
/// `UndefinedBehavior` — which, per the glossary, includes possible physical
/// damage. There is no illegal-instruction trap to fall back on.
#[cfg(feature = "tenstorrent")]
fn check_instruction_set(objdump: &Path, elf: &Path, name: &str) {
    // Every image is disassembled whole and judged. No image contains a
    // `.ttinsn` word: the compiler emits them as RISC-V stores into the
    // instruction FIFO, which the disassembler prints as ordinary `sw`s.
    let checked = judge(&disassemble(objdump, elf, name, None), name);

    // A parser that matches nothing passes everything. This check exists because
    // an earlier version of it did exactly that: llvm-objdump puts spaces between
    // the address colon and the mnemonic, and the first parser looked for ":\t".
    assert!(
        checked > 8,
        "the instruction-set check only recognised {checked} instructions in `{name}`, \
         which means it is not actually reading the disassembly"
    );
}

/// The disassembly of `elf`, or of `range` of it.
#[cfg(feature = "tenstorrent")]
fn disassemble(objdump: &Path, elf: &Path, name: &str, range: Option<(u64, u64)>) -> String {
    let mut cmd = Command::new(objdump);
    // `F` on top of the image's own attributes (rv32im): firmware reaches
    // the FP32 unit through inline assembly (`tt_firmware::float`), and
    // those instructions must decode so this gate can judge them -- it
    // refuses the ones Blackhole lacks or gets wrong, below -- rather than
    // meet them as `<unknown>`.
    cmd.args(["-d", "--no-show-raw-insn", "--mattr=+f"]);
    if let Some((lo, hi)) = range {
        cmd.arg(format!("--start-address={lo:#x}"))
            .arg(format!("--stop-address={hi:#x}"));
    }
    let out = cmd.arg(elf).output().expect("could not run llvm-objdump");
    assert!(out.status.success(), "objdump failed for {name}");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Judge every instruction line of `text`, and say how many there were.
#[cfg(feature = "tenstorrent")]
fn judge(text: &str, name: &str) -> usize {
    let mut checked = 0usize;
    for line in text.lines() {
        // Instruction lines are "<hex address>:<whitespace><mnemonic><whitespace><operands>".
        // Label lines ("00006000 <_start>:") end in ':' and have no mnemonic after it.
        let Some((addr, rest)) = line.split_once(':') else {
            continue;
        };
        if addr.trim().is_empty() || !addr.trim().chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let Some(mnemonic) = rest.split_whitespace().next() else {
            continue;
        };
        checked += 1;
        // Data in a code section. The assembler marks `.word` as data, so the
        // disassembler prints it as such instead of failing to decode it -- which
        // means an instruction the gate refuses (a compressed one, a `.ttinsn`
        // word) would pass as a harmless literal. No image has any.
        if rest.split_whitespace().any(|t| {
            matches!(
                t,
                ".word" | ".byte" | ".half" | ".short" | ".2byte" | ".4byte"
            )
        }) {
            panic!(
                "firmware `{name}` has data in a code section, which the gate cannot \
                 judge: the disassembler prints undecodable words as `.word`\n  in:{line}"
            );
        }
        let operands = rest.trim().strip_prefix(mnemonic).unwrap_or("").trim();
        if let Some(why) = forbidden_reason(mnemonic).or_else(|| fence_reason(mnemonic, operands)) {
            panic!(
                "firmware `{name}` contains `{mnemonic}`, which Blackhole does not \
                 implement: {why}\n  in:{line}"
            );
        }
    }
    checked
}

/// Why this `fence` must not appear, if it must not.
///
/// `pause` (Zihintpause) shares an encoding with `fence w, 0`, and a disassembler
/// that does not know the extension prints the latter — so checking the mnemonic
/// alone misses it. Blackhole does not list Zihintpause among the extensions it
/// implements, and `core::hint::spin_loop()` emits it, which is an easy way to
/// put an unimplemented instruction into a spin loop without noticing.
///
/// The permitted forms are the ordering fences the specification's own recipes
/// use. Anything else is flagged rather than assumed safe.
#[cfg(feature = "tenstorrent")]
fn fence_reason(mnemonic: &str, operands: &str) -> Option<&'static str> {
    if mnemonic != "fence" {
        return None;
    }
    const ALLOWED: &[&str] = &["", "iorw, iorw", "rw, rw", "rw, w", "r, rw"];
    if ALLOWED.contains(&operands) {
        return None;
    }
    if operands == "w, 0" {
        return Some(
            "this is the `pause` encoding (Zihintpause), which Blackhole does not implement. \
             `core::hint::spin_loop()` emits it -- use an empty asm block instead",
        );
    }
    Some(
        "an unrecognised fence ordering; only the forms the specification's recipes use are allowed",
    )
}

/// Why this mnemonic must not appear in a firmware image, if it must not.
///
/// Each is a documented gap in the baby RISC-V implementation, and executing an
/// unimplemented instruction is `UndefinedBehavior` — which per `Glossary.md`
/// includes possible physical damage. There is no illegal-instruction trap to
/// fall back on, so this is a correctness gate rather than hygiene.
#[cfg(feature = "tenstorrent")]
fn forbidden_reason(mnemonic: &str) -> Option<&'static str> {
    // Strip the ordering suffixes AMOs may carry, so `lr.w.aq` is caught too.
    let base = mnemonic.trim_end_matches(".aq").trim_end_matches(".rl");
    if mnemonic == "<unknown>" {
        return Some(
            "the disassembler could not decode it, so neither this gate nor a reader can \
             tell what it is. Usually a missing `.riscv.attributes` section or an \
             instruction outside the target's extension set",
        );
    }
    if mnemonic == "fence.i" {
        return Some(
            "Zifencei is not implemented. `fence.i` executes as a `nop`, which \
             `InstructionCache.md:23` classes as NonContractualBehavior, and ttsim refuses it \
             outright: \"fence.i does not flush the instruction cache on babyrisc and should \
             not be used\". Write the 5-bit mask to `RISCV_IC_INVALIDATE_InvalidateAll` instead",
        );
    }
    if mnemonic.starts_with("c.") {
        return Some(
            "the C extension is not implemented, and its encoding space is reused by `.ttinsn`",
        );
    }
    if base.starts_with("lr.") || base.starts_with("sc.") {
        return Some("Zalrsc is not implemented; there is no load-reserved/store-conditional");
    }
    // Every F-extension instruction, `fence` and `fence.i` aside (the latter
    // refused above): the baby cores move data and push instructions, and do
    // no floating-point arithmetic of their own. Float math is the SFPU's and
    // the Matrix Unit's, where it is fast and held to Flex; a slow scalar copy
    // on RISCV B is a trap someone would one day route a tensor through.
    if mnemonic.starts_with('f') && !mnemonic.starts_with("fence") {
        return Some(
            "no floating point on the baby cores: float math belongs on the SFPU \
             or the Matrix Unit (the data mover only moves data)",
        );
    }
    // Integer divide and remainder are allowed: every baby core implements
    // the whole M extension (`BabyRISCV/InstructionSet.md:7`). The only
    // divides missing are T2's vector `vdiv`/`vrem` (`InstructionSet.md:33`),
    // which no image uses. A scalar divide takes 6-33 cycles
    // (`BabyRISCV/README.md:51`); this gate once refused it on a misreading.
    None
}

/// `e_entry` of a 32-bit little-endian ELF.
#[cfg(feature = "tenstorrent")]
fn elf_entry(elf: &Path) -> u32 {
    let bytes = std::fs::read(elf).expect("could not read the firmware ELF");
    assert!(
        bytes.len() >= 28 && &bytes[..4] == b"\x7fELF" && bytes[4] == 1 && bytes[5] == 1,
        "{} is not a 32-bit little-endian ELF",
        elf.display()
    );
    u32::from_le_bytes(bytes[24..28].try_into().unwrap())
}

/// The C header, generated next to the library it declares.
///
/// Rendered from the sources, not kept by hand: `src/status.rs` is the one list
/// of failure codes and `src/abi.rs` the one list of entry points, so a new code
/// or function reaches the header on the next build. A function the table above
/// has no C spelling for fails the build rather than being left out.
fn write_c_header() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/status.rs");
    println!("cargo:rerun-if-changed=src/abi.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let status = std::fs::read_to_string(format!("{manifest}/src/status.rs"))
        .expect("src/status.rs is readable");
    let abi =
        std::fs::read_to_string(format!("{manifest}/src/abi.rs")).expect("src/abi.rs is readable");
    let header = render_header(&status, &abi);
    // `target/<profile>/build/<pkg>-<hash>/out`: the directory the library lands
    // in is three levels up.
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let mut profile = PathBuf::from(&out_dir);
    for _ in 0..3 {
        profile = profile
            .parent()
            .unwrap_or_else(|| panic!("OUT_DIR {out_dir} is not a cargo build directory"))
            .to_path_buf();
    }
    let path = profile.join("skamandros.h");
    std::fs::write(&path, header)
        .unwrap_or_else(|e| panic!("could not write {}: {e}", path.display()));
}

/// The header's text: the ABI version, the opaque module handle, the status
/// enum with its own doc comments, and one declaration per entry point.
fn render_header(status: &str, abi: &str) -> String {
    let version = abi
        .lines()
        .find_map(|line| line.trim().strip_prefix("pub const ABI_VERSION: u32 = "))
        .and_then(|rest| rest.trim_end_matches(';').trim().parse::<u32>().ok())
        .expect("src/abi.rs declares ABI_VERSION");
    let entries = abi_entry_points(abi);
    for (name, _, _) in DECLARATIONS {
        assert!(
            entries.iter().any(|entry| entry == name),
            "build.rs declares `{name}` and src/abi.rs has no such entry point"
        );
    }
    let mut out = String::new();
    out.push_str("/* skamandros.h - generated by build.rs from src/status.rs and src/abi.rs.\n");
    out.push_str(" * Do not edit it: change the sources and rebuild. The library is\n");
    out.push_str(
        " * libskamandros.so, next to this header in the build's target directory. */\n\n",
    );
    out.push_str("#ifndef SKAMANDROS_H\n#define SKAMANDROS_H\n\n");
    out.push_str("#include <stddef.h>\n#include <stdint.h>\n\n");
    out.push_str(&format!("#define TT_ABI_VERSION {version}\n\n"));
    out.push_str(
        "/* A compiled module. Opaque: the pointer is the handle, and its\n * contents are the library's. */\ntypedef struct ttModule ttModule;\n\n",
    );
    out.push_str("/* What every fallible call returns. */\ntypedef enum ttStatus {\n");
    for variant in status_variants(status) {
        out.push_str(&variant);
    }
    out.push_str("} ttStatus;\n\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
    for name in &entries {
        let (_, returns, params) = DECLARATIONS
            .iter()
            .find(|(declared, _, _)| declared == name)
            .unwrap_or_else(|| {
                panic!("src/abi.rs declares `{name}` and build.rs has no C declaration for it")
            });
        out.push_str(&format!("{returns} {name}({params});\n"));
    }
    out.push_str("\n#ifdef __cplusplus\n}\n#endif\n\n#endif\n");
    out
}

/// Every `ttStatus` variant, as a C enum member with its doc comment above it.
fn status_variants(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut doc: Vec<&str> = Vec::new();
    let mut inside = false;
    for line in source.lines() {
        let text = line.trim();
        if !inside {
            inside = text.starts_with("pub enum ttStatus {");
            continue;
        }
        if text == "}" {
            break;
        }
        if let Some(rest) = text.strip_prefix("///") {
            doc.push(rest.strip_prefix(' ').unwrap_or(rest));
            continue;
        }
        // The enum body is divided by `// ---- Shapes and definitions ----`
        // section comments; they are the source's, not a variant's.
        if text.starts_with("//") {
            continue;
        }
        if text.is_empty() || text.starts_with("#[") {
            continue;
        }
        let (name, value) = text
            .split_once('=')
            .unwrap_or_else(|| panic!("unexpected line in ttStatus: {text}"));
        let name = name.trim();
        let value = value.trim().trim_end_matches(',').trim();
        if doc.is_empty() {
            out.push(format!("    {name} = {value},\n"));
        } else {
            let mut member = String::from("    /* ");
            for (i, line) in doc.iter().enumerate() {
                if i > 0 {
                    member.push_str("\n       ");
                }
                // A doc comment may quote a C comment (`tt_llk_*/…`): the one
                // sequence that must not reach the inside of a C comment is
                // `*/`, so the line is spelled out until it cannot close it.
                member.push_str(&line.replace("*/", "* /"));
            }
            member.push_str(&format!(" */\n    {name} = {value},\n"));
            out.push(member);
        }
        doc.clear();
    }
    assert!(inside, "src/status.rs has no ttStatus enum");
    out
}

/// The `extern "C"` function names in `src/abi.rs`, in source order.
fn abi_entry_points(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in source.lines() {
        let text = line.trim();
        if !text.starts_with("pub ") || !text.contains("extern \"C\" fn tt") {
            continue;
        }
        let at = text.find("fn tt").expect("the function marker");
        let name = text[at + 3..]
            .split('(')
            .next()
            .expect("a function name")
            .trim()
            .to_string();
        names.push(name);
    }
    assert!(!names.is_empty(), "src/abi.rs declares no entry point");
    names
}
