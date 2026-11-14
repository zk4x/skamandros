// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! Every op, in every loop and grid shape the IR can express.
//!
//! The suite's single-program tests pin one shape each. This file is the other
//! axis: the same op table through a constant loop, a launch-bound (symbolic)
//! loop, a symbolic `2,2` grid, and a symbolic loop inside that grid. A grid is
//! one tile per core, selected from `grid.x`/`grid.y`; the loop is one page per
//! iteration; and the grid+loop program gives every core two pages of its own.
//!
//! Everything goes through the loaded C ABI, like `e2e.rs`.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use libloading::{Library, Symbol};

/// One device at a time: `libttsim` is a process-wide singleton.
fn device() -> MutexGuard<'static, ()> {
    static DEVICE: OnceLock<Mutex<()>> = OnceLock::new();
    DEVICE
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type Compile =
    unsafe extern "C" fn(*const c_char, usize, *mut *mut c_void, *mut *mut c_char) -> c_int;
type ModuleUnload = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_int;
type Free = unsafe extern "C" fn(*mut c_void);
type Init = unsafe extern "C" fn(*mut *mut c_char) -> c_int;
type Release = unsafe extern "C" fn(*mut *mut c_char) -> c_int;
type MemAlloc = unsafe extern "C" fn(usize, *mut u64, *mut *mut c_char) -> c_int;
type MemcpyH2D = unsafe extern "C" fn(u64, *const u8, usize, *mut *mut c_char) -> c_int;
type MemcpyD2H = unsafe extern "C" fn(*mut u8, u64, usize, *mut *mut c_char) -> c_int;
type Launch = unsafe extern "C" fn(*mut c_void, *const u64, usize, *mut *mut c_char) -> c_int;

const TT_OK: c_int = 0;

fn library_path() -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary's path");
    let dir = exe.parent().expect("the test binary's directory");
    dir.parent()
        .expect("the profile directory")
        .join(format!("{}skamandros.so", std::env::consts::DLL_PREFIX))
}

fn library() -> &'static Library {
    static LIB: OnceLock<Library> = OnceLock::new();
    LIB.get_or_init(|| {
        let path = library_path();
        build_library(&path);
        unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()))
    })
}

/// `cargo build --lib` in the profile this test binary was built in: the
/// `cdylib` next to it is only as fresh as the last build.
fn build_library(path: &Path) {
    let profile = path
        .parent()
        .and_then(Path::file_name)
        .expect("a profile directory")
        .to_string_lossy()
        .into_owned();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut command = std::process::Command::new(cargo);
    command
        .arg("build")
        .arg("--lib")
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    match profile.as_str() {
        "debug" => {}
        "release" => {
            command.arg("--release");
        }
        other => {
            command.args(["--profile", other]);
        }
    }
    let out = command.output().expect("cargo runs");
    assert!(
        out.status.success(),
        "cargo build --lib failed for {profile}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn simulator_path() -> PathBuf {
    if let Some(p) = std::env::var_os("TT_METAL_SIMULATOR") {
        return PathBuf::from(p);
    }
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/libttsim_bh.so");
    if local.exists() {
        return local;
    }
    panic!("no libttsim_bh.so: set TT_METAL_SIMULATOR to the simulator's shared object");
}

fn take_error(free: &Symbol<Free>, err: *mut c_char) -> String {
    if err.is_null() {
        return String::from("<no message>");
    }
    let msg = unsafe { CStr::from_ptr(err) }
        .to_string_lossy()
        .into_owned();
    unsafe { free(err as *mut c_void) };
    msg
}

/// Compile `program`, upload `inputs`, launch with `scalars` first and the
/// output parameter last, and return the output bytes.
fn run_kernel_scalars(
    program: &str,
    scalars: &[u64],
    inputs: &[&[u8]],
    out_bytes: usize,
) -> Vec<u8> {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );
    let mut tensors: Vec<u64> = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let mut at = 0u64;
        assert_eq!(
            unsafe { alloc(input.len(), &mut at, &mut err) },
            TT_OK,
            "input {i}: {}",
            take_error(&free, err)
        );
        assert_eq!(
            unsafe { h2d(at, input.as_ptr(), input.len(), &mut err) },
            TT_OK,
            "upload {i}: {}",
            take_error(&free, err)
        );
        tensors.push(at);
    }
    let mut output_at = 0u64;
    assert_eq!(
        unsafe { alloc(out_bytes, &mut output_at, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );
    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            program.as_ptr().cast(),
            program.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));

    let mut params: Vec<u64> = scalars.to_vec();
    params.extend_from_slice(&tensors);
    params.push(output_at);
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; out_bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output_at, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
    back
}

// ---------------------------------------------------------------------------
// The programs
// ---------------------------------------------------------------------------

/// A page-major host pattern that is positive, away from zero and fractional:
/// safe for `log2`, `sqrt`, `rsqrt`, `reciprocal` and `div`, and visible to
/// `floor`/`trunc`. `seed` picks a different pattern for a second operand.
fn f32_values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| 0.25 + (((i * 7 + seed * 5 + 3) % 19) as f32) * 0.3125)
        .collect()
}

fn i32_values(count: usize, seed: usize) -> Vec<i32> {
    (0..count)
        .map(|i| {
            let v = ((i * 37 + seed * 11) % 61) as i32 - 30;
            v
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn check_f32_pages(got: &[u8], want: &[f32], tol: f32, what: &str) {
    let got: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got.len(), want.len(), "{what}: output size");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        assert!(
            err <= tol,
            "{what}: datum {i}: got {g}, want {w} (rel {err} > {tol})"
        );
    }
}

fn check_i32_pages(got: &[u8], want: &[i32], what: &str) {
    let got: Vec<i32> = got
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got.len(), want.len(), "{what}: output size");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "{what}: datum {i}");
    }
}

/// The bf16 datum a lane value truncates to: the SFPU's `BF16` store keeps the
/// high half of the f32 (`0x43058000` becomes `0x4305`), so a 16-bit `Dst`
/// kernel computes at f32 precision and lands at bf16 precision at the store.
fn truncate_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// A bf16 datum widened into the 32-bit lane. The SFPU flushes a datum whose
/// exponent is zero (a subnormal, or a zero of either sign).
fn widen_bf16(bits: u16) -> f32 {
    if bits & 0x7f80 == 0 {
        0.0
    } else {
        f32::from_bits(u32::from(bits) << 16)
    }
}

/// The same positive, bounded values the f32 matrix uses, carried as the bf16
/// datums a caller would hand in.
fn bf16_values(count: usize, seed: usize) -> Vec<u16> {
    f32_values(count, seed)
        .iter()
        .map(|v| truncate_to_bf16(*v))
        .collect()
}

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|h| h.to_le_bytes()).collect()
}

/// Every datum of a bf16 kernel against the widen/op/truncate rule: the input
/// is the exact f32 a bf16 widens to, the op runs in the f32 lane, and the
/// store truncates. A case with a tolerance is an instruction the SFPU
/// approximates, so the check allows two bf16 ulps; `tol == 0` is exact.
fn check_bf16_pages(got: &[u8], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len() * 2, "{what}: output size");
    for (i, w) in want.iter().enumerate() {
        let g = u16::from_le_bytes(got[2 * i..2 * i + 2].try_into().unwrap());
        let expect = truncate_to_bf16(*w);
        if tol == 0.0 {
            assert_eq!(
                g, expect,
                "{what}: datum {i}: got {g:#06x}, want {expect:#06x} ({w})"
            );
        } else {
            let (gv, wv) = (widen_bf16(g), *w);
            let err = (gv - wv).abs() / wv.abs().max(1e-6);
            let allowed = 0.008f32.max(tol);
            assert!(
                err <= allowed,
                "{what}: datum {i}: got {gv}, want {wv} (rel {err} > {allowed})"
            );
        }
    }
}

/// `pages` pages through one ring in a loop, one unary op in the body.
fn unary_looped(dtype: &str, op: &str, pages: usize, bound: &str, scalars: &str) -> String {
    let size = pages * 1024;
    format!(
        "\
version 9
grid 1,1
{scalars}%px = param {dtype} {size}
%out = param {dtype} {size}
%bx = buffer {dtype} 32,32
%bo = buffer {dtype} 32,32
%j = loop {bound}
copy 32,32 %bx [0] = %px [%j]
load 32,32 %x = %bx [0]
%y = {op} %x
store 32,32 %bo [0] = %y
copy 32,32 %out [%j] = %bo [0]
endloop
"
    )
}

/// The same in a two-operand body; `body` names `%x` and `%c` and defines `%y`.
fn binary_looped(dtype: &str, body: &str, pages: usize, bound: &str, scalars: &str) -> String {
    let size = pages * 1024;
    format!(
        "\
version 9
grid 1,1
{scalars}%px = param {dtype} {size}
%pc = param {dtype} {size}
%out = param {dtype} {size}
%bx = buffer {dtype} 32,32
%bc = buffer {dtype} 32,32
%bo = buffer {dtype} 32,32
%j = loop {bound}
copy 32,32 %bx [0] = %px [%j]
copy 32,32 %bc [0] = %pc [%j]
load 32,32 %x = %bx [0]
load 32,32 %c = %bc [0]
{body}
store 32,32 %bo [0] = %y
copy 32,32 %out [%j] = %bo [0]
endloop
"
    )
}

/// A `2,2` launch grid whose extents are scalars; each core takes the tile its
/// `(x, y)` names and writes it back to the same place.
fn unary_grid(dtype: &str, op: &str) -> String {
    format!(
        "\
version 9
%nx = param i32 0
%ny = param i32 0
%px = param {dtype} 4096
%out = param {dtype} 4096
%gx = grid.x
%gy = grid.y
%row = mul %gy %nx
%idx = add %row %gx
%bx = buffer {dtype} 32,32
%bo = buffer {dtype} 32,32
copy 32,32 %bx [0] = %px [%idx]
load 32,32 %v = %bx [0]
%r = {op} %v
store 32,32 %bo [0] = %r
copy 32,32 %out [%idx] = %bo [0]
grid %nx,%ny
"
    )
}

fn binary_grid(dtype: &str, body: &str) -> String {
    format!(
        "\
version 9
%nx = param i32 0
%ny = param i32 0
%px = param {dtype} 4096
%pc = param {dtype} 4096
%out = param {dtype} 4096
%gx = grid.x
%gy = grid.y
%row = mul %gy %nx
%idx = add %row %gx
%bx = buffer {dtype} 32,32
%bc = buffer {dtype} 32,32
%bo = buffer {dtype} 32,32
copy 32,32 %bx [0] = %px [%idx]
copy 32,32 %bc [0] = %pc [%idx]
load 32,32 %x = %bx [0]
load 32,32 %c = %bc [0]
{body}
store 32,32 %bo [0] = %y
copy 32,32 %out [%idx] = %bo [0]
grid %nx,%ny
"
    )
}

/// A `2,2` symbolic grid whose cores each run a symbolic loop over two pages.
fn unary_grid_loop(dtype: &str, op: &str) -> String {
    format!(
        "\
version 9
%nx = param i32 0
%ny = param i32 0
%n = param i32 0
%px = param {dtype} 8192
%out = param {dtype} 8192
%gx = grid.x
%gy = grid.y
%row = mul %gy %nx
%idx = add %row %gx
%two = const i32 2
%base = mul %idx %two
%bx = buffer {dtype} 32,32
%bo = buffer {dtype} 32,32
%j = loop %n
%off = add %base %j
copy 32,32 %bx [0] = %px [%off]
load 32,32 %v = %bx [0]
%r = {op} %v
store 32,32 %bo [0] = %r
copy 32,32 %out [%off] = %bo [0]
endloop
grid %nx,%ny
"
    )
}

fn binary_grid_loop(dtype: &str, body: &str) -> String {
    format!(
        "\
version 9
%nx = param i32 0
%ny = param i32 0
%n = param i32 0
%px = param {dtype} 8192
%pc = param {dtype} 8192
%out = param {dtype} 8192
%gx = grid.x
%gy = grid.y
%row = mul %gy %nx
%idx = add %row %gx
%two = const i32 2
%base = mul %idx %two
%bx = buffer {dtype} 32,32
%bc = buffer {dtype} 32,32
%bo = buffer {dtype} 32,32
%j = loop %n
%off = add %base %j
copy 32,32 %bx [0] = %px [%off]
copy 32,32 %bc [0] = %pc [%off]
load 32,32 %x = %bx [0]
load 32,32 %c = %bc [0]
{body}
store 32,32 %bo [0] = %y
copy 32,32 %out [%off] = %bo [0]
endloop
grid %nx,%ny
"
    )
}

// ---------------------------------------------------------------------------
// The op tables
// ---------------------------------------------------------------------------

type UnaryCase = (&'static str, fn(f32) -> f32, f32);
type BinaryCase = (&'static str, fn(f32, f32) -> f32, f32);

const UNARY_CASES: &[UnaryCase] = &[
    ("neg", |x| -x, 0.0),
    ("abs", |x| x.abs(), 0.0),
    ("floor", |x| x.floor(), 0.0),
    ("trunc", |x| x.trunc(), 0.0),
    ("exp2", |x| x.exp2(), 1e-5),
    ("log2", |x| x.log2(), 1e-5),
    ("reciprocal", |x| 1.0 / x, 1e-6),
    ("sqrt", |x| x.sqrt(), 1e-5),
    ("rsqrt", |x| 1.0 / x.sqrt(), 1e-5),
    ("sin", f32::sin, 1e-5),
    ("cos", f32::cos, 1e-5),
];

const BINARY_CASES: &[BinaryCase] = &[
    ("%y = add %x %c", |a, b| a + b, 0.0),
    ("%y = sub %x %c", |a, b| a - b, 0.0),
    ("%y = mul %x %c", |a, b| a * b, 0.0),
    ("%y = div %x %c", |a, b| a / b, 1e-6),
    ("%y = pow %x %c", |a, b| a.powf(b), 1e-4),
    ("%y = max %x %c", |a, b| a.max(b), 0.0),
    ("%y = min %x %c", |a, b| a.min(b), 0.0),
    ("%t = cmplt %x %c\n%y = sel %t %x %c", |a, b| a.min(b), 0.0),
    ("%t = cmpgt %x %c\n%y = sel %t %x %c", |a, b| a.max(b), 0.0),
    ("%t = cmpge %x %c\n%y = sel %t %x %c", |a, b| a.max(b), 0.0),
    (
        "%t = noteq %x %c\n%y = sel %t %x %c",
        |a, b| if a != b { a } else { b },
        0.0,
    ),
    (
        "%t = eq %x %c\n%y = sel %t %x %c",
        |a, b| if a == b { a } else { b },
        0.0,
    ),
];

type I32UnaryCase = (&'static str, fn(i32) -> i32);
type I32BinaryCase = (&'static str, fn(i32, i32) -> i32);

const I32_UNARY_CASES: &[I32UnaryCase] = &[("bitnot", |x| !x)];

const I32_BINARY_CASES: &[I32BinaryCase] = &[
    ("%y = bitand %x %c", |a, b| a & b),
    ("%y = bitor %x %c", |a, b| a | b),
    ("%y = bitxor %x %c", |a, b| a ^ b),
    ("%y = bitshiftleft %x %c", |a, b| a.wrapping_shl(b as u32)),
    // A shift moves the bits: zeros in, even for a negative `i32`.
    ("%y = bitshiftright %x %c", |a, b| ((a as u32) >> b) as i32),
    ("%y = mod %x %c", |a, b| a % b),
];

// ---------------------------------------------------------------------------
// Unary, four contexts
// ---------------------------------------------------------------------------

#[test]
fn every_unary_op_runs_in_a_loop() {
    let x = f32_values(3072, 0);
    let input = f32_bytes(&x);
    for &(op, want, tol) in UNARY_CASES {
        let program = unary_looped("f32", op, 3, "3", "");
        let out = run_kernel_scalars(&program, &[], &[&input], 3072 * 4);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_f32_pages(&out, &expect, tol, op);
    }
}

#[test]
fn every_unary_op_runs_in_a_symbolic_loop() {
    let x = f32_values(3072, 0);
    let input = f32_bytes(&x);
    for &(op, want, tol) in UNARY_CASES {
        let program = unary_looped("f32", op, 3, "%n", "%n = param i32 0\n");
        let out = run_kernel_scalars(&program, &[3], &[&input], 3072 * 4);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_f32_pages(&out, &expect, tol, op);
    }
}

#[test]
fn every_unary_op_runs_on_a_symbolic_grid() {
    let x = f32_values(4096, 0);
    let input = f32_bytes(&x);
    for &(op, want, tol) in UNARY_CASES {
        let program = unary_grid("f32", op);
        let out = run_kernel_scalars(&program, &[2, 2], &[&input], 4096 * 4);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_f32_pages(&out, &expect, tol, op);
    }
}

#[test]
fn every_unary_op_runs_in_a_loop_on_a_symbolic_grid() {
    let x = f32_values(8192, 0);
    let input = f32_bytes(&x);
    for &(op, want, tol) in UNARY_CASES {
        let program = unary_grid_loop("f32", op);
        let out = run_kernel_scalars(&program, &[2, 2, 2], &[&input], 8192 * 4);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_f32_pages(&out, &expect, tol, op);
    }
}

// ---------------------------------------------------------------------------
// Binary, four contexts
// ---------------------------------------------------------------------------

#[test]
fn every_binary_op_runs_in_a_loop() {
    let x = f32_values(3072, 0);
    let c = f32_values(3072, 1);
    let (bx, bc) = (f32_bytes(&x), f32_bytes(&c));
    for &(body, want, tol) in BINARY_CASES {
        let program = binary_looped("f32", body, 3, "3", "");
        let out = run_kernel_scalars(&program, &[], &[&bx, &bc], 3072 * 4);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_f32_pages(&out, &expect, tol, body);
    }
}

#[test]
fn every_binary_op_runs_in_a_symbolic_loop() {
    let x = f32_values(3072, 0);
    let c = f32_values(3072, 1);
    let (bx, bc) = (f32_bytes(&x), f32_bytes(&c));
    for &(body, want, tol) in BINARY_CASES {
        let program = binary_looped("f32", body, 3, "%n", "%n = param i32 0\n");
        let out = run_kernel_scalars(&program, &[3], &[&bx, &bc], 3072 * 4);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_f32_pages(&out, &expect, tol, body);
    }
}

#[test]
fn every_binary_op_runs_on_a_symbolic_grid() {
    let x = f32_values(4096, 0);
    let c = f32_values(4096, 1);
    let (bx, bc) = (f32_bytes(&x), f32_bytes(&c));
    for &(body, want, tol) in BINARY_CASES {
        let program = binary_grid("f32", body);
        let out = run_kernel_scalars(&program, &[2, 2], &[&bx, &bc], 4096 * 4);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_f32_pages(&out, &expect, tol, body);
    }
}

#[test]
fn every_binary_op_runs_in_a_loop_on_a_symbolic_grid() {
    let x = f32_values(8192, 0);
    let c = f32_values(8192, 1);
    let (bx, bc) = (f32_bytes(&x), f32_bytes(&c));
    for &(body, want, tol) in BINARY_CASES {
        let program = binary_grid_loop("f32", body);
        let out = run_kernel_scalars(&program, &[2, 2, 2], &[&bx, &bc], 8192 * 4);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_f32_pages(&out, &expect, tol, body);
    }
}

// ---------------------------------------------------------------------------
// i32, four contexts
// ---------------------------------------------------------------------------

#[test]
fn every_i32_op_runs_in_a_loop() {
    let x = i32_values(3072, 0);
    let c: Vec<i32> = (0..3072).map(|i| (i % 7) as i32 + 1).collect();
    let (bx, bc) = (i32_bytes(&x), i32_bytes(&c));
    for &(op, want) in I32_UNARY_CASES {
        let program = unary_looped("i32", op, 3, "3", "");
        let out = run_kernel_scalars(&program, &[], &[&bx], 3072 * 4);
        let expect: Vec<i32> = x.iter().map(|v| want(*v)).collect();
        check_i32_pages(&out, &expect, op);
    }
    for &(body, want) in I32_BINARY_CASES {
        let program = binary_looped("i32", body, 3, "3", "");
        let out = run_kernel_scalars(&program, &[], &[&bx, &bc], 3072 * 4);
        let expect: Vec<i32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_i32_pages(&out, &expect, body);
    }
}

#[test]
fn every_i32_op_runs_in_a_symbolic_loop() {
    let x = i32_values(3072, 0);
    let c: Vec<i32> = (0..3072).map(|i| (i % 7) as i32 + 1).collect();
    let (bx, bc) = (i32_bytes(&x), i32_bytes(&c));
    for &(op, want) in I32_UNARY_CASES {
        let program = unary_looped("i32", op, 3, "%n", "%n = param i32 0\n");
        let out = run_kernel_scalars(&program, &[3], &[&bx], 3072 * 4);
        let expect: Vec<i32> = x.iter().map(|v| want(*v)).collect();
        check_i32_pages(&out, &expect, op);
    }
    for &(body, want) in I32_BINARY_CASES {
        let program = binary_looped("i32", body, 3, "%n", "%n = param i32 0\n");
        let out = run_kernel_scalars(&program, &[3], &[&bx, &bc], 3072 * 4);
        let expect: Vec<i32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_i32_pages(&out, &expect, body);
    }
}

#[test]
fn every_i32_op_runs_on_a_symbolic_grid() {
    let x = i32_values(4096, 0);
    let c: Vec<i32> = (0..4096).map(|i| (i % 7) as i32 + 1).collect();
    let (bx, bc) = (i32_bytes(&x), i32_bytes(&c));
    for &(op, want) in I32_UNARY_CASES {
        let program = unary_grid("i32", op);
        let out = run_kernel_scalars(&program, &[2, 2], &[&bx], 4096 * 4);
        let expect: Vec<i32> = x.iter().map(|v| want(*v)).collect();
        check_i32_pages(&out, &expect, op);
    }
    for &(body, want) in I32_BINARY_CASES {
        let program = binary_grid("i32", body);
        let out = run_kernel_scalars(&program, &[2, 2], &[&bx, &bc], 4096 * 4);
        let expect: Vec<i32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_i32_pages(&out, &expect, body);
    }
}

#[test]
fn every_i32_op_runs_in_a_loop_on_a_symbolic_grid() {
    let x = i32_values(8192, 0);
    let c: Vec<i32> = (0..8192).map(|i| (i % 7) as i32 + 1).collect();
    let (bx, bc) = (i32_bytes(&x), i32_bytes(&c));
    for &(op, want) in I32_UNARY_CASES {
        let program = unary_grid_loop("i32", op);
        let out = run_kernel_scalars(&program, &[2, 2, 2], &[&bx], 8192 * 4);
        let expect: Vec<i32> = x.iter().map(|v| want(*v)).collect();
        check_i32_pages(&out, &expect, op);
    }
    for &(body, want) in I32_BINARY_CASES {
        let program = binary_grid_loop("i32", body);
        let out = run_kernel_scalars(&program, &[2, 2, 2], &[&bx, &bc], 8192 * 4);
        let expect: Vec<i32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_i32_pages(&out, &expect, body);
    }
}

// ---------------------------------------------------------------------------
// bf16, four contexts
// ---------------------------------------------------------------------------
//
// The bf16 column of the matrix. A 16-bit `Dst` kernel stores every datum as
// bf16, so the op runs at f32 lane precision and the store truncates; the
// oracle widens the bf16 input, applies the f32 rule, and truncates, which is
// exact for every non-approximated instruction.

/// The f32 the lane sees: the bf16 input widened.
fn widened(values: &[u16]) -> Vec<f32> {
    values.iter().map(|h| widen_bf16(*h)).collect()
}

/// The bf16 matrix's unary set: the instructions the SFPU runs on a 16-bit
/// `Dst`. `floor`/`trunc`/`log2`/`sin`/`cos`/`reciprocal`/`sqrt` are unfolded
/// into f32 bit manipulation (`bitcast` to `i32`, f32 constants), so a bf16
/// operand has no spelling there yet; they are absent from this table rather
/// than silently made f32.
const BF16_UNARY_CASES: &[UnaryCase] = &[
    ("neg", |x| -x, 0.0),
    ("abs", |x| x.abs(), 0.0),
    // `exp2` keeps `lower`'s machine body, so it runs in a 16-bit `Dst`.
    ("exp2", |x| x.exp2(), 1e-5),
];

/// Every `BOp` the lane takes directly in 16-bit mode: the arithmetic, the two
/// folds, and the ordered comparisons through `sel` (the `bool` lane is the
/// `0xBF80`/`0` the store keeps).
const BF16_BINARY_CASES: &[BinaryCase] = &[
    ("%y = add %x %c", |a, b| a + b, 0.0),
    ("%y = sub %x %c", |a, b| a - b, 0.0),
    ("%y = mul %x %c", |a, b| a * b, 0.0),
    ("%y = max %x %c", |a, b| a.max(b), 0.0),
    ("%y = min %x %c", |a, b| a.min(b), 0.0),
    ("%t = cmplt %x %c\n%y = sel %t %x %c", |a, b| a.min(b), 0.0),
    ("%t = cmpgt %x %c\n%y = sel %t %x %c", |a, b| a.max(b), 0.0),
    ("%t = cmpge %x %c\n%y = sel %t %x %c", |a, b| a.max(b), 0.0),
    (
        "%t = noteq %x %c\n%y = sel %t %x %c",
        |a, b| if a != b { a } else { b },
        0.0,
    ),
    (
        "%t = eq %x %c\n%y = sel %t %x %c",
        |a, b| if a == b { a } else { b },
        0.0,
    ),
];

#[test]
fn every_unary_op_runs_in_a_loop_in_bf16() {
    let bits = bf16_values(3072, 0);
    let input = bf16_bytes(&bits);
    let x = widened(&bits);
    for &(op, want, tol) in BF16_UNARY_CASES {
        let program = unary_looped("bf16", op, 3, "3", "");
        let out = run_kernel_scalars(&program, &[], &[&input], 3072 * 2);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_bf16_pages(&out, &expect, tol, op);
    }
}

#[test]
fn every_binary_op_runs_in_a_loop_in_bf16() {
    let (xb, cb) = (bf16_values(3072, 0), bf16_values(3072, 1));
    let (bx, bc) = (bf16_bytes(&xb), bf16_bytes(&cb));
    let (x, c) = (widened(&xb), widened(&cb));
    for &(body, want, tol) in BF16_BINARY_CASES {
        let program = binary_looped("bf16", body, 3, "3", "");
        let out = run_kernel_scalars(&program, &[], &[&bx, &bc], 3072 * 2);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_bf16_pages(&out, &expect, tol, body);
    }
}

#[test]
fn every_unary_op_runs_in_a_symbolic_loop_in_bf16() {
    let bits = bf16_values(3072, 0);
    let input = bf16_bytes(&bits);
    let x = widened(&bits);
    for &(op, want, tol) in BF16_UNARY_CASES {
        let program = unary_looped("bf16", op, 3, "%n", "%n = param i32 0\n");
        let out = run_kernel_scalars(&program, &[3], &[&input], 3072 * 2);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_bf16_pages(&out, &expect, tol, op);
    }
}

#[test]
fn every_binary_op_runs_in_a_symbolic_loop_in_bf16() {
    let (xb, cb) = (bf16_values(3072, 0), bf16_values(3072, 1));
    let (bx, bc) = (bf16_bytes(&xb), bf16_bytes(&cb));
    let (x, c) = (widened(&xb), widened(&cb));
    for &(body, want, tol) in BF16_BINARY_CASES {
        let program = binary_looped("bf16", body, 3, "%n", "%n = param i32 0\n");
        let out = run_kernel_scalars(&program, &[3], &[&bx, &bc], 3072 * 2);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_bf16_pages(&out, &expect, tol, body);
    }
}

#[test]
fn every_unary_op_runs_on_a_symbolic_grid_in_bf16() {
    let bits = bf16_values(4096, 0);
    let input = bf16_bytes(&bits);
    let x = widened(&bits);
    for &(op, want, tol) in BF16_UNARY_CASES {
        let program = unary_grid("bf16", op);
        let out = run_kernel_scalars(&program, &[2, 2], &[&input], 4096 * 2);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_bf16_pages(&out, &expect, tol, op);
    }
}

#[test]
fn every_binary_op_runs_on_a_symbolic_grid_in_bf16() {
    let (xb, cb) = (bf16_values(4096, 0), bf16_values(4096, 1));
    let (bx, bc) = (bf16_bytes(&xb), bf16_bytes(&cb));
    let (x, c) = (widened(&xb), widened(&cb));
    for &(body, want, tol) in BF16_BINARY_CASES {
        let program = binary_grid("bf16", body);
        let out = run_kernel_scalars(&program, &[2, 2], &[&bx, &bc], 4096 * 2);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_bf16_pages(&out, &expect, tol, body);
    }
}

#[test]
fn every_unary_op_runs_in_a_loop_on_a_symbolic_grid_in_bf16() {
    let bits = bf16_values(8192, 0);
    let input = bf16_bytes(&bits);
    let x = widened(&bits);
    for &(op, want, tol) in BF16_UNARY_CASES {
        let program = unary_grid_loop("bf16", op);
        let out = run_kernel_scalars(&program, &[2, 2, 2], &[&input], 8192 * 2);
        let expect: Vec<f32> = x.iter().map(|v| want(*v)).collect();
        check_bf16_pages(&out, &expect, tol, op);
    }
}

#[test]
fn every_binary_op_runs_in_a_loop_on_a_symbolic_grid_in_bf16() {
    let (xb, cb) = (bf16_values(8192, 0), bf16_values(8192, 1));
    let (bx, bc) = (bf16_bytes(&xb), bf16_bytes(&cb));
    let (x, c) = (widened(&xb), widened(&cb));
    for &(body, want, tol) in BF16_BINARY_CASES {
        let program = binary_grid_loop("bf16", body);
        let out = run_kernel_scalars(&program, &[2, 2, 2], &[&bx, &bc], 8192 * 2);
        let expect: Vec<f32> = x.iter().zip(&c).map(|(a, b)| want(*a, *b)).collect();
        check_bf16_pages(&out, &expect, tol, body);
    }
}
