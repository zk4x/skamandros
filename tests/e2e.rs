// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! End to end through the shipped artifact: load `libTT.so` with `dlopen`, call
//! the C ABI, and read the module back.
//!
//! Nothing here links the Rust library: the point is that a consumer with no Rust
//! sees exactly this surface.

use std::ffi::{CStr, c_char, c_int};
use std::path::PathBuf;

use libloading::{Library, Symbol};

type AbiVersion = unsafe extern "C" fn() -> u32;
type Compile = unsafe extern "C" fn(*const u8, usize, *mut *mut u8, *mut usize, *mut *mut c_char, *mut usize) -> c_int;
type Free = unsafe extern "C" fn(*mut u8, usize);

/// The `libTT.so` this test binary was built next to.
fn library_path() -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary's path");
    let dir = exe.parent().expect("the test binary's directory");
    // `target/<profile>/deps/<test>` -> `target/<profile>/libTT.so`
    dir.parent().expect("the profile directory").join(format!("{}TT.so", std::env::consts::DLL_PREFIX))
}

const COPY: &str = "\
version 1
grid 1 1
param in f32 8192
param out f32 8192
buf 0 4096 plain rm
buf 1 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
copy_out buf 1 param 1 offset 64 len 4096
";

#[test]
fn the_so_compiles_a_program_through_the_c_abi() {
    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let version: Symbol<AbiVersion> = unsafe { lib.get(b"tt_abi_version\0") }.expect("tt_abi_version");
    let compile: Symbol<Compile> = unsafe { lib.get(b"tt_compile\0") }.expect("tt_compile");
    let free: Symbol<Free> = unsafe { lib.get(b"tt_free\0") }.expect("tt_free");
    assert_eq!(unsafe { version() }, 1, "the ABI version");

    let (mut out, mut out_len) = (std::ptr::null_mut(), 0usize);
    let (mut err, mut err_len) = (std::ptr::null_mut(), 0usize);
    let code = unsafe {
        compile(COPY.as_ptr(), COPY.len(), &mut out, &mut out_len, &mut err, &mut err_len)
    };
    if code != 0 {
        let msg = if err.is_null() { String::from("<no message>") } else { unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned() };
        unsafe { free(err as *mut u8, err_len) };
        panic!("tt_compile failed with {code}: {msg}");
    }
    assert!(!out.is_null() && out_len >= 16, "a module came back");
    let bytes = unsafe { std::slice::from_raw_parts(out, out_len) };
    assert_eq!(&bytes[..4], b"TTMD", "the module magic");
    let module = TT::compile::Module::from_bytes(bytes).expect("the module reads back");
    unsafe { free(out, out_len) };

    // The mover programs are real RISC-V: the reader ends in `ret`.
    assert!(!module.reader.is_empty(), "the reader program");
    assert_eq!(*module.reader.last().unwrap(), 0x0000_8067, "the reader ends in ret");
    assert!(!module.writer.is_empty(), "the writer program");
    assert_eq!(*module.writer.last().unwrap(), 0x0000_8067, "the writer ends in ret");
    assert!(module.cores.iter().all(|c| c.is_empty()), "a copy kernel has no compute");
    assert_eq!(module.grid, [1, 1]);
}

#[test]
fn a_module_can_be_decoded_through_the_c_abi() {
    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let compile: Symbol<Compile> = unsafe { lib.get(b"tt_compile\0") }.expect("tt_compile");
    let free: Symbol<Free> = unsafe { lib.get(b"tt_free\0") }.expect("tt_free");
    let grid: Symbol<unsafe extern "C" fn(*const u8, usize, *mut u32, *mut u32) -> c_int> =
        unsafe { lib.get(b"tt_module_grid\0") }.expect("tt_module_grid");
    let program: Symbol<unsafe extern "C" fn(*const u8, usize, u32, *mut u32, usize, *mut usize) -> c_int> =
        unsafe { lib.get(b"tt_module_program\0") }.expect("tt_module_program");

    let (mut out, mut out_len) = (std::ptr::null_mut(), 0usize);
    let (mut err, mut err_len) = (std::ptr::null_mut(), 0usize);
    let code = unsafe { compile(COPY.as_ptr(), COPY.len(), &mut out, &mut out_len, &mut err, &mut err_len) };
    assert_eq!(code, 0, "the copy kernel compiles");

    let (mut gx, mut gy) = (0u32, 0u32);
    assert_eq!(unsafe { grid(out, out_len, &mut gx, &mut gy) }, 0);
    assert_eq!((gx, gy), (1, 1), "the grid a module launches over");

    // Query the reader program's length, then fetch it into a caller buffer.
    let mut len = 0usize;
    assert_eq!(unsafe { program(out, out_len, 0, std::ptr::null_mut(), 0, &mut len) }, 0);
    assert!(len > 0, "the reader program has words");
    let mut words = vec![0u32; len];
    assert_eq!(unsafe { program(out, out_len, 0, words.as_mut_ptr(), words.len(), &mut len) }, 0);
    assert_eq!(len, words.len());
    assert_eq!(*words.last().unwrap(), 0x0000_8067, "the reader ends in ret");

    // A core with no program reports length 0, and a sixth core is refused.
    let mut empty = 0usize;
    assert_eq!(unsafe { program(out, out_len, 2, std::ptr::null_mut(), 0, &mut empty) }, 0);
    assert_eq!(empty, 0, "a copy kernel has no T0 program");
    assert_ne!(unsafe { program(out, out_len, 5, std::ptr::null_mut(), 0, &mut empty) }, 0);
    // A buffer too small is an error, not a truncation.
    assert_ne!(unsafe { program(out, out_len, 0, words.as_mut_ptr(), words.len() - 1, &mut len) }, 0);

    unsafe { free(out, out_len) };
}

#[test]
fn a_refused_program_comes_back_as_an_error_not_a_crash() {
    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let compile: Symbol<Compile> = unsafe { lib.get(b"tt_compile\0") }.expect("tt_compile");
    let free: Symbol<Free> = unsafe { lib.get(b"tt_free\0") }.expect("tt_free");

    let bad = "version 1\ngrid 1 1\nparam out f32 4096\nbuf 0 4096 plain rm\n%0 = sfpu neg %1\n";
    let (mut out, mut out_len) = (std::ptr::null_mut(), 0usize);
    let (mut err, mut err_len) = (std::ptr::null_mut(), 0usize);
    let code = unsafe { compile(bad.as_ptr(), bad.len(), &mut out, &mut out_len, &mut err, &mut err_len) };
    assert_ne!(code, 0, "a use before its definition is refused");
    assert!(out.is_null() && out_len == 0);
    assert!(!err.is_null(), "there is a message");
    let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned();
    unsafe { free(err as *mut u8, err_len) };
    assert!(msg.contains("%1"), "{msg}");
}

/// A copy kernel: DRAM into L1, and the same L1 buffer back into DRAM.
const COPY_8K: &str = "\
version 1
grid 1 1
param in f32 8192
param out f32 8192
buf 0 8192 plain rm
copy_in param 0 offset 0 len 8192 buf 0
copy_out buf 0 param 1 offset 0 len 8192
";

/// Where the simulator is. The library only reads `TT_TTSIM_LIB`, so the test
/// supplies it: the simulator is not part of this repository.
fn simulator_path() -> PathBuf {
    if let Some(p) = std::env::var_os("TT_TTSIM_LIB") {
        return PathBuf::from(p);
    }
    let mut candidates = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/libttsim_bh.so")];
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join("Dev/rust/metalium-rs/vendor/libttsim_bh.so"));
    }
    match candidates.into_iter().find(|p| p.exists()) {
        Some(p) => p,
        None => panic!("no libttsim_bh.so: set TT_TTSIM_LIB to the simulator's shared object"),
    }
}

type Init = unsafe extern "C" fn(*mut *mut c_char, *mut usize) -> c_int;
type Release = unsafe extern "C" fn() -> c_int;
type MemAlloc = unsafe extern "C" fn(usize, *mut u64, *mut *mut c_char, *mut usize) -> c_int;
type MemcpyH2D = unsafe extern "C" fn(u64, *const u8, usize, *mut *mut c_char, *mut usize) -> c_int;
type MemcpyD2H = unsafe extern "C" fn(*mut u8, u64, usize, *mut *mut c_char, *mut usize) -> c_int;
type Launch = unsafe extern "C" fn(*const u8, usize, *const u64, usize, *mut *mut c_char, *mut usize) -> c_int;
type Synchronize = unsafe extern "C" fn() -> c_int;

/// Read and clear the message a failed call left behind.
fn take_error(free: &Symbol<Free>, err: *mut c_char, err_len: usize) -> String {
    if err.is_null() {
        return String::from("<no message>");
    }
    let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned();
    unsafe { free(err as *mut u8, err_len) };
    msg
}

#[test]
fn a_copy_kernel_runs_on_the_simulator() {
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_TTSIM_LIB", &sim) };

    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let compile: Symbol<Compile> = unsafe { lib.get(b"tt_compile\0") }.expect("tt_compile");
    let free: Symbol<Free> = unsafe { lib.get(b"tt_free\0") }.expect("tt_free");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");
    let synchronize: Symbol<Synchronize> = unsafe { lib.get(b"ttSynchronize\0") }.expect("ttSynchronize");

    let (mut err, mut err_len) = (std::ptr::null_mut(), 0usize);
    assert_eq!(unsafe { init(&mut err, &mut err_len) }, 0, "ttInit: {}", take_error(&free, err, err_len));

    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(unsafe { alloc(8192, &mut input, &mut err, &mut err_len) }, 0, "input: {}", take_error(&free, err, err_len));
    assert_eq!(unsafe { alloc(8192, &mut output, &mut err, &mut err_len) }, 0, "output: {}", take_error(&free, err, err_len));
    assert_ne!(input, output, "two distinct DRAM allocations");

    let host: Vec<u8> = (0..8192u32).map(|i| (i.wrapping_mul(37) ^ (i >> 3)) as u8).collect();
    assert_eq!(
        unsafe { h2d(input, host.as_ptr(), host.len(), &mut err, &mut err_len) },
        0,
        "upload: {}",
        take_error(&free, err, err_len)
    );

    let (mut module, mut module_len) = (std::ptr::null_mut(), 0usize);
    let code = unsafe { compile(COPY_8K.as_ptr(), COPY_8K.len(), &mut module, &mut module_len, &mut err, &mut err_len) };
    assert_eq!(code, 0, "compile: {}", take_error(&free, err, err_len));

    let params = [input, output];
    let code = unsafe { launch(module, module_len, params.as_ptr(), params.len(), &mut err, &mut err_len) };
    assert_eq!(code, 0, "ttLaunch: {}", take_error(&free, err, err_len));
    assert_eq!(unsafe { synchronize() }, 0, "ttSynchronize");

    let mut back = vec![0u8; 8192];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err, &mut err_len) },
        0,
        "download: {}",
        take_error(&free, err, err_len)
    );
    assert_eq!(back, host, "the kernel moved every byte from DRAM through L1 and back");

    unsafe { free(module, module_len) };
    assert_eq!(unsafe { release() }, 0, "ttRelease");
}
