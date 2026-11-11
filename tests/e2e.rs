// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! End to end through the shipped artifact: load `libTT.so` with `dlopen` and
//! drive the C ABI.
//!
//! Every call goes through the loaded library; the crate itself is linked only
//! to build the byte-form fixture in one test. A compiled module is an opaque
//! handle here exactly as it is for a C consumer.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::PathBuf;

use libloading::{Library, Symbol};

type AbiVersion = unsafe extern "C" fn() -> u32;
type Compile = unsafe extern "C" fn(*const c_char, usize, *mut *mut c_void, *mut *mut c_char) -> c_int;
type ModuleLoad = unsafe extern "C" fn(*const u8, usize, *mut *mut c_void, *mut *mut c_char) -> c_int;
type ModuleUnload = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_int;
type Free = unsafe extern "C" fn(*mut c_void);
type Init = unsafe extern "C" fn(*mut *mut c_char) -> c_int;
type Release = unsafe extern "C" fn(*mut *mut c_char) -> c_int;
type MemAlloc = unsafe extern "C" fn(usize, *mut u64, *mut *mut c_char) -> c_int;
type MemcpyH2D = unsafe extern "C" fn(u64, *const u8, usize, *mut *mut c_char) -> c_int;
type MemcpyD2H = unsafe extern "C" fn(*mut u8, u64, usize, *mut *mut c_char) -> c_int;
type Launch = unsafe extern "C" fn(*mut c_void, *const u64, usize, *mut *mut c_char) -> c_int;

/// Error codes, as the header defines them.
const TT_OK: c_int = 0;
const TT_ERROR_ARGUMENT: c_int = 1;
const TT_ERROR_PROGRAM: c_int = 2;
const TT_ERROR_MODULE: c_int = 3;

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

/// Read and clear the message a failed call left behind.
fn take_error(free: &Symbol<Free>, err: *mut c_char) -> String {
    if err.is_null() {
        return String::from("<no message>");
    }
    let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned();
    unsafe { free(err as *mut c_void) };
    msg
}

#[test]
fn the_so_compiles_a_program_through_the_c_abi() {
    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let version: Symbol<AbiVersion> = unsafe { lib.get(b"ttAbiVersion\0") }.expect("ttAbiVersion");
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> = unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    assert_eq!(unsafe { version() }, 2, "the ABI version");

    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut module, &mut err) };
    assert_eq!(code, TT_OK, "ttCompile: {}", take_error(&free, err));
    assert!(!module.is_null(), "a module handle came back");
    assert!(err.is_null(), "no error on success");
    assert_eq!(unsafe { unload(module, &mut err) }, TT_OK, "unload: {}", take_error(&free, err));
}

#[test]
fn the_byte_form_compiles_through_the_c_abi() {
    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let load: Symbol<ModuleLoad> = unsafe { lib.get(b"ttModuleLoad\0") }.expect("ttModuleLoad");
    let unload: Symbol<ModuleUnload> = unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    // The test builds the byte form with the crate; the compile is the `.so`'s.
    let bytes = TT::ir::Kernel::parse(COPY).expect("the fixture parses").to_bytes();
    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { load(bytes.as_ptr(), bytes.len(), &mut module, &mut err) };
    assert_eq!(code, TT_OK, "ttModuleLoad: {}", take_error(&free, err));
    assert!(!module.is_null());
    assert_eq!(unsafe { unload(module, &mut err) }, TT_OK, "unload: {}", take_error(&free, err));

    // Text is the other entry point's input, and is refused here.
    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe { load(COPY.as_ptr(), COPY.len(), &mut module, &mut err) };
    assert_eq!(code, TT_ERROR_PROGRAM, "text is not the byte form");
    assert!(module.is_null());
    assert!(!err.is_null(), "there is a message");
    let _ = take_error(&free, err);
}

#[test]
fn a_refused_program_comes_back_as_an_error_not_a_crash() {
    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    let bad = "version 1\ngrid 1 1\nparam out f32 4096\nbuf 0 4096 plain rm\n%0 = sfpu neg %1\n";
    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { compile(bad.as_ptr().cast(), bad.len(), &mut module, &mut err) };
    assert_eq!(code, TT_ERROR_PROGRAM, "a use before its definition is refused");
    assert!(module.is_null() && !err.is_null());
    let msg = take_error(&free, err);
    assert!(msg.contains("%1"), "{msg}");
}

#[test]
fn the_error_codes_are_the_documented_ones() {
    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> = unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut module, &mut err) }, TT_OK);

    // A handle that is not one of ours, and a live one unloaded twice.
    let bogus = 0x1000usize as *mut c_void;
    let params = [0u64, 0u64];
    assert_eq!(unsafe { launch(bogus, params.as_ptr(), 2, &mut err) }, TT_ERROR_MODULE);
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(bogus, &mut err) }, TT_ERROR_MODULE);
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(module, &mut err) }, TT_OK);
    assert_eq!(unsafe { unload(module, &mut err) }, TT_ERROR_MODULE);
    let _ = take_error(&free, err);

    // A live handle with the wrong parameter count.
    let mut again: *mut c_void = std::ptr::null_mut();
    assert_eq!(unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut again, &mut err) }, TT_OK);
    assert_eq!(unsafe { launch(again, params.as_ptr(), 1, &mut err) }, TT_ERROR_ARGUMENT);
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(again, &mut err) }, TT_OK);

    // Argument checks that need no device.
    let mut addr = 0u64;
    assert_eq!(unsafe { alloc(0, &mut addr, &mut err) }, TT_ERROR_ARGUMENT);
    assert_eq!(addr, 0);
    let _ = take_error(&free, err);
    let host = [0u8; 4];
    assert_eq!(unsafe { h2d(0, std::ptr::null(), 4, &mut err) }, TT_ERROR_ARGUMENT);
    let _ = take_error(&free, err);
    assert_eq!(unsafe { h2d(0, host.as_ptr(), 0, &mut err) }, TT_ERROR_ARGUMENT);
    let _ = take_error(&free, err);
    unsafe { free(std::ptr::null_mut()) };
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

#[test]
fn a_copy_kernel_runs_on_the_simulator() {
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_TTSIM_LIB", &sim) };

    let path = library_path();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> = unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(unsafe { init(&mut err) }, TT_OK, "ttInit: {}", take_error(&free, err));

    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(unsafe { alloc(8192, &mut input, &mut err) }, TT_OK, "input: {}", take_error(&free, err));
    assert_eq!(unsafe { alloc(8192, &mut output, &mut err) }, TT_OK, "output: {}", take_error(&free, err));
    assert_ne!(input, output, "two distinct DRAM allocations");

    let host: Vec<u8> = (0..8192u32).map(|i| (i.wrapping_mul(37) ^ (i >> 3)) as u8).collect();
    assert_eq!(
        unsafe { h2d(input, host.as_ptr(), host.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe { compile(COPY_8K.as_ptr().cast(), COPY_8K.len(), &mut module, &mut err) };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    assert!(!module.is_null());

    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; 8192];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    assert_eq!(back, host, "the kernel moved every byte from DRAM through L1 and back");

    assert_eq!(unsafe { unload(module, &mut err) }, TT_OK, "unload: {}", take_error(&free, err));
    assert_eq!(unsafe { release(&mut err) }, TT_OK, "ttRelease: {}", take_error(&free, err));
}
