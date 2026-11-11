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
use std::sync::{Mutex, MutexGuard, OnceLock};

use libloading::{Library, Symbol};

/// One device at a time.
///
/// `libttsim` is a process-wide singleton and `ttInit` hands out DRAM and L1
/// addresses from a fresh base, so two device tests running in parallel would
/// free each other's allocations and stage into each other's L1. Every test that
/// opens the device holds this for its whole body.
fn device() -> MutexGuard<'static, ()> {
    static DEVICE: OnceLock<Mutex<()>> = OnceLock::new();
    DEVICE
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type AbiVersion = unsafe extern "C" fn() -> u32;
type Compile =
    unsafe extern "C" fn(*const c_char, usize, *mut *mut c_void, *mut *mut c_char) -> c_int;
type ModuleLoad =
    unsafe extern "C" fn(*const u8, usize, *mut *mut c_void, *mut *mut c_char) -> c_int;
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
    dir.parent()
        .expect("the profile directory")
        .join(format!("{}TT.so", std::env::consts::DLL_PREFIX))
}

/// The artifact, loaded once per test process.
///
/// A `dlclose` of `libTT.so` between two tests unloads the library's own
/// process-wide state, and `libttsim` — which stays resident — refuses a second
/// `libttsim_init`. Keeping one handle for the whole process is also how a real
/// consumer uses the `.so`.
fn library() -> &'static Library {
    static LIB: OnceLock<Library> = OnceLock::new();
    LIB.get_or_init(|| {
        let path = library_path();
        unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()))
    })
}

const COPY: &str = "\
version 3
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
copy_to %b0 from %in [0]
copy_to %out from %b0 [0]
";

/// Read and clear the message a failed call left behind.
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

#[test]
fn the_so_compiles_a_program_through_the_c_abi() {
    let lib = library();
    let version: Symbol<AbiVersion> = unsafe { lib.get(b"ttAbiVersion\0") }.expect("ttAbiVersion");
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    assert_eq!(unsafe { version() }, 2, "the ABI version");

    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut module, &mut err) };
    assert_eq!(code, TT_OK, "ttCompile: {}", take_error(&free, err));
    assert!(!module.is_null(), "a module handle came back");
    assert!(err.is_null(), "no error on success");
    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
}

#[test]
fn the_byte_form_compiles_through_the_c_abi() {
    let lib = library();
    let load: Symbol<ModuleLoad> = unsafe { lib.get(b"ttModuleLoad\0") }.expect("ttModuleLoad");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    // The test builds the byte form with the crate; the compile is the `.so`'s.
    let bytes = TT::ir::Kernel::parse(COPY)
        .expect("the fixture parses")
        .to_bytes();
    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { load(bytes.as_ptr(), bytes.len(), &mut module, &mut err) };
    assert_eq!(code, TT_OK, "ttModuleLoad: {}", take_error(&free, err));
    assert!(!module.is_null());
    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );

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
    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    let bad = "version 3\ngrid 1,1\n%out = param f32 1024\n%x = neg %y\n";
    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { compile(bad.as_ptr().cast(), bad.len(), &mut module, &mut err) };
    assert_eq!(
        code, TT_ERROR_PROGRAM,
        "a use before its definition is refused"
    );
    assert!(module.is_null() && !err.is_null());
    let msg = take_error(&free, err);
    assert!(msg.contains("%y"), "{msg}");
}

#[test]
fn the_error_codes_are_the_documented_ones() {
    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut module, &mut err) },
        TT_OK
    );

    // A handle that is not one of ours, and a live one unloaded twice.
    let bogus = 0x1000usize as *mut c_void;
    let params = [0u64, 0u64];
    assert_eq!(
        unsafe { launch(bogus, params.as_ptr(), 2, &mut err) },
        TT_ERROR_MODULE
    );
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(bogus, &mut err) }, TT_ERROR_MODULE);
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(module, &mut err) }, TT_OK);
    assert_eq!(unsafe { unload(module, &mut err) }, TT_ERROR_MODULE);
    let _ = take_error(&free, err);

    // A live handle with the wrong parameter count.
    let mut again: *mut c_void = std::ptr::null_mut();
    assert_eq!(
        unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut again, &mut err) },
        TT_OK
    );
    assert_eq!(
        unsafe { launch(again, params.as_ptr(), 1, &mut err) },
        TT_ERROR_ARGUMENT
    );
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(again, &mut err) }, TT_OK);

    // Argument checks that need no device.
    let mut addr = 0u64;
    assert_eq!(unsafe { alloc(0, &mut addr, &mut err) }, TT_ERROR_ARGUMENT);
    assert_eq!(addr, 0);
    let _ = take_error(&free, err);
    let host = [0u8; 4];
    assert_eq!(
        unsafe { h2d(0, std::ptr::null(), 4, &mut err) },
        TT_ERROR_ARGUMENT
    );
    let _ = take_error(&free, err);
    assert_eq!(
        unsafe { h2d(0, host.as_ptr(), 0, &mut err) },
        TT_ERROR_ARGUMENT
    );
    let _ = take_error(&free, err);
    unsafe { free(std::ptr::null_mut()) };
}

/// A copy kernel over a two-page buffer: DRAM into L1, and the same L1 buffer
/// back into DRAM, in 32x32 units.
const COPY_8K: &str = "\
version 3
grid 1,1
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,64
copy_to %b0 from %in [0]
copy_to %b0 from %in [1024]
copy_to %out from %b0 [0]
copy_to %out from %b0 [1024]
";

/// One 32x32 tile through the datapath: DRAM into L1, unpack, negate, pack,
/// L1 into DRAM. The program names no core, no `Dst` slot and no unpacker.
const NEG_TILE: &str = "\
version 3
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
copy_to %b0 from %in [0]
%x = load %b0 [0]
%y = neg %x
store_to %b0 from %y [0]
copy_to %out from %b0 [0]
";

/// The same elementwise kernel over two tiles, written as a loop. The bound and
/// the step are constants, so the loop is unrolled and each iteration's offset
/// becomes a literal.
const NEG_TWO_TILES: &str = "\
version 3
grid 1,1
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,64
%two = const i32 2
%step = const i32 1024
%i = loop %two
%o = mul %i %step
copy_to %b0 from %in [%o]
%x = load %b0 [%o]
%y = neg %x
store_to %b0 from %y [%o]
copy_to %out from %b0 [%o]
endloop
";

/// Where the simulator is. The library only reads `TT_TTSIM_LIB`, so the test
/// supplies it: the simulator is not part of this repository.
fn simulator_path() -> PathBuf {
    if let Some(p) = std::env::var_os("TT_TTSIM_LIB") {
        return PathBuf::from(p);
    }
    let mut candidates =
        vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/libttsim_bh.so")];
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
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_TTSIM_LIB", &sim) };

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

    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(8192, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(8192, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );
    assert_ne!(input, output, "two distinct DRAM allocations");

    let host: Vec<u8> = (0..8192u32)
        .map(|i| (i.wrapping_mul(37) ^ (i >> 3)) as u8)
        .collect();
    assert_eq!(
        unsafe { h2d(input, host.as_ptr(), host.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            COPY_8K.as_ptr().cast(),
            COPY_8K.len(),
            &mut module,
            &mut err,
        )
    };
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
    assert_eq!(
        back, host,
        "the kernel moved every byte from DRAM through L1 and back"
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
}

/// An elementwise kernel runs the whole datapath: the inbound mover, the three
/// compute cores and the outbound mover, all derived from one unified program.
#[test]
fn an_elementwise_kernel_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_TTSIM_LIB", &sim) };

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

    let bytes = 1024 * 4;
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(bytes, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(bytes, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    let host: Vec<f32> = (0..1024)
        .map(|i| {
            let v = (i as f32 - 512.0) * 0.25;
            if i % 7 == 0 { -v } else { v }
        })
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_TILE.as_ptr().cast(),
            NEG_TILE.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    assert!(!module.is_null());

    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

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
}

/// The loop form of the elementwise kernel runs too: the constant loop unrolls
/// into two literal offsets, and both tiles come back negated.
#[test]
fn an_unrolled_loop_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_TTSIM_LIB", &sim) };

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

    let count = 2048;
    let bytes = count * 4;
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(bytes, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(bytes, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    let host: Vec<f32> = (0..count)
        .map(|i| {
            let v = (i as f32 - 1024.0) * 0.125;
            if i % 5 == 0 { -v } else { v }
        })
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_TWO_TILES.as_ptr().cast(),
            NEG_TWO_TILES.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    assert!(!module.is_null());

    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

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
}

/// The unary and binary ops the SFPU has an encoding for, run through the C ABI:
/// `abs` on one loaded tile, and `add`/`sub`/`mul` on two.
#[test]
fn arithmetic_ops_run_on_the_simulator() {
    const TWO_LOAD: &str = "\
version 3
grid 1,1
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,64
copy_to %b0 from %in [0]
copy_to %b0 from %in [1024]
%x = load %b0 [0]
%y = load %b0 [1024]
%r = OP %x %y
store_to %b0 from %r [0]
copy_to %out from %b0 [0]
";
    const ONE_LOAD: &str = "\
version 3
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
copy_to %b0 from %in [0]
%x = load %b0 [0]
%r = abs %x
store_to %b0 from %r [0]
copy_to %out from %b0 [0]
";

    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_TTSIM_LIB", &sim) };

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

    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(8192, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(8192, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    let host: Vec<f32> = (0..2048)
        .map(|i| {
            let v = (i as f32 - 1024.0) * 0.125;
            if i % 3 == 0 { -v } else { v }
        })
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut run = |program: &str, expected: &dyn Fn(&[f32]) -> Vec<f32>, out_bytes: usize| {
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
        let params = [input, output];
        let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
        assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));
        let mut back = vec![0u8; out_bytes];
        assert_eq!(
            unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
            TT_OK,
            "download: {}",
            take_error(&free, err)
        );
        let got: Vec<f32> = back
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want = expected(&host);
        assert_eq!(got.len(), want.len());
        for (i, (x, y)) in want.iter().zip(&got).enumerate() {
            assert_eq!(*y, *x, "{program}: datum {i}: got {y}, want {x}");
        }
        assert_eq!(
            unsafe { unload(module, &mut err) },
            TT_OK,
            "unload: {}",
            take_error(&free, err)
        );
    };

    run(
        &ONE_LOAD.replace("OP", ""),
        &|h| h[..1024].iter().map(|v| v.abs()).collect(),
        4096,
    );
    for (op, f) in [
        (
            "add",
            (&|h: &[f32]| {
                h[..1024]
                    .iter()
                    .zip(&h[1024..])
                    .map(|(a, b)| a + b)
                    .collect()
            }) as &dyn Fn(&[f32]) -> Vec<f32>,
        ),
        ("sub", &|h: &[f32]| {
            h[..1024]
                .iter()
                .zip(&h[1024..])
                .map(|(a, b)| a - b)
                .collect()
        }),
        ("mul", &|h: &[f32]| {
            h[..1024]
                .iter()
                .zip(&h[1024..])
                .map(|(a, b)| a * b)
                .collect()
        }),
    ] {
        let program = TWO_LOAD.replace("OP", op);
        run(&program, f, 4096);
    }

    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}
